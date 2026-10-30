"""Q1: a holistic "authoritative tone" score from an LLM judge (pilot).

The judge reads one whole email, with names removed and no role shown, and
rates how authoritative its overall tone feels on a 1-7 scale. Unlike the
Layer 1 prompt in :mod:`thesis.analysis.role_inference`, the prompt names no
cues (orders, hedges, word choice), so the score is an overall impression.

Steps:

1. Draw a pilot sample: real emails from the Layer 1 set, and draw-1 replies
   from each writer's full grid, balanced by direction.
2. Reliability gate: score a subset three times. Stop if the mean pairwise
   quadratic-weighted kappa is below 0.4.
3. Score the rest once. Fit the direction contrasts used for the section 20
   measures: real email with a random intercept per sender and sender rank
   as a control, writers with persona fixed effects and persona-clustered
   errors.
4. Cue check: correlate the score with orders and hedges on real email.

Run with ``python -m thesis.analysis.tone_judge``.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Final

import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from thesis.analysis.hierarchy import fit_direction_fixed_effects, fit_direction_mixed_model
from thesis.analysis.judge_swap import client_for_model
from thesis.analysis.q1_real import load_bodies
from thesis.analysis.role_inference import (
    GENERATING_GRIDS,
    REAL_SAMPLE_SEED,
    build_absolute_items_from_grid,
    build_absolute_items_from_real_email,
    sample_real_directions,
)
from thesis.judge.prompt import JudgeItem, render_item_block
from thesis.llm.base import CompletionRequest, Message
from thesis.llm.cache import ResponseCache, cache_key
from thesis.llm.cost import CostLedger, LedgerEntry
from thesis.llm.ollama_client import OllamaClient, OllamaUnavailableError
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import CACHE_DIR, COST_LEDGER, INTERIM_DIR, MANIFESTS_DIR, ensure_dirs

log = get_logger(__name__)

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b@low"
MANIFEST_PATH = MANIFESTS_DIR / "tone_judge_pilot.json"
PILOT_SEED = 20261030
KAPPA_BAR = 0.4
SCALE: Final[tuple[int, ...]] = tuple(range(1, 8))

_TONE_FRAMING = (
    "You will be shown a single workplace email.\n\n"
    "Read the whole email once. How authoritative does its overall tone "
    "feel? Rate your overall impression of the email as a whole. Do not "
    "count or look for specific words.\n\n"
    "1 = defers to the reader: asks, checks, leaves the decision to them.\n"
    "7 = writes as the one in charge: decides, directs, expects it to be done.\n\n"
    "Describe the overall impression in a few words, then give the score."
)

TONE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "impression": {
            "type": "string",
            "description": "A few words on the overall tone of the email.",
        },
        "authority": {
            "type": "integer",
            "enum": list(SCALE),
            "description": "1 = fully deferential, 7 = fully in charge.",
        },
    },
    "required": ["impression", "authority"],
    "additionalProperties": False,
}


class InvalidToneResponseError(ValueError):
    """Raised when a tone response does not satisfy its shape."""


def validate_tone_response(payload: dict[str, Any]) -> int:
    score = payload.get("authority")
    if isinstance(score, bool) or not isinstance(score, int) or score not in SCALE:
        msg = f"authority {score!r} is not an integer from 1 to 7"
        raise InvalidToneResponseError(msg)
    return score


def build_tone_request(item: JudgeItem, model: str, *, replicate: int = 1) -> CompletionRequest:
    """One tone call. ``replicate`` is the cache-key draw index, so repeated
    passes on the same item get their own cache entries."""
    return CompletionRequest(
        model=model,
        messages=[Message(role="user", content=render_item_block(item))],
        max_tokens=512,
        system=_TONE_FRAMING,
        output_schema=TONE_SCHEMA,
        cache_system=True,
        variant=replicate,
        metadata={"item_id": item.item_id, "task": "tone_absolute"},
    )


@dataclass
class ToneSummary:
    n_scored: int = 0
    n_invalid: int = 0
    n_from_cache: int = 0


def score_tone(
    items: list[JudgeItem],
    client: Any,
    *,
    model: str,
    cache: ResponseCache,
    ledger: CostLedger,
    replicate: int = 1,
) -> tuple[dict[str, int], ToneSummary]:
    """Score every item, cache first. Returns {item_id: score}."""
    scores: dict[str, int] = {}
    summary = ToneSummary()
    for item in items:
        request = build_tone_request(item, model, replicate=replicate)
        key = cache_key(request, client.provider)
        response = cache.get(key)
        if response is None:
            response = client.complete(request)
            cache.put(key, request, response, client.provider)
        else:
            summary.n_from_cache += 1
        try:
            if response.parsed is None:
                raise InvalidToneResponseError("no parseable structured output")
            scores[item.item_id] = validate_tone_response(response.parsed)
        except InvalidToneResponseError as exc:
            log.warning("item %s: %s", item.item_id, exc)
            summary.n_invalid += 1
            continue
        summary.n_scored += 1
        ledger.record(
            LedgerEntry(
                run_id=f"tone_judge-pass{replicate}",
                provider=client.provider,
                model=response.model,
                call_kind="tone_absolute",
                usage=response.usage,
                # Only free services are used, so no call is billed.
                from_cache=True,
            )
        )
        if summary.n_scored % 50 == 0:
            log.info("pass %d: %d of %d scored", replicate, summary.n_scored, len(items))
    return scores, summary


def weighted_kappa(passes: list[dict[str, int]]) -> tuple[float, int]:
    """Mean pairwise quadratic-weighted kappa over items scored on every pass."""
    ids = sorted(set.intersection(*(set(p) for p in passes)))
    kappas = [
        cohen_kappa_score(
            [a[i] for i in ids], [b[i] for i in ids], labels=list(SCALE), weights="quadratic"
        )
        for a, b in combinations(passes, 2)
    ]
    return round(float(sum(kappas) / len(kappas)), 4), len(ids)


def _contrasts(result: Any) -> dict[str, dict[str, float]]:
    out = {}
    for level in ("down", "up"):
        coefficient, p_value = result.contrast(level)
        out[level] = {"coefficient": round(coefficient, 4), "p": round(p_value, 6)}
    return out


def balanced_sample(frame: pd.DataFrame, *, per_direction: int, seed: int) -> pd.DataFrame:
    """The same number of rows from each direction, fixed seed."""
    return frame.groupby("direction", group_keys=False).sample(n=per_direction, random_state=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--per-direction", type=int, default=67)
    parser.add_argument("--gate-per-source", type=int, default=20)
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()
    client = client_for_model(args.judge_model)
    if isinstance(client, OllamaClient) and not client.is_available():
        raise OllamaUnavailableError(f"no Ollama server reachable for {args.judge_model}")
    cache = ResponseCache(CACHE_DIR)
    ledger = CostLedger(COST_LEDGER)

    real_emails = pd.read_parquet(INTERIM_DIR / "q1_real_emails.parquet").drop_duplicates(
        subset="message_uid"
    )
    layer1 = sample_real_directions(
        real_emails[["message_uid", "direction"]], n=1000, seed=REAL_SAMPLE_SEED
    )
    real_pick = balanced_sample(layer1, per_direction=args.per_direction, seed=PILOT_SEED)
    real_items, _ = build_absolute_items_from_real_email(
        load_bodies(real_pick["message_uid"]), real_pick
    )
    sources: dict[str, list[JudgeItem]] = {"real email": real_items}
    grids: dict[str, pd.DataFrame] = {}
    for label, path in GENERATING_GRIDS.items():
        grid = pd.read_parquet(path)
        grid = grid[grid["replicate"] == 1]
        grids[label] = grid
        pick = balanced_sample(grid, per_direction=args.per_direction, seed=PILOT_SEED)
        sources[label], _ = build_absolute_items_from_grid(pick)

    # Writers share cell ids, so gate answers are keyed by source too.
    rng = random.Random(PILOT_SEED)
    gate = {label: rng.sample(items, args.gate_per_source) for label, items in sources.items()}
    passes = []
    for replicate in (1, 2, 3):
        scores: dict[str, int] = {}
        for label, items in gate.items():
            by_item, summary = score_tone(
                items,
                client,
                model=args.judge_model,
                cache=cache,
                ledger=ledger,
                replicate=replicate,
            )
            log.info("gate pass %d, %s: %s", replicate, label, summary)
            scores |= {f"{label}|{item_id}": score for item_id, score in by_item.items()}
        passes.append(scores)
    kappa, n_gate = weighted_kappa(passes)
    log.info("tone self-consistency: weighted kappa %.3f on %d items", kappa, n_gate)
    manifest: dict[str, Any] = {
        "judge": args.judge_model,
        "gate": {"weighted_kappa": kappa, "n_items": n_gate, "bar": KAPPA_BAR},
    }
    if kappa < KAPPA_BAR:
        log.error("tone self-consistency below %.1f; stopping before the pilot", KAPPA_BAR)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return

    results: dict[str, Any] = {}
    for label, items in sources.items():
        scores, summary = score_tone(
            items, client, model=args.judge_model, cache=cache, ledger=ledger
        )
        log.info("%s: %s", label, summary)
        frame = pd.DataFrame({"item_id": list(scores), "tone": list(scores.values())})
        entry: dict[str, Any] = {"n": len(frame), "n_invalid": summary.n_invalid}
        if label == "real email":
            frame = frame.merge(real_emails, left_on="item_id", right_on="message_uid")
            entry["mixed"] = _contrasts(
                fit_direction_mixed_model(
                    frame, "tone", cluster_col="sender_id", covariates=("sender_rank_level",)
                )
            )
            entry["fixed_effects"] = _contrasts(
                fit_direction_fixed_effects(frame, "tone", cluster_col="sender_id")
            )
            entry["cue_spearman"] = {
                cue: round(float(stats.spearmanr(frame["tone"], frame[cue]).statistic), 4)
                for cue in ("imperative_ratio", "hedge_rate", "question_ratio")
            }
        else:
            frame = frame.merge(grids[label], left_on="item_id", right_on="cell_id")
            entry["fixed_effects"] = _contrasts(fit_direction_fixed_effects(frame, "tone"))
        entry["mean_by_direction"] = {
            str(k): round(float(v), 3) for k, v in frame.groupby("direction")["tone"].mean().items()
        }
        results[label] = entry
    manifest["results"] = results
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
