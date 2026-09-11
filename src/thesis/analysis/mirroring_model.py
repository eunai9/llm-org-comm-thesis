"""Ask a model the mirroring question directly, then check whether it works.

Section 42 measured mirroring by counting words. ``borrowed_words`` is the
share of the reply's content words that already appeared in the incoming
message. It scores 0.834 against the 100 hand codes. Section 44 tried to beat
that with embeddings and every variant scored worse.

One idea was left. Show a model the incoming message and the reply, and ask the
narrow question: does this reply hand the sender's own request back? This
module does that.

**It is validated before it is used.** Section 35 showed the judge gets more
generous when it sees the incoming message, without getting any better at
spotting mirroring. Every rubric score rose by 0.4 to 0.7 points and the gap
between mirrored and sound replies stayed inside noise. So a model shown more
context is not automatically a sharper instrument, and this check has to earn
its place against the same 100 hand codes ``mirroring.validate`` uses.

**The model call lives here, not in mirroring.py.** That module states it calls
no model, and it runs in seconds over every reply the project has. Keeping the
two apart keeps that true.

Three design choices, each for one reason:

- **A 1 to 5 scale, not yes/no.** AUC needs a ranked score. A binary answer
  gives one point on the ROC curve and throws away every ordering the model
  could express.
- **Evidence before score**, in the schema's own property order. Same
  convention as :mod:`thesis.judge.rubric`, for the same calibration reason.
- **Three draws per item**, each with its own ``CompletionRequest.variant`` so
  the cache stores them apart. Without that the cache serves draw 1 three times
  and the measured spread is zero by construction. The spread across draws is
  reported as its own number: a check that disagrees with itself cannot
  separate anything.

Local models only. No paid API.

Run with ``python -m thesis.analysis.mirroring_model``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from thesis.llm.base import CompletionRequest, CompletionResponse, Message, Provider
from thesis.llm.cache import ResponseCache, cache_key
from thesis.llm.cost import CostLedger, LedgerEntry, cost_usd
from thesis.llm.ollama_client import is_local_model
from thesis.llm.stub_client import is_stub_model
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import (
    CACHE_DIR,
    COST_LEDGER,
    MANIFESTS_DIR,
    TABLES_DIR,
    ensure_dirs,
)

log = get_logger(__name__)

DEFAULT_MODEL = "qwen2.5:3b"
CODED_PATH: Path = TABLES_DIR / "manual_review_coded_first_pass.csv"

# The number to beat. Section 42's lexical measure, on these same 100 items.
LEXICAL_AUC = 0.834

SCORE_MIN = 1
SCORE_MAX = 5
SCORE_VALUES: tuple[int, ...] = tuple(range(SCORE_MIN, SCORE_MAX + 1))

# Three draws for the validation pass, one for scoring a full run. Three is
# enough to see whether the model agrees with itself and cheap enough to run
# 100 items twice on a laptop.
DEFAULT_DRAWS = 3

# How many words of the incoming message the model is shown. Some stimuli run
# past a thousand words, and a 3B model with a 4k context window loses the
# question at the end of a long prompt.
MAX_STIMULUS_WORDS = 400

_ANCHORS: tuple[str, ...] = (
    "1 = The reply acts on the request. It answers it, decides it, does it, "
    "or says what the writer will do.",
    "2 = The reply mostly acts on the request. It also asks for one detail it "
    "genuinely needs first.",
    "3 = Unclear. The reply neither clearly acts nor clearly hands the task "
    "back, or it does some of each.",
    "4 = The reply mostly hands the task back. It gives little or nothing and "
    "asks the sender to do most of the work.",
    "5 = The reply hands the sender's own request straight back. It asks the "
    "sender to do, confirm, decide or send the very thing the sender asked "
    "for.",
)

_SYSTEM_PROMPT = "\n".join(
    [
        "You check one workplace email reply for one specific fault.",
        "",
        "The fault: the reply hands the sender's own request back instead of "
        "acting on it. The reply is fluent, polite and on topic. It still "
        "leaves the work with the person who asked.",
        "",
        "Two worked examples. The sender writes: 'Please book the meeting " "room for Thursday.'",
        "- The reply 'Can you book the room for Thursday?' is the fault. " "Score 5.",
        "- The reply 'Booked. Room 12B, Thursday at 9.' is not the fault. " "Score 1.",
        "",
        "Rate only this fault. Ignore grammar, tone, length and politeness. A "
        "blunt reply that acts is not the fault. A polite reply that hands "
        "the task back is.",
        "",
        "Asking a question is not the fault by itself. A reply that acts and "
        "then asks something new is acting. The fault is giving back the same "
        "task.",
    ]
)

# Restated in the user turn, right after the two messages. A 3B model reads
# the end of its prompt best, and the scale has to be the last thing it sees
# before it answers.
_TASK_BLOCK = "\n".join(
    [
        "## Your task",
        "",
        "Does this reply act on what the sender asked for, or does it hand " "the same task back?",
        "",
        *_ANCHORS,
        "",
        "Quote a short piece of the reply, then give the score.",
    ]
)


def truncate_words(text: str, limit: int = MAX_STIMULUS_WORDS) -> str:
    """Cut text to at most ``limit`` words."""
    words = str(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit])


@dataclass(frozen=True, slots=True)
class MirroringItem:
    """One reply, with the message it answers."""

    item_id: str
    stimulus: str
    reply: str


def render_item_block(item: MirroringItem) -> str:
    """The variable part of the prompt: the two messages, and nothing else.

    No item id and no provenance. The model is never told a reply was
    generated, for the same blinding reason :mod:`thesis.judge.prompt` gives.
    """
    return (
        "## The message that came in\n\n"
        f"{truncate_words(item.stimulus)}\n\n"
        "## The reply to rate\n\n"
        f"{item.reply}\n\n"
        f"{_TASK_BLOCK}"
    )


def build_schema() -> dict[str, Any]:
    """The structured-output shape: evidence first, then an integer score.

    ``score`` is an enum rather than a minimum/maximum pair. This project
    already found that structured-output APIs do not reliably enforce numeric
    bounds, so a range would be hopeful and an enum is exact.
    """
    return {
        "type": "object",
        "properties": {
            "evidence": {
                "type": "string",
                "description": (
                    "A short quote from the reply showing whether it acts on "
                    "the request or hands it back."
                ),
            },
            "score": {
                "type": "integer",
                "enum": list(SCORE_VALUES),
                "description": " ".join(_ANCHORS),
            },
        },
        "required": ["evidence", "score"],
        "additionalProperties": False,
    }


class InvalidMirroringResponseError(ValueError):
    """Raised when a response does not have the shape the check needs."""


def validate_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Check a decoded response and return it unchanged.

    A second line of defence behind the model's own schema enforcement, the
    same role :func:`thesis.judge.rubric.validate_judge_response` plays. A 3B
    model drifts from a schema often enough that this is not theoretical.
    """
    for key in ("evidence", "score"):
        if key not in payload:
            msg = f"response missing {key!r}"
            raise InvalidMirroringResponseError(msg)
    if payload["score"] not in SCORE_VALUES:
        msg = f"score {payload['score']!r} not in {SCORE_VALUES}"
        raise InvalidMirroringResponseError(msg)
    if not isinstance(payload["evidence"], str):
        msg = "evidence must be a string"
        raise InvalidMirroringResponseError(msg)
    return payload


def build_request(item: MirroringItem, model: str, *, draw: int = 0) -> CompletionRequest:
    """Assemble one call.

    ``draw`` becomes the request's ``variant`` field, which is part of the
    cache key. Without it every draw of the same item collides on one cache
    entry and the three answers are the same answer three times.
    """
    return CompletionRequest(
        model=model,
        messages=[Message(role="user", content=render_item_block(item))],
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        output_schema=build_schema(),
        cache_system=True,
        variant=draw,
        metadata={"item_id": item.item_id, "draw": str(draw)},
    )


class _CompletionClient(Protocol):
    """The two capabilities this pass needs, and nothing else."""

    provider: Provider

    def complete(self, request: CompletionRequest) -> CompletionResponse: ...


@dataclass(frozen=True, slots=True)
class MirroringScore:
    """One reply's score, averaged over its draws."""

    item_id: str
    score: float
    draws: tuple[int, ...]
    evidence: str
    spread: float


@dataclass
class ScoringSummary:
    """What actually happened, rather than what was requested."""

    n_requested: int = 0
    n_scored: int = 0
    n_invalid: int = 0
    n_from_cache: int = 0
    total_cost_usd: float = 0.0


def score_items(
    items: Sequence[MirroringItem],
    client: _CompletionClient,
    *,
    model: str,
    cache: ResponseCache,
    ledger: CostLedger,
    run_id: str,
    draws: int = DEFAULT_DRAWS,
    log_every: int = 20,
) -> tuple[list[MirroringScore], ScoringSummary]:
    """Score every item, cache-first, and average the draws.

    Follows :func:`thesis.judge.run.score_items`: cache lookup before any call,
    a validated structured response, one ledger entry per call, and an invalid
    response logged and skipped rather than raised. One malformed answer must
    not discard the other 99 items.

    An item keeps whatever draws came back valid. It is dropped only when all
    of them fail.
    """
    summary = ScoringSummary(n_requested=len(items))
    results: list[MirroringScore] = []

    for position, item in enumerate(items, start=1):
        values: list[int] = []
        evidence = ""
        for draw in range(draws):
            request = build_request(item, model, draw=draw)
            key = cache_key(request, client.provider)

            response = cache.get(key)
            if response is None:
                response = client.complete(request)
                cache.put(key, request, response, client.provider)
            else:
                summary.n_from_cache += 1

            _record_cost(response, client.provider, ledger, run_id, summary)

            if response.parsed is None:
                log.warning("item %s draw %d: no parseable output", item.item_id, draw)
                summary.n_invalid += 1
                continue
            try:
                payload = validate_response(response.parsed)
            except InvalidMirroringResponseError as exc:
                log.warning("item %s draw %d: %s", item.item_id, draw, exc)
                summary.n_invalid += 1
                continue

            values.append(int(payload["score"]))
            evidence = evidence or str(payload["evidence"])

        if not values:
            log.warning("item %s: no valid draw", item.item_id)
            continue

        results.append(
            MirroringScore(
                item_id=item.item_id,
                score=round(sum(values) / len(values), 4),
                draws=tuple(values),
                evidence=evidence,
                spread=float(max(values) - min(values)),
            )
        )
        summary.n_scored += 1
        if log_every and position % log_every == 0:
            log.info("scored %d/%d items", position, len(items))

    return results, summary


def _record_cost(
    response: CompletionResponse,
    provider: Provider,
    ledger: CostLedger,
    run_id: str,
    summary: ScoringSummary,
) -> None:
    """One ledger entry per call. A local model is priced at zero."""
    billable = (
        not response.from_cache
        and not is_stub_model(response.model)
        and not is_local_model(response.model)
    )
    summary.total_cost_usd += cost_usd(response.model, response.usage) if billable else 0.0
    ledger.record(
        LedgerEntry(
            run_id=run_id,
            provider=provider,
            model=response.model,
            call_kind="mirroring_check",
            usage=response.usage,
            from_cache=response.from_cache or not billable,
        )
    )


def self_consistency(results: Sequence[MirroringScore]) -> dict[str, float]:
    """How much the draws for one item disagree with each other.

    Three numbers, because one would hide the shape of the disagreement:
    the share of items where all draws agree exactly, the mean spread between
    the highest and lowest draw, and the share where the spread is at most one
    point. A check whose draws wander over the whole 1 to 5 scale is measuring
    its own noise.
    """
    graded = [result for result in results if len(result.draws) > 1]
    if not graded:
        return {"n": 0, "exact_agreement": 0.0, "mean_spread": 0.0, "within_one_point": 0.0}
    spreads = [result.spread for result in graded]
    return {
        "n": len(graded),
        "exact_agreement": round(sum(s == 0 for s in spreads) / len(spreads), 3),
        "mean_spread": round(sum(spreads) / len(spreads), 3),
        "within_one_point": round(sum(s <= 1 for s in spreads) / len(spreads), 3),
    }


def auc_against(scores: Sequence[float], is_mirrored: Sequence[bool]) -> float:
    """How well the scores separate the hand-coded mirrored replies from the rest."""
    labels = [int(value) for value in is_mirrored]
    return round(float(roc_auc_score(labels, list(scores))), 3)


def build_items(frame: pd.DataFrame) -> list[MirroringItem]:
    """One item per coded reply, keyed by the coding sheet's own item number."""
    return [
        MirroringItem(
            item_id=str(row.item),
            stimulus=str(row.stimulus_text),
            reply=str(row.generated_reply),
        )
        for row in frame.itertuples()
    ]


def validate_against_codes(
    frame: pd.DataFrame,
    client: _CompletionClient,
    *,
    model: str,
    draws: int = DEFAULT_DRAWS,
    lexical: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score the hand-coded replies and report how well the check finds mirroring.

    ``lexical`` is the section 42 measure on the same rows, when it is
    available. It is used for one extra number: whether the model check
    separates the replies the lexical measure misses, which is where section 44
    found its own attempt failed hardest.
    """
    cache = ResponseCache(CACHE_DIR)
    ledger = CostLedger(COST_LEDGER)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    results, scoring = score_items(
        build_items(frame),
        client,
        model=model,
        cache=cache,
        ledger=ledger,
        run_id=run_id,
        draws=draws,
    )
    log.info(
        "%s: %d scored, %d invalid draws, %d from cache",
        model,
        scoring.n_scored,
        scoring.n_invalid,
        scoring.n_from_cache,
    )

    by_id = {result.item_id: result for result in results}
    rows = []
    for position, row in enumerate(frame.itertuples()):
        result = by_id.get(str(row.item))
        if result is None:
            continue
        rows.append(
            {
                "item": int(row.item),
                "model": model,
                "model_score": result.score,
                "draws": ",".join(str(value) for value in result.draws),
                "spread": result.spread,
                "evidence": result.evidence,
                "mirrored": bool(row.failure_mode == "mirrors_request"),
                "lexical": float(lexical[position]) if lexical is not None else float("nan"),
            }
        )
    scored = pd.DataFrame.from_records(rows)

    summary: dict[str, object] = {
        "model": model,
        "draws_per_item": draws,
        "n_items": len(scored),
        "n_mirrored": int(scored["mirrored"].sum()),
        "n_invalid_draws": scoring.n_invalid,
        "auc": auc_against(scored["model_score"], scored["mirrored"]),
        "lexical_auc_for_reference": LEXICAL_AUC,
        "self_consistency": self_consistency(results),
        "mean_score": {
            "coded_as_mirroring": round(
                float(scored.loc[scored["mirrored"], "model_score"].mean()), 3
            ),
            "coded_sound": round(float(scored.loc[~scored["mirrored"], "model_score"].mean()), 3),
        },
        "cost_usd": round(scoring.total_cost_usd, 4),
    }
    if lexical is not None:
        summary["on_the_replies_the_lexical_measure_misses"] = _lexical_misses(scored)
    return scored, summary


def _lexical_misses(scored: pd.DataFrame, threshold: float = 0.80) -> dict[str, object]:
    """The mirrored replies the lexical measure scores below its own cut-off.

    Section 44 checked exactly this subset and found no separation at all: its
    semantic signal gave those replies 0.398 against 0.383 for sound replies.
    The same check applies here, and it is the decisive one. A measure that
    only agrees with ``borrowed_words`` where ``borrowed_words`` already works
    adds nothing.
    """
    missed = scored[scored["mirrored"] & (scored["lexical"] < threshold)]
    sound = scored[~scored["mirrored"]]
    gap = float(missed["model_score"].mean() - sound["model_score"].mean())
    return {
        "n_missed": len(missed),
        "lexical_threshold": threshold,
        "mean_model_score_on_missed": round(float(missed["model_score"].mean()), 3),
        "mean_model_score_on_sound": round(float(sound["model_score"].mean()), 3),
        # Reported as a gap, not as a verdict. A field named "separates_them"
        # would read as a result, and a tenth of a point on seven items out of
        # a five-point scale is not one.
        "mean_gap": round(gap, 3),
    }


@dataclass(frozen=True, slots=True)
class RunComparison:
    """What a prompt change did to the model check, on the same stimuli."""

    n_paired: int
    before: float
    after: float
    change: float
    p_value: float


def compare_runs(
    before: pd.DataFrame,
    after: pd.DataFrame,
    client: _CompletionClient,
    *,
    model: str,
    draws: int = 1,
) -> RunComparison:
    """Did the act instruction move the model check, on the same stimuli?

    Paired by ``cell_id``, the same way :func:`thesis.analysis.mirroring.compare_runs`
    pairs, so the test is a signed-rank on the per-reply difference rather than
    a two-sample test that throws the pairing away.

    One draw per item by default. Three draws on 366 replies is over a thousand
    local calls, and the averaging matters for ranking 100 items against hand
    codes, not for a paired mean.
    """
    cache = ResponseCache(CACHE_DIR)
    ledger = CostLedger(COST_LEDGER)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    merged = before.merge(after, on="cell_id", suffixes=("_before", "_after"))
    if merged.empty:
        msg = "no cell_id appears in both runs; the two files are not the same design"
        raise ValueError(msg)

    scores: dict[str, pd.Series] = {}
    for arm in ("before", "after"):
        items = [
            MirroringItem(
                item_id=f"{arm}_{row.cell_id}",
                stimulus=str(getattr(row, f"stimulus_text_{arm}")),
                reply=str(getattr(row, f"generated_reply_{arm}")),
            )
            for row in merged.itertuples()
        ]
        results, _ = score_items(
            items,
            client,
            model=model,
            cache=cache,
            ledger=ledger,
            run_id=run_id,
            draws=draws,
        )
        by_id = {result.item_id: result.score for result in results}
        scores[arm] = pd.Series(
            [by_id.get(f"{arm}_{cell_id}", float("nan")) for cell_id in merged["cell_id"]]
        )

    paired = pd.DataFrame(scores).dropna()
    difference = paired["after"] - paired["before"]
    # Every reply scoring identically would make the signed-rank test
    # undefined. That is a real outcome, not an error.
    p_value = (
        float(stats.wilcoxon(paired["after"], paired["before"]).pvalue) if difference.any() else 1.0
    )
    return RunComparison(
        n_paired=len(paired),
        before=round(float(paired["before"].mean()), 3),
        after=round(float(paired["after"].mean()), 3),
        change=round(float(difference.mean()), 3),
        p_value=round(p_value, 4),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coded", default=str(CODED_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the per-item scores. Defaults to one file per model.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Where to write the summary. Defaults to one file per model.",
    )
    parser.add_argument(
        "--no-lexical",
        action="store_true",
        help="Skip the lexical measure. It needs spaCy and is only used for comparison.",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    from thesis.llm.ollama_client import OllamaClient, OllamaUnavailableError

    client = OllamaClient(args.model)
    if not client.is_available():
        msg = "no Ollama server reachable; start it with 'ollama serve'."
        raise OllamaUnavailableError(msg)

    frame = pd.read_csv(args.coded)
    lexical = None
    if not args.no_lexical:
        from thesis.analysis.mirroring import HEADLINE_SIGNAL, score_texts

        lexical = score_texts(frame["stimulus_text"], frame["generated_reply"])[
            HEADLINE_SIGNAL
        ].tolist()

    scored, summary = validate_against_codes(
        frame, client, model=args.model, draws=args.draws, lexical=lexical
    )

    slug = args.model.replace(":", "_").replace("/", "_")
    out = Path(args.out or TABLES_DIR / f"mirroring_model_scores_{slug}.csv")
    manifest = Path(args.manifest or MANIFESTS_DIR / f"mirroring_model_{slug}.json")
    scored.to_csv(out, index=False)
    manifest.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log.info("wrote %s and %s", out, manifest)
    log.info("mirroring model summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
