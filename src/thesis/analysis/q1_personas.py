"""The second Q1 experiment: 40 persona variants writing to named receivers.

Generation goes through :func:`thesis.sim.run.run_grid`, the same path as
every other grid. The analysis mode reads a saved grid and calls no model.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import statsmodels.formula.api as smf

from thesis.analysis.hierarchy import (
    FixedEffectsResult,
    InsufficientDataError,
    fit_direction_fixed_effects,
)
from thesis.analysis.judge_swap import client_for_model
from thesis.analysis.q1 import (
    Q1Design,
    extract_q1_sentence_features,
    parse_replies,
    scenarios_for_design,
)
from thesis.analysis.q1_measures import GILBERT_PATH, load_gilbert_phrases
from thesis.analysis.role_cues import reply_cues
from thesis.config import load_config
from thesis.llm.base import LLMClient
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import CACHE_DIR, COST_LEDGER, ensure_dirs
from thesis.sim.memory import MemoryItem
from thesis.sim.memory_generation import load_frozen_memory
from thesis.sim.persona import Persona, load_frozen_personas
from thesis.sim.persona_variants import (
    RECEIVER_STEP,
    build_variant_cells,
    build_variants,
    persona_grid_prompt_hash,
    variant_stores,
    variant_traits,
)
from thesis.sim.run import RunManifest, run_grid

log = get_logger(__name__)

ROLE_LABEL = "sim_q1_personas"
REPLY_OUTCOMES: tuple[str, ...] = (
    "imperative_ratio",
    "hedge_rate",
    "you_rate",
    "gratitude",
    "gilbert_score",
)


def generate_persona_grid(
    client: LLMClient,
    *,
    model: str,
    design: Q1Design,
    personas: Sequence[Persona] | None = None,
    stores: Mapping[str, Sequence[MemoryItem]] | None = None,
    cache: ResponseCache | None = None,
    ledger: CostLedger | None = None,
    limit: int | None = None,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Generate (or load from cache) one model's persona-variant grid."""
    base = list(personas) if personas is not None else load_frozen_personas()
    stores = stores if stores is not None else load_frozen_memory()
    cache = cache if cache is not None else ResponseCache(CACHE_DIR)
    ledger = ledger if ledger is not None else CostLedger(COST_LEDGER)
    rank_labels = {p.seniority_rank: p.rank_label for p in base}
    variants = build_variants(base)
    cells = build_variant_cells(
        variants,
        scenarios_for_design(design),
        model=model,
        role_label=ROLE_LABEL,
        rank_labels=rank_labels,
    )
    if limit is not None:
        cells = cells[:limit]
    prompt_hash = persona_grid_prompt_hash()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = RunManifest(
        run_id=run_id,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit="",
        git_dirty=False,
        config_hash="",
        models=[model],
        design={"kind": "q1_persona_grid", "design": design, "prompt_text_hash": prompt_hash},
        n_cells=len(cells),
    )
    rows: list[dict[str, Any]] = run_grid(
        cells,
        client,
        variant_stores(stores, variants),
        load_config(),
        run_id=run_id,
        cache=cache,
        ledger=ledger,
        manifest=manifest,
        progress_every=progress_every,
    )
    if not rows:
        msg = f"{model}: 0 of {len(cells)} cells returned a valid reply"
        raise RuntimeError(msg)
    frame = pd.DataFrame.from_records(rows)
    traits = frame["persona_id"].map(variant_traits)
    frame["base_persona_id"] = traits.str[0]
    frame["tenure"] = traits.str[1]
    frame["origin"] = traits.str[2]
    frame["receiver_rank"] = frame["seniority_rank"] + frame["direction"].map(RECEIVER_STEP)
    frame["prompt_text_hash"] = prompt_hash
    log.info("%s: %d of %d cells returned a valid reply", model, len(frame), len(cells))
    return frame


def _contrasts(fit: FixedEffectsResult) -> dict[str, dict[str, float]]:
    out = {}
    for level in ("down", "up"):
        if f"direction[T.{level}]" not in fit.coefficients:
            continue
        coefficient, p = fit.contrast(level)
        se = next(v for k, v in fit.std_errors.items() if k.endswith(f"[T.{level}]"))
        out[level] = {"coefficient": round(coefficient, 4), "se": round(se, 4), "p": round(p, 6)}
    return out


def analyse_persona_grid(
    frame: pd.DataFrame, phrases: Mapping[tuple[str, ...], float]
) -> dict[str, Any]:
    """Direction contrasts with persona fixed effects, clustered by persona.

    A grid without persona variants (the 10-persona design) has no tenure or
    origin, so its trait check is skipped.
    """
    docs = parse_replies(frame)
    sentences = extract_q1_sentence_features(frame, docs=docs)
    primary = fit_direction_fixed_effects(sentences, "is_imperative", family="logistic")

    cues = pd.DataFrame(
        [
            reply_cues(body, docs[cid], phrases)
            for cid, body in zip(frame["cell_id"], frame["body"], strict=True)
        ]
    )
    has_traits = {"tenure", "origin", "base_persona_id"} <= set(frame.columns)
    id_cols = ["persona_id", "direction"]
    if has_traits:
        id_cols += ["tenure", "origin", "base_persona_id"]
    reply = pd.concat([frame[id_cols].reset_index(drop=True), cues], axis=1)
    reply_level: dict[str, Any] = {}
    for outcome in REPLY_OUTCOMES:
        if reply[outcome].std() == 0:
            continue
        try:
            reply_level[outcome] = _contrasts(fit_direction_fixed_effects(reply, outcome))
        except InsufficientDataError:
            continue

    traits = {}
    if has_traits:
        working = reply.assign(tenure=reply["tenure"].astype(str))
        trait_fit = smf.ols(
            "imperative_ratio ~ C(direction, Treatment(reference='lateral'))"
            " + C(tenure) + C(origin) + C(base_persona_id)",
            data=working,
        ).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(working["persona_id"])[0]})
        for name in ("tenure", "origin"):
            term = next(t for t in trait_fit.params.index if t.startswith(f"C({name})"))
            traits[name] = {
                "term": term,
                "coefficient": round(float(trait_fit.params[term]), 4),
                "p": round(float(trait_fit.pvalues[term]), 6),
            }

    return {
        "n_replies": len(frame),
        "n_personas": int(frame["persona_id"].nunique()),
        "directions": sorted(set(frame["direction"]) - {"lateral"}),
        "primary": _contrasts(primary),
        "reply_level": reply_level,
        "traits": traits,
        "mean_sentences_per_reply": round(float(len(sentences) / max(len(frame), 1)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", help="e.g. qwen/qwen3.8-27b or llama3.2:3b")
    mode.add_argument("--analyse", type=Path, help="saved grid to analyse; calls no model")
    parser.add_argument("--manifest", type=Path, help="JSON output for --analyse")
    parser.add_argument("--design", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--ollama-host", default=None)
    args = parser.parse_args()
    configure_logging()
    if args.analyse is not None:
        if args.manifest is None:
            parser.error("--analyse needs --manifest")
        result = analyse_persona_grid(
            pd.read_parquet(args.analyse), load_gilbert_phrases(GILBERT_PATH)
        )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("wrote %s", args.manifest)
        return
    if args.out is None:
        parser.error("--model needs --out")
    ensure_dirs()
    frame = generate_persona_grid(
        client_for_model(args.model, ollama_host=args.ollama_host),
        model=args.model,
        design=args.design,
        cache=ResponseCache(CACHE_DIR, cache_only=args.cache_only),
        limit=args.limit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
