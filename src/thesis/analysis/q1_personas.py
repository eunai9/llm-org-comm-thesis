"""The second Q1 experiment: 40 persona variants writing to named receivers.

Generation goes through :func:`thesis.sim.run.run_grid`, the same path as
every other grid. The analysis mode reads a saved grid and calls no model.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from thesis.analysis.judge_swap import client_for_model
from thesis.analysis.q1 import Q1Design, scenarios_for_design
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
    frame = pd.DataFrame.from_records(rows)
    traits = frame["persona_id"].map(variant_traits)
    frame["base_persona_id"] = traits.str[0]
    frame["tenure"] = traits.str[1]
    frame["origin"] = traits.str[2]
    frame["receiver_rank"] = frame["seniority_rank"] + frame["direction"].map(RECEIVER_STEP)
    frame["prompt_text_hash"] = prompt_hash
    log.info("%s: %d of %d cells returned a valid reply", model, len(frame), len(cells))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="e.g. qwen/qwen3.8-27b or llama3.2:3b")
    parser.add_argument("--design", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--ollama-host", default=None)
    args = parser.parse_args()
    configure_logging()
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
