"""Materialise the matched real-vs-generated reply corpus as one tidy table.

Every validation check in this package -- the embedding map, the manual review
packet, the fidelity statistics -- needs the *same* set of paired replies,
identified the same way. Until now that pairing was rebuilt inside each
analysis, which is how two checks quietly end up describing slightly different
samples. This module builds it once and writes it down.

A row is one usable thread from ``S_shots``: the real incoming message, the
real human reply to it, and the reply the simulated persona wrote when given
that same incoming message. Threads whose real replier has no matching persona
are skipped upstream by :mod:`thesis.sim.real_stimuli` and counted here, not
silently dropped.

Generation goes through :func:`thesis.sim.run.run_grid` rather than a private
loop, so these rows are produced by exactly the code path the experimental grid
uses -- same prompt assembly, same cache, same validation. Run it without
``--local`` to rebuild the table for free from responses already archived.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from thesis.config import load_config
from thesis.data.rfc822 import clean_body
from thesis.llm.base import LLMClient
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import CACHE_DIR, COST_LEDGER, INTERIM_DIR, MESSAGES_PARQUET_GLOB, ensure_dirs
from thesis.sim.memory import MemoryItem
from thesis.sim.memory_generation import load_frozen_memory
from thesis.sim.persona import Persona, load_frozen_personas
from thesis.sim.real_stimuli import RealStimulusPair, build_real_stimulus_pairs
from thesis.sim.run import RunManifest, run_grid

log = get_logger(__name__)

PAIRS_PATH: Path = INTERIM_DIR / "real_vs_generated_pairs.parquet"


@dataclass(frozen=True, slots=True)
class PairTable:
    """The paired corpus plus the provenance needed to interpret it."""

    frame: pd.DataFrame
    run_id: str
    model: str
    n_pairs_built: int
    n_rows: int


def _role_by_address() -> dict[str, tuple[int, str]]:
    from thesis.data.identity import resolve_owners
    from thesis.data.roles import build_role_index, load_employees, load_title_rank_table

    role_index, _ = build_role_index(
        load_employees(), resolve_owners(MESSAGES_PARQUET_GLOB), load_title_rank_table()
    )
    return {a: (r.seniority_rank, r.department) for a, r in role_index.items()}


def build_pair_table(
    client: LLMClient,
    *,
    model: str,
    role_label: str,
    personas: Sequence[Persona] | None = None,
    stores: Mapping[str, Sequence[MemoryItem]] | None = None,
    cache_only: bool = True,
    limit: int | None = None,
) -> PairTable:
    """Generate (or serve from cache) one reply per real stimulus and return the table."""
    config = load_config()
    personas = personas if personas is not None else load_frozen_personas()
    stores = stores if stores is not None else load_frozen_memory()

    pairs: list[RealStimulusPair] = build_real_stimulus_pairs(
        personas, _role_by_address(), model=model, role_label=role_label
    )
    if limit is not None:
        pairs = pairs[:limit]
    by_cell = {p.cell.cell_id: p for p in pairs}

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = RunManifest(
        run_id=run_id,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit="",
        git_dirty=False,
        config_hash="",
        models=[model],
        design={"kind": "real_stimulus_pairs"},
        n_cells=len(pairs),
    )
    rows: list[dict[str, Any]] = run_grid(
        [p.cell for p in pairs],
        client,
        stores,
        config,
        run_id=run_id,
        cache=ResponseCache(CACHE_DIR, cache_only=cache_only),
        ledger=CostLedger(COST_LEDGER),
        manifest=manifest,
    )

    records: list[dict[str, Any]] = []
    for row in rows:
        pair = by_cell[row["cell_id"]]
        records.append(
            {
                "cell_id": row["cell_id"],
                "thread_id": pair.thread_id,
                "persona_id": row["persona_id"],
                "seniority_rank": row["seniority_rank"],
                "department": row["department"],
                "direction": row["direction"],
                "model": row["response_model"],
                "stimulus_text": pair.stimulus_text,
                "real_reply": pair.real_reply_text,
                "real_reply_subject": pair.real_reply_subject,
                "real_reply_body": pair.real_reply_body,
                # The stored corpus body re-cleaned by the *current* cleaner.
                # data/interim was built before the Lotus Notes quoting fix
                # (see thesis.data.rfc822), so 60% of these bodies still carry
                # a quoted ancestor; comparing that against a generated body
                # measures the corpus's quoting conventions, not authorship.
                # Recomputing here repairs the comparison without waiting on a
                # full re-ingest, and keeps the unrepaired column beside it so
                # the difference stays visible rather than being assumed away.
                "real_reply_body_recleaned": clean_body(pair.real_reply_body),
                "generated_subject": row["subject"],
                "generated_reply": row["body"],
                "decision": row["decision"],
                "confidence": row["confidence"],
                "reasoning_brief": row["reasoning_brief"],
                "from_cache": row["from_cache"],
            }
        )

    frame = pd.DataFrame.from_records(records)
    return PairTable(
        frame=frame,
        run_id=run_id,
        model=model,
        n_pairs_built=len(pairs),
        n_rows=len(frame),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        metavar="MODEL",
        default=None,
        help="Generate missing replies with a local Ollama model instead of failing.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(PAIRS_PATH))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    if args.local:
        from thesis.llm.ollama_client import OllamaClient, OllamaUnavailableError

        ollama = OllamaClient(args.local)
        if not ollama.is_available():
            msg = "no Ollama server reachable; start it with 'ollama serve'."
            raise OllamaUnavailableError(msg)
        client: LLMClient = ollama
        model, role_label, cache_only = args.local, "sim_local", False
    else:
        from thesis.llm.anthropic_client import AnthropicClient

        simulator = load_config().models.simulator[0]
        client = AnthropicClient()
        model, role_label, cache_only = simulator.model_id, simulator.role_label, True

    table = build_pair_table(
        client, model=model, role_label=role_label, cache_only=cache_only, limit=args.limit
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.frame.to_parquet(out, compression="zstd", index=False)
    log.info(
        "wrote %d of %d pairs to %s (model %s)",
        table.n_rows,
        table.n_pairs_built,
        out,
        table.model,
    )


if __name__ == "__main__":
    main()
