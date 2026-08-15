"""Walk the grid, generate the responses, and record where they came from.

This is the module that spends money, so it is built to make the expensive
mistakes hard rather than merely unlikely:

- **Nothing runs before its cost is projected.** ``--dry-run`` expands the full
  grid, counts tokens with the provider's own tokenizer, and prints the
  projection without sending anything. It is the default way to inspect a run.
- **The cache is consulted first, always.** A cell already generated is never
  paid for twice, and ``--cache-only`` refuses to call out at all -- which is
  how the January analysis re-runs at zero cost and how a reviewer without an
  API key can reproduce the downstream work.
- **A dirty working tree blocks a real run.** Every result carries the commit
  that produced it. Without that guard, "which code produced Figure 4?"
  becomes unanswerable in February, when it matters most and nobody remembers.
- **Cells run in cache order, not nesting order.** Cells sharing a prompt
  prefix must be consecutive or each re-writes an entry that has already
  expired, paying the write premium every time and reading almost never.

Memory generation is deliberately *not* here. It is a separate pass over
(persona x direction) groups, costing its own ~100 calls, and keeping it
separate means a re-run of the response grid does not regenerate the memories
the grid is supposed to hold constant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from thesis.config import Config, load_config
from thesis.llm.base import CompletionRequest, LLMClient, Message, Usage
from thesis.llm.cache import ResponseCache, cache_key
from thesis.llm.cost import CostLedger, LedgerEntry, cost_usd, guard_budget
from thesis.llm.stub_client import is_stub_model
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import CACHE_DIR, COST_LEDGER, MANIFESTS_DIR, RUNS_DIR, ensure_dirs
from thesis.sim.grid import GridCell, expand, order_for_cache, summarize
from thesis.sim.memory import MemoryItem
from thesis.sim.persona import Persona
from thesis.sim.prompt import assemble, retrieve_for_group
from thesis.sim.scenario import Scenario
from thesis.sim.schemas import RESPONSE_SCHEMA, InvalidResponseError, validate_response

log = get_logger(__name__)

MAX_OUTPUT_TOKENS = 2048

RESULT_SCHEMA = pa.schema(
    [
        pa.field("cell_id", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("persona_id", pa.string()),
        pa.field("seniority_rank", pa.int32()),
        pa.field("department", pa.string()),
        pa.field("scenario_id", pa.string()),
        pa.field("task_type", pa.string()),
        pa.field("direction", pa.string()),
        pa.field("stakes", pa.string()),
        pa.field("replicate", pa.int32()),
        # The model the design asked for. This is the experimental factor,
        # and is what analysis should group by.
        pa.field("model", pa.string()),
        # The model that actually answered. Normally identical to `model`, but
        # an offline run records a "stub-" id here -- so stub output is
        # identifiable from the results file alone, without needing to know
        # how the run was invoked.
        pa.field("response_model", pa.string()),
        pa.field("role_label", pa.string()),
        pa.field("subject", pa.string()),
        pa.field("body", pa.string()),
        pa.field("decision", pa.string()),
        pa.field("confidence", pa.string()),
        pa.field("reasoning_brief", pa.string()),
        pa.field("from_cache", pa.bool_()),
        pa.field("input_tokens", pa.int32()),
        pa.field("cache_read_input_tokens", pa.int32()),
        pa.field("output_tokens", pa.int32()),
        pa.field("cost_usd", pa.float64()),
    ]
)


class DirtyWorkingTreeError(RuntimeError):
    """Raised when a non-pilot run is started from uncommitted code."""


def git_state(repo: Path | None = None) -> tuple[str, bool]:
    """Return ``(commit_sha, is_dirty)`` for the working tree.

    A missing or broken git checkout returns ``("unknown", True)`` rather than
    raising: unknown provenance must behave like dirty provenance, since both
    mean a result cannot be traced to code.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True
    return sha, bool(status)


def config_hash(config: Config) -> str:
    """Stable hash of the resolved configuration.

    Hashes the *resolved* config rather than the YAML files, so an override
    supplied at the command line is reflected -- otherwise two runs with
    materially different settings could claim the same provenance.
    """
    blob = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class RunManifest:
    """Everything needed to say where a set of results came from."""

    run_id: str
    started_at: str
    git_commit: str
    git_dirty: bool
    config_hash: str
    models: list[str]
    design: dict[str, Any] = field(default_factory=dict)
    n_cells: int = 0
    n_from_cache: int = 0
    n_generated: int = 0
    n_invalid: int = 0
    # True when any response came from the stub client. Recorded so a manifest
    # cannot be mistaken for the provenance of a real run.
    offline: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    finished_at: str | None = None

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


def build_request(
    cell: GridCell,
    memories: Sequence[MemoryItem],
) -> CompletionRequest:
    """Assemble the API request for one cell.

    ``cache_system=True`` marks the stable prefix as cacheable; without it the
    ordering work in :mod:`thesis.sim.grid` buys nothing.
    """
    prompt = assemble(cell.persona, cell.scenario, memories)
    return CompletionRequest(
        model=cell.model,
        messages=[Message(role="user", content=prompt.variable_suffix)],
        max_tokens=MAX_OUTPUT_TOKENS,
        system=prompt.stable_prefix,
        output_schema=RESPONSE_SCHEMA,
        cache_system=True,
        # The replicate index is the draw index: without it every replicate of
        # a cell would share one cache entry and the design's variance term
        # would be structurally zero.
        variant=cell.replicate,
        metadata={"cell_id": cell.cell_id, "cache_group": cell.cache_group},
    )


def _memories_for(
    cell: GridCell,
    stores: Mapping[str, Sequence[MemoryItem]],
) -> list[MemoryItem]:
    """Retrieve this cell's memories, once per cache group."""
    store = stores.get(cell.persona.persona_id, [])
    return retrieve_for_group(cell.persona, cell.scenario.direction, store)


def _is_billable(response_model: str, *, from_cache: bool) -> bool:
    """Whether this response actually cost money.

    A cache hit never reached the provider, and a stub response never left the
    machine. Pricing either one would inflate the cost ledger -- the file the
    thesis's total-spend figure is summed from -- with money that was never
    spent.
    """
    return not from_cache and not is_stub_model(response_model)


def _result_row(
    cell: GridCell,
    run_id: str,
    payload: dict[str, Any],
    usage: Usage,
    *,
    from_cache: bool,
    response_model: str,
) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "run_id": run_id,
        "persona_id": cell.persona.persona_id,
        "seniority_rank": cell.persona.seniority_rank,
        "department": cell.persona.department,
        "scenario_id": cell.scenario.scenario_id,
        "task_type": cell.scenario.task_type,
        "direction": cell.scenario.direction,
        "stakes": cell.scenario.stakes,
        "replicate": cell.replicate,
        "model": cell.model,
        "response_model": response_model,
        "role_label": cell.role_label,
        "subject": payload["subject"],
        "body": payload["body"],
        "decision": payload["decision"],
        "confidence": payload["confidence"],
        "reasoning_brief": payload["reasoning_brief"],
        "from_cache": from_cache,
        "input_tokens": usage.input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "output_tokens": usage.output_tokens,
        # Priced against the model that actually answered, not the one the
        # design asked for: in an offline run those differ, and pricing the
        # configured model would bill a stub run at real rates.
        "cost_usd": (
            cost_usd(cell.model, usage)
            if _is_billable(response_model, from_cache=from_cache)
            else 0.0
        ),
    }


def dry_run(
    cells: Sequence[GridCell],
    client: LLMClient,
    stores: Mapping[str, Sequence[MemoryItem]],
    config: Config,
) -> dict[str, Any]:
    """Project the cost of a run without sending anything.

    Counts one representative request per cache group with the provider's own
    tokenizer, then scales. Counting every cell would itself be thousands of
    API calls to avoid making thousands of API calls.
    """
    by_group: dict[str, list[GridCell]] = {}
    for cell in cells:
        by_group.setdefault(cell.cache_group, []).append(cell)

    total = 0.0
    per_model: dict[str, int] = {}
    for group, group_cells in by_group.items():
        representative = group_cells[0]
        request = build_request(representative, _memories_for(representative, stores))
        input_tokens = client.count_tokens(request)
        usage = Usage(
            input_tokens=input_tokens * len(group_cells),
            output_tokens=MAX_OUTPUT_TOKENS // 4 * len(group_cells),
        )
        total += cost_usd(representative.model, usage)
        per_model[representative.model] = per_model.get(representative.model, 0) + len(group_cells)
        log.debug("group %s: %d cells at ~%d input tokens", group, len(group_cells), input_tokens)

    return {
        "n_cells": len(cells),
        "n_cache_groups": len(by_group),
        "cells_per_model": per_model,
        "projected_cost_usd": round(total, 2),
        "max_cost_usd": config.run.max_cost_usd,
    }


def run_grid(
    cells: Sequence[GridCell],
    client: LLMClient,
    stores: Mapping[str, Sequence[MemoryItem]],
    config: Config,
    *,
    run_id: str,
    cache: ResponseCache,
    ledger: CostLedger,
    manifest: RunManifest,
) -> list[dict[str, Any]]:
    """Generate every cell, serving from cache where possible."""
    rows: list[dict[str, Any]] = []
    totals = Usage()

    for index, cell in enumerate(cells, start=1):
        request = build_request(cell, _memories_for(cell, stores))
        key = cache_key(request, client.provider)

        response = cache.get(key)
        if response is None:
            response = client.complete(request)
            cache.put(key, request, response, client.provider)
            manifest.n_generated += 1
        else:
            manifest.n_from_cache += 1

        payload = response.parsed
        if payload is None:
            log.warning("cell %s returned no parseable JSON; skipping", cell.cell_id)
            manifest.n_invalid += 1
            continue
        try:
            payload = validate_response(payload)
        except InvalidResponseError as exc:
            log.warning("cell %s failed validation: %s", cell.cell_id, exc)
            manifest.n_invalid += 1
            continue

        rows.append(
            _result_row(
                cell,
                run_id,
                payload,
                response.usage,
                from_cache=response.from_cache,
                response_model=response.model,
            )
        )
        totals = totals + response.usage
        billable = _is_billable(response.model, from_cache=response.from_cache)
        ledger.record(
            LedgerEntry(
                run_id=run_id,
                provider=client.provider,
                model=cell.model,
                # Labelled distinctly so a stub run is identifiable in the
                # ledger rather than looking like a real call that cost $0.
                call_kind="simulate" if not is_stub_model(response.model) else "simulate-stub",
                usage=response.usage,
                from_cache=not billable,
            )
        )

        if is_stub_model(response.model):
            manifest.offline = True

        if index % 100 == 0:
            log.info(
                "%d/%d cells (%d cached, %d generated)",
                index,
                len(cells),
                manifest.n_from_cache,
                manifest.n_generated,
            )

    manifest.usage = asdict(totals)
    manifest.total_cost_usd = round(ledger.total_usd(), 4)
    return rows


def build_cells(
    personas: Sequence[Persona],
    scenarios: Sequence[Scenario],
    config: Config,
    n_replicates: int,
) -> list[GridCell]:
    """Expand and order the grid for this configuration."""
    models = [(m.model_id, m.role_label) for m in config.models.simulator]
    return order_for_cache(expand(personas, scenarios, models, n_replicates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Cap cells, for smoke tests.")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot run: permitted from a dirty tree, and not budget-guarded.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Project cost and exit without sending anything.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Serve only from cache; fail rather than call the API.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Use the stub client: no API key, no cost, no network. Exercises "
            "the whole pipeline, but the output is templated text and is NOT "
            "usable as a result."
        ),
    )
    parser.add_argument("--out", default=str(RUNS_DIR / "simulation.parquet"))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()
    config = load_config()

    commit, dirty = git_state()
    if (
        dirty
        and not args.pilot
        and not args.offline
        and config.run.require_clean_git
        and not args.dry_run
    ):
        msg = (
            "refusing to start a non-pilot run from a dirty working tree: "
            "results could not be traced back to a commit. Commit first, or "
            "pass --pilot."
        )
        raise DirtyWorkingTreeError(msg)

    from thesis.data.identity import resolve_owners
    from thesis.data.roles import build_role_index, load_employees, load_title_rank_table
    from thesis.llm.anthropic_client import AnthropicClient
    from thesis.paths import MESSAGES_PARQUET_GLOB
    from thesis.sim.persona import derive_personas
    from thesis.sim.scenario import build_scenarios

    title_ranks = load_title_rank_table()
    role_index, _ = build_role_index(
        load_employees(), resolve_owners(MESSAGES_PARQUET_GLOB), title_ranks
    )
    personas = derive_personas(
        {a: (r.seniority_rank, r.department) for a, r in role_index.items()},
        {rank: label for _, (rank, label) in title_ranks.items()},
    )
    cells = build_cells(personas, build_scenarios(), config, args.replicates)
    if args.limit is not None:
        cells = cells[: args.limit]

    log.info("design: %s", json.dumps(summarize(cells)))

    client: LLMClient
    if args.offline:
        from thesis.llm.stub_client import StubClient

        client = StubClient()
        log.warning(
            "OFFLINE MODE: responses are templated, not generated. Output is "
            "for pipeline testing only and must not be reported as a result."
        )
    else:
        client = AnthropicClient()
    stores: dict[str, Sequence[MemoryItem]] = {}

    if args.dry_run:
        projection = dry_run(cells, client, stores, config)
        log.info("dry run: %s", json.dumps(projection, indent=2))
        return

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = RunManifest(
        run_id=run_id,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=commit,
        git_dirty=dirty,
        config_hash=config_hash(config),
        models=[m.model_id for m in config.models.simulator],
        design=summarize(cells),
        n_cells=len(cells),
    )

    if not args.pilot and not args.offline:
        projection = dry_run(cells, client, stores, config)
        guard_budget(float(projection["projected_cost_usd"]), config.run.max_cost_usd)

    cache = ResponseCache(CACHE_DIR, cache_only=args.cache_only)
    ledger = CostLedger(COST_LEDGER)
    rows = run_grid(
        cells,
        client,
        stores,
        config,
        run_id=run_id,
        cache=cache,
        ledger=ledger,
        manifest=manifest,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=RESULT_SCHEMA), out_path, compression="zstd")

    manifest.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest.write(MANIFESTS_DIR / f"run_{run_id}.json")

    log.info(
        "wrote %d rows to %s (%d cached, %d generated, %d invalid, $%.4f)",
        len(rows),
        out_path,
        manifest.n_from_cache,
        manifest.n_generated,
        manifest.n_invalid,
        manifest.total_cost_usd,
    )


if __name__ == "__main__":
    main()
