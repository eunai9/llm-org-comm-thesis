"""Submit the grid as provider batches, at half the price.

Batching is what makes this study affordable: the planned ~13,300 calls cost
half as much submitted as batches, which is the difference between a
comfortable budget and an uncomfortable one. The trade is latency -- a batch
returns within hours rather than seconds -- which is why this is a separate
pass rather than a mode of the interactive runner.

The design is deliberately **cache-filling rather than result-producing**:

    submit  ->  (wait)  ->  collect  ->  responses written into the cache
                                     ->  then the ordinary runner serves
                                         every cell from cache, at zero cost

Keeping it to that one job avoids duplicating the runner's validation,
ledger, and Parquet logic in a second place where the two could drift apart.
It also makes the whole thing resumable for free: if collection is
interrupted, the responses already written stay in the cache and only the
remainder is fetched.

**Cells already in the cache are never submitted.** A resumed or partially
completed run therefore costs only what is genuinely still missing, which
matters when a batch of a thousand cells fails partway.

**Custom ids are short hashes, not cell ids.** The descriptive cell ids this
project uses run to 70 characters, and provider limits on ``custom_id`` are
tighter than that -- so submitting them directly would be rejected at the
point where a large batch is being accepted. Each request instead gets a short
deterministic id, and the mapping back to cell ids is persisted alongside the
batch, because results come back keyed by ``custom_id`` and are useless
without it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from thesis.llm.base import BatchClient, CompletionRequest
from thesis.llm.cache import ResponseCache, cache_key
from thesis.logging_setup import get_logger
from thesis.paths import RUNS_DIR
from thesis.sim.grid import GridCell
from thesis.sim.memory import MemoryItem
from thesis.sim.prompt import retrieve_for_group
from thesis.sim.run import build_request

log = get_logger(__name__)

BATCH_STATE_DIR = RUNS_DIR / "batches"

# Chunk size. Well under any provider ceiling on requests or payload bytes,
# and small enough that one rejected chunk does not invalidate an entire
# submission.
MAX_REQUESTS_PER_BATCH = 500


class BatchNotReadyError(RuntimeError):
    """Raised when results are requested for a batch still processing."""


def short_id(cell_id: str) -> str:
    """A short, deterministic ``custom_id`` for one cell.

    Deterministic so that resubmitting the same cell produces the same id,
    which keeps a resumed submission consistent with an earlier one. Truncated
    to 16 hex characters: collision risk across even the full ~13,300-call
    design is negligible, and :func:`_assign_ids` asserts uniqueness anyway
    rather than trusting the estimate.
    """
    return "c_" + hashlib.sha256(cell_id.encode("utf-8")).hexdigest()[:16]


def _assign_ids(cells: Sequence[GridCell]) -> dict[str, str]:
    """Map ``custom_id -> cell_id``, refusing to proceed on a collision.

    A collision would silently attach one cell's response to another cell,
    which is the kind of error that survives into a results table unnoticed.
    """
    mapping: dict[str, str] = {}
    for cell in cells:
        key = short_id(cell.cell_id)
        if key in mapping and mapping[key] != cell.cell_id:
            msg = f"custom_id collision: {mapping[key]!r} and {cell.cell_id!r} -> {key}"
            raise ValueError(msg)
        mapping[key] = cell.cell_id
    return mapping


@dataclass
class BatchState:
    """Everything needed to collect a submission later, from a fresh process."""

    run_id: str
    submitted_at: str
    provider: str
    batch_ids: list[str] = field(default_factory=list)
    custom_id_to_cell: dict[str, str] = field(default_factory=dict)
    n_submitted: int = 0
    n_skipped_cached: int = 0
    collected_batch_ids: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return BATCH_STATE_DIR / f"{self.run_id}.json"

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, run_id: str) -> BatchState:
        path = BATCH_STATE_DIR / f"{run_id}.json"
        if not path.is_file():
            msg = f"no batch state for run {run_id!r} at {path}"
            raise FileNotFoundError(msg)
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def pending_cells(
    cells: Sequence[GridCell],
    stores: Mapping[str, Sequence[MemoryItem]],
    cache: ResponseCache,
    provider: str,
) -> list[GridCell]:
    """Cells with no cached response yet -- the only ones worth paying for."""
    pending = []
    for cell in cells:
        request = _request_for(cell, stores)
        if cache_key(request, provider) not in cache:
            pending.append(cell)
    return pending


def _request_for(cell: GridCell, stores: Mapping[str, Sequence[MemoryItem]]) -> CompletionRequest:
    """Build a cell's request exactly as the interactive runner would.

    Routed through the runner's own ``build_request`` so a batched response and
    an interactive one are produced from byte-identical prompts. If they
    diverged, a batched response could never satisfy a later cache lookup and
    the whole cost saving would silently evaporate.
    """
    memories = retrieve_for_group(
        cell.persona, cell.scenario.direction, stores.get(cell.persona.persona_id, [])
    )
    return build_request(cell, memories)


def chunk(items: Sequence[Any], size: int = MAX_REQUESTS_PER_BATCH) -> list[list[Any]]:
    """Split into submission-sized chunks."""
    if size <= 0:
        msg = "chunk size must be positive"
        raise ValueError(msg)
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def submit(
    cells: Sequence[GridCell],
    client: BatchClient,
    stores: Mapping[str, Sequence[MemoryItem]],
    cache: ResponseCache,
    *,
    run_id: str,
) -> BatchState:
    """Submit every not-yet-cached cell, and persist enough state to collect.

    State is written **before** returning, and after each chunk, so a process
    that dies mid-submission still leaves a record of the batches already
    accepted. Without that, those batches would be paid for and then orphaned:
    running but unreachable.
    """
    outstanding = pending_cells(cells, stores, cache, client.provider)
    state = BatchState(
        run_id=run_id,
        submitted_at=datetime.now(UTC).isoformat(timespec="seconds"),
        provider=client.provider,
        custom_id_to_cell=_assign_ids(outstanding),
        n_skipped_cached=len(cells) - len(outstanding),
    )

    if not outstanding:
        log.info("nothing to submit: all %d cell(s) already cached", len(cells))
        state.write()
        return state

    by_cell_id = {cell.cell_id: cell for cell in outstanding}
    for group in chunk(outstanding):
        payload = [(short_id(cell.cell_id), _request_for(cell, stores)) for cell in group]
        batch_id = client.submit_batch(payload)
        state.batch_ids.append(batch_id)
        state.n_submitted += len(payload)
        state.write()
        log.info("submitted batch %s with %d request(s)", batch_id, len(payload))

    log.info(
        "submitted %d cell(s) across %d batch(es); %d already cached",
        state.n_submitted,
        len(state.batch_ids),
        state.n_skipped_cached,
    )
    assert set(by_cell_id) <= set(state.custom_id_to_cell.values())
    return state


def collect(
    state: BatchState,
    client: BatchClient,
    cells: Sequence[GridCell],
    stores: Mapping[str, Sequence[MemoryItem]],
    cache: ResponseCache,
) -> dict[str, int]:
    """Fetch finished batches and write their responses into the cache.

    Batches still running are left alone and reported, so this can be called
    repeatedly until everything has landed. Each response is written under the
    same key the interactive runner will look up, which is what turns a
    completed batch into a zero-cost run afterwards.
    """
    by_cell_id = {cell.cell_id: cell for cell in cells}
    counts = {"written": 0, "pending_batches": 0, "missing": 0}

    for batch_id in state.batch_ids:
        if batch_id in state.collected_batch_ids:
            continue

        results = client.fetch_batch(batch_id)
        if results is None:
            counts["pending_batches"] += 1
            log.info("batch %s still processing", batch_id)
            continue

        for custom_id, response in results.items():
            cell_id = state.custom_id_to_cell.get(custom_id)
            cell = by_cell_id.get(cell_id or "")
            if cell is None:
                # A response we cannot map back to a cell is unusable, and
                # guessing would risk attaching it to the wrong one.
                counts["missing"] += 1
                log.warning("batch %s returned unmappable custom_id %s", batch_id, custom_id)
                continue

            request = _request_for(cell, stores)
            cache.put(cache_key(request, client.provider), request, response, client.provider)
            counts["written"] += 1

        state.collected_batch_ids.append(batch_id)
        state.write()
        log.info("collected batch %s: %d response(s)", batch_id, len(results))

    return counts


def is_complete(state: BatchState) -> bool:
    """Whether every submitted batch has been collected."""
    return set(state.batch_ids) == set(state.collected_batch_ids)
