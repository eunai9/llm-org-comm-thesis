"""Batch-submission tests.

A batch is paid for at submission and collected hours later, so the expensive
failures are the ones where money is spent and the results cannot be used:
orphaned batches, unmappable responses, and resubmitting work already cached.
Those are what these tests cover.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from tests.conftest import make_cells

from thesis.llm.base import CompletionRequest, CompletionResponse, Provider, Usage
from thesis.llm.batch import (
    BatchState,
    _assign_ids,
    _request_for,
    chunk,
    collect,
    is_complete,
    pending_cells,
    short_id,
    submit,
)
from thesis.llm.cache import ResponseCache, cache_key

_PAYLOAD = {
    "subject": "s",
    "body": "b",
    "decision": "accept",
    "confidence": "high",
    "reasoning_brief": "r",
}


class RecordingBatchClient:
    """Records submissions; returns results only when told they are ready."""

    provider: Provider = "anthropic"

    def __init__(self, *, ready: bool = True) -> None:
        self.submitted: list[list[tuple[str, CompletionRequest]]] = []
        self.ready = ready
        self._counter = 0

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        self.submitted.append(list(requests))
        self._counter += 1
        return f"batch_{self._counter}"

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        if not self.ready:
            return None
        index = int(batch_id.split("_")[1]) - 1
        return {
            custom_id: CompletionResponse(
                text=json.dumps(_PAYLOAD),
                usage=Usage(input_tokens=100, output_tokens=20),
                model="claude-opus-5",
                parsed=dict(_PAYLOAD),
            )
            for custom_id, _ in self.submitted[index]
        }


def _cached(cache: ResponseCache, cell: Any) -> None:
    request = _request_for(cell, {})
    cache.put(
        cache_key(request, "anthropic"),
        request,
        CompletionResponse(text="{}", usage=Usage(), model="claude-opus-5"),
        "anthropic",
    )


def test_short_id_is_deterministic_and_short() -> None:
    """Descriptive cell ids reach 70 chars, beyond provider custom_id limits."""
    cell_id = "sim_anthropic__r5_trading__schedule_coordination__lateral__routine__r1"
    assert len(cell_id) > 64
    assert short_id(cell_id) == short_id(cell_id)
    assert len(short_id(cell_id)) <= 32


def test_assign_ids_is_bijective_for_real_cells() -> None:
    cells = make_cells(6)
    mapping = _assign_ids(cells)
    assert len(mapping) == len({c.cell_id for c in cells})


def test_chunking_splits_and_preserves_everything() -> None:
    items = list(range(1050))
    chunks = chunk(items, 500)
    assert [len(c) for c in chunks] == [500, 500, 50]
    assert [x for c in chunks for x in c] == items


def test_chunk_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        chunk([1, 2], 0)


def test_pending_excludes_already_cached_cells(tmp_path: Path) -> None:
    """Resubmitting cached work would pay twice for the same response."""
    cells = make_cells(4)
    cache = ResponseCache(tmp_path / "cache")
    assert len(pending_cells(cells, {}, cache, "anthropic")) == 4

    _cached(cache, cells[0])
    remaining = pending_cells(cells, {}, cache, "anthropic")
    assert len(remaining) == 3
    assert cells[0].cell_id not in {c.cell_id for c in remaining}


def test_submit_records_state_and_mapping(tmp_path: Path, monkeypatch: Any) -> None:
    import thesis.llm.batch as batch_module

    monkeypatch.setattr(batch_module, "BATCH_STATE_DIR", tmp_path / "batches")
    client = RecordingBatchClient()
    cells = make_cells(6)

    state = submit(cells, client, {}, ResponseCache(tmp_path / "cache"), run_id="r1")

    assert state.n_submitted == 6
    assert len(state.batch_ids) == 1
    assert set(state.custom_id_to_cell.values()) == {c.cell_id for c in cells}
    assert state.path.is_file(), "state must be persisted or batches are orphaned"


def test_submit_skips_when_everything_is_cached(tmp_path: Path, monkeypatch: Any) -> None:
    import thesis.llm.batch as batch_module

    monkeypatch.setattr(batch_module, "BATCH_STATE_DIR", tmp_path / "batches")
    cells = make_cells(3)
    cache = ResponseCache(tmp_path / "cache")
    for cell in cells:
        _cached(cache, cell)

    client = RecordingBatchClient()
    state = submit(cells, client, {}, cache, run_id="r2")

    assert client.submitted == []
    assert state.n_submitted == 0
    assert state.n_skipped_cached == 3


def test_collect_writes_responses_into_the_cache(tmp_path: Path, monkeypatch: Any) -> None:
    """A collected batch must make the ordinary runner free afterwards."""
    import thesis.llm.batch as batch_module

    monkeypatch.setattr(batch_module, "BATCH_STATE_DIR", tmp_path / "batches")
    cells = make_cells(5)
    cache = ResponseCache(tmp_path / "cache")
    client = RecordingBatchClient()

    state = submit(cells, client, {}, cache, run_id="r3")
    counts = collect(state, client, cells, {}, cache)

    assert counts["written"] == 5
    assert is_complete(state)
    for cell in cells:
        assert cache_key(_request_for(cell, {}), "anthropic") in cache


def test_collect_leaves_unfinished_batches_alone(tmp_path: Path, monkeypatch: Any) -> None:
    import thesis.llm.batch as batch_module

    monkeypatch.setattr(batch_module, "BATCH_STATE_DIR", tmp_path / "batches")
    cells = make_cells(3)
    cache = ResponseCache(tmp_path / "cache")
    client = RecordingBatchClient(ready=False)

    state = submit(cells, client, {}, cache, run_id="r4")
    counts = collect(state, client, cells, {}, cache)

    assert counts["written"] == 0
    assert counts["pending_batches"] == 1
    assert not is_complete(state)


def test_collect_is_idempotent(tmp_path: Path, monkeypatch: Any) -> None:
    """Re-collecting must not double-write or crash."""
    import thesis.llm.batch as batch_module

    monkeypatch.setattr(batch_module, "BATCH_STATE_DIR", tmp_path / "batches")
    cells = make_cells(4)
    cache = ResponseCache(tmp_path / "cache")
    client = RecordingBatchClient()

    state = submit(cells, client, {}, cache, run_id="r5")
    collect(state, client, cells, {}, cache)
    second = collect(state, client, cells, {}, cache)
    assert second["written"] == 0


def test_state_round_trips(tmp_path: Path, monkeypatch: Any) -> None:
    """Collection happens in a later process; state must survive to disk."""
    import thesis.llm.batch as batch_module

    monkeypatch.setattr(batch_module, "BATCH_STATE_DIR", tmp_path / "batches")
    cells = make_cells(3)
    submit(cells, RecordingBatchClient(), {}, ResponseCache(tmp_path / "c"), run_id="r6")

    reloaded = BatchState.load("r6")
    assert reloaded.n_submitted == 3
    assert len(reloaded.custom_id_to_cell) == 3


def test_missing_state_raises_clearly(tmp_path: Path, monkeypatch: Any) -> None:
    import thesis.llm.batch as batch_module

    monkeypatch.setattr(batch_module, "BATCH_STATE_DIR", tmp_path / "batches")
    with pytest.raises(FileNotFoundError, match="no batch state"):
        BatchState.load("does-not-exist")


def test_batched_request_matches_the_interactive_one() -> None:
    """If the two diverged, a batched response could never satisfy a later
    cache lookup and the entire cost saving would silently evaporate."""
    from thesis.sim.run import build_request

    cell = make_cells(1)[0]
    assert cache_key(_request_for(cell, {}), "anthropic") == cache_key(
        build_request(cell, []), "anthropic"
    )
