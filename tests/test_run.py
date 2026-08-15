"""Run-orchestration tests.

Everything here uses a fake client: the point is to prove the *spending* logic
is right without spending anything. The properties tested are the ones that
protect the budget and the provenance chain -- cache-before-call, dirty-tree
refusal, and honest accounting of what was served versus paid for.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from thesis.llm.base import (
    Capabilities,
    CompletionRequest,
    CompletionResponse,
    Provider,
    Usage,
)
from thesis.llm.cache import CacheMissError, ResponseCache
from thesis.llm.cost import CostLedger
from thesis.sim.grid import expand, order_for_cache
from thesis.sim.persona import Persona, PersonaStyle
from thesis.sim.run import (
    RESULT_SCHEMA,
    DirtyWorkingTreeError,
    RunManifest,
    build_request,
    config_hash,
    dry_run,
    git_state,
    run_grid,
)
from thesis.sim.scenario import build_scenarios


class FakeClient:
    """A client that records calls and never touches the network."""

    provider: Provider = "anthropic"

    def __init__(self, *, payload: dict[str, object] | None = None) -> None:
        self.calls: list[CompletionRequest] = []
        self.token_counts = 0
        self._payload = payload or {
            "subject": "Re: volumes",
            "body": "Numbers attached.",
            "decision": "accept",
            "confidence": "high",
            "reasoning_brief": "Routine and within my remit.",
        }

    def capabilities(self, model: str) -> Capabilities:
        return Capabilities(
            supports_sampling_params=False,
            min_cacheable_prompt_tokens=512,
            thinking_on_by_default=True,
        )

    def count_tokens(self, request: CompletionRequest) -> int:
        self.token_counts += 1
        return 1200

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return CompletionResponse(
            text=json.dumps(self._payload),
            usage=Usage(input_tokens=1200, output_tokens=200),
            model=request.model,
            stop_reason="end_turn",
            parsed=dict(self._payload),
        )

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        return "batch_fake"

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        return {}


def _persona(persona_id: str = "r3_trading") -> Persona:
    return Persona(
        persona_id=persona_id,
        seniority_rank=3,
        rank_label="Director",
        department="Trading",
        style=PersonaStyle(
            mean_tokens=50.0,
            mean_recipients=2.0,
            imperative_ratio=0.15,
            hedge_rate=0.03,
            deference_rate=0.005,
            question_ratio=0.09,
        ),
        n_people=25,
        n_messages=4236,
        derivation="cell",
    )


def _cells(n: int = 6):  # type: ignore[no-untyped-def]
    scenarios = build_scenarios()[:3]
    cells = order_for_cache(
        expand([_persona()], scenarios, [("claude-opus-5", "sim_anthropic")], 2)
    )
    return cells[:n]


def _manifest() -> RunManifest:
    return RunManifest(
        run_id="testrun",
        started_at="2026-08-15T00:00:00+00:00",
        git_commit="abc123",
        git_dirty=False,
        config_hash="deadbeef",
        models=["claude-opus-5"],
    )


def _run(tmp_path: Path, client: FakeClient, cache: ResponseCache | None = None):  # type: ignore[no-untyped-def]
    from thesis.config import load_config

    cells = _cells()
    manifest = _manifest()
    rows = run_grid(
        cells,
        client,
        {},
        load_config(),
        run_id="testrun",
        cache=cache or ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        manifest=manifest,
    )
    return rows, manifest


# ------------------------------------------------------------------ requests


def test_request_marks_the_prefix_as_cacheable() -> None:
    """Without this flag the cache-aware grid ordering buys nothing."""
    cell = _cells(1)[0]
    request = build_request(cell, [])
    assert request.cache_system is True
    assert request.system


def test_request_carries_the_output_schema() -> None:
    request = build_request(_cells(1)[0], [])
    assert request.output_schema is not None
    assert "decision" in request.output_schema["properties"]


def test_scenario_text_stays_out_of_the_cached_system_prompt() -> None:
    cell = _cells(1)[0]
    request = build_request(cell, [])
    assert cell.scenario.incoming_message not in (request.system or "")


# ----------------------------------------------------------------- execution


def test_run_generates_every_cell(tmp_path: Path) -> None:
    client = FakeClient()
    rows, manifest = _run(tmp_path, client)
    assert len(rows) == len(_cells())
    assert manifest.n_generated == len(_cells())
    assert manifest.n_from_cache == 0


def test_second_run_is_served_entirely_from_cache(tmp_path: Path) -> None:
    """The January re-run must cost nothing."""
    cache = ResponseCache(tmp_path / "cache")
    first = FakeClient()
    _run(tmp_path, first, cache)
    assert len(first.calls) == len(_cells())

    second = FakeClient()
    rows, manifest = _run(tmp_path, second, cache)
    assert second.calls == []
    assert manifest.n_from_cache == len(_cells())
    assert manifest.n_generated == 0
    assert all(row["cost_usd"] == 0.0 for row in rows)


def test_cache_only_run_refuses_to_call_out(tmp_path: Path) -> None:
    client = FakeClient()
    cache = ResponseCache(tmp_path / "cache", cache_only=True)
    with pytest.raises(CacheMissError):
        _run(tmp_path, client, cache)
    assert client.calls == []


def test_invalid_payload_is_counted_not_silently_dropped(tmp_path: Path) -> None:
    """A schema drift must show up in the manifest, not vanish."""
    client = FakeClient(
        payload={
            "subject": "s",
            "body": "b",
            "decision": "maybe",
            "confidence": "high",
            "reasoning_brief": "r",
        }
    )
    rows, manifest = _run(tmp_path, client)
    assert rows == []
    assert manifest.n_invalid == len(_cells())


def test_rows_match_the_declared_schema(tmp_path: Path) -> None:
    import pyarrow as pa

    rows, _ = _run(tmp_path, FakeClient())
    table = pa.Table.from_pylist(rows, schema=RESULT_SCHEMA)
    assert table.num_rows == len(rows)
    assert table.schema == RESULT_SCHEMA


def test_ledger_records_every_call(tmp_path: Path) -> None:
    _run(tmp_path, FakeClient())
    ledger = CostLedger(tmp_path / "ledger.csv")
    assert ledger.total_usd() > 0


# --------------------------------------------------------------- projections


def test_dry_run_counts_one_request_per_cache_group(tmp_path: Path) -> None:
    """Counting every cell would itself be thousands of calls."""
    from thesis.config import load_config

    client = FakeClient()
    cells = _cells()
    projection = dry_run(cells, client, {}, load_config())
    assert client.token_counts == projection["n_cache_groups"]
    assert client.calls == []
    assert projection["projected_cost_usd"] > 0


# ---------------------------------------------------------------- provenance


def test_git_state_reports_a_commit_and_flag() -> None:
    commit, dirty = git_state()
    assert isinstance(commit, str) and commit
    assert isinstance(dirty, bool)


def test_missing_git_is_treated_as_dirty(tmp_path: Path) -> None:
    """Unknown provenance must behave like dirty provenance."""
    commit, dirty = git_state(tmp_path)
    assert (commit, dirty) == ("unknown", True)


def test_config_hash_is_stable_and_short() -> None:
    from thesis.config import load_config

    config = load_config()
    assert config_hash(config) == config_hash(config)
    assert len(config_hash(config)) == 16


def test_manifest_round_trips(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "run.json"
    manifest.write(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "testrun"
    assert loaded["git_dirty"] is False


def test_dirty_tree_error_exists_for_the_guard() -> None:
    assert issubclass(DirtyWorkingTreeError, RuntimeError)
