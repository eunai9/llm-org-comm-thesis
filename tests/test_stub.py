"""Stub-client tests.

The stub exists so the pipeline can be exercised without spending. These tests
guard the two ways that could go wrong: stub output being billed as if it were
real, and stub output being mistaken for real data downstream.
"""

from __future__ import annotations

from pathlib import Path

from thesis.llm.base import CompletionRequest, Message
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.llm.stub_client import StubClient, is_stub_model
from thesis.sim.run import RESULT_SCHEMA


def _request(variant: int = 1, content: str = "reply please") -> CompletionRequest:
    return CompletionRequest(
        model="claude-opus-5",
        messages=[Message(role="user", content=content)],
        max_tokens=512,
        system="x" * 3000,
        variant=variant,
    )


def test_stub_model_ids_are_identifiable() -> None:
    assert is_stub_model("stub-sim")
    assert not is_stub_model("claude-opus-5")


def test_stub_label_always_carries_the_prefix() -> None:
    """Even if constructed with a plain name, the marker must survive."""
    assert StubClient(model_label="sim").model_label.startswith("stub-")


def test_response_is_stamped_with_the_stub_model() -> None:
    response = StubClient().complete(_request())
    assert is_stub_model(response.model)


def test_stub_output_satisfies_the_response_schema() -> None:
    from thesis.sim.schemas import validate_response

    parsed = StubClient().complete(_request()).parsed
    assert parsed is not None
    validate_response(parsed)


def test_stub_is_deterministic_for_the_same_request() -> None:
    """A cache hit and a fresh stub call must agree, or cache tests would pass
    or fail depending on which path ran first."""
    a = StubClient().complete(_request()).text
    b = StubClient().complete(_request()).text
    assert a == b


def test_stub_varies_across_replicates() -> None:
    """A stub returning one constant string would hide the very bug class it
    exists to surface."""
    texts = {StubClient().complete(_request(variant=v)).text for v in range(1, 12)}
    assert len(texts) > 1


def test_stub_varies_across_prompts() -> None:
    texts = {StubClient().complete(_request(content=f"scenario {i}")).text for i in range(12)}
    assert len(texts) > 1


def test_results_schema_records_the_responding_model() -> None:
    """Stub output must be identifiable from the results file alone."""
    assert "response_model" in RESULT_SCHEMA.names
    assert "model" in RESULT_SCHEMA.names


def test_stub_run_costs_nothing(tmp_path: Path) -> None:
    from tests.conftest import make_cells, make_manifest

    from thesis.config import load_config
    from thesis.sim.run import run_grid

    manifest = make_manifest()
    rows = run_grid(
        make_cells(4),
        StubClient(),
        {},
        load_config(),
        run_id="stubrun",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        manifest=manifest,
    )
    assert rows
    assert all(row["cost_usd"] == 0.0 for row in rows)
    assert CostLedger(tmp_path / "ledger.csv").total_usd() == 0.0
    assert manifest.offline is True
