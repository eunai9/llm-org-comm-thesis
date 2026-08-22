"""Discrimination-check tests.

No live model calls: a scripted fake client returns controlled structured
payloads. What's tested is what this module exists to guarantee -- the
request never reveals an item's true label, an out-of-range rating is
rejected, and the cache/cost discipline matches the rest of the judge
package exactly (including the local-billing bug already caught once in
judge/run.py -- this module is checked for the identical mistake).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from thesis.judge.discrimination import (
    DISCRIMINATION_SCHEMA,
    ORIGIN_VALUES,
    DiscriminationSummary,
    InvalidDiscriminationResponseError,
    build_discrimination_request,
    results_to_rows,
    run_discrimination,
    validate_discrimination_response,
)
from thesis.judge.prompt import JudgeItem
from thesis.llm.base import CompletionRequest, CompletionResponse, Provider, Usage
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger


def _item(
    item_id: str = "i1", generated: bool = True, text: str = "Sure, I'll take care of it."
) -> JudgeItem:
    return JudgeItem(item_id=item_id, text=text, is_generated=generated, source_id="msg_42")


def _payload(likely_origin: int = 3) -> dict[str, Any]:
    return {"evidence": "some quoted text", "likely_origin": likely_origin}


class _ScriptedClient:
    provider: Provider = "anthropic"

    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return self._responses.pop(0)


def _response(payload: dict[str, Any] | None, model: str = "claude-opus-5") -> CompletionResponse:
    return CompletionResponse(
        text="{}", usage=Usage(input_tokens=80, output_tokens=30), model=model, parsed=payload
    )


# --------------------------------------------------------------------- schema


def test_schema_uses_an_ordinal_enum_not_a_binary_label() -> None:
    """The whole point: a rankable score, not a forced real/generated pick,
    is what makes an AUC computable downstream."""
    spec = DISCRIMINATION_SCHEMA["properties"]["likely_origin"]
    assert spec["enum"] == list(ORIGIN_VALUES)
    assert "minimum" not in spec
    assert "maximum" not in spec


def test_schema_declares_evidence_before_likely_origin() -> None:
    keys = list(DISCRIMINATION_SCHEMA["properties"])
    assert keys == ["evidence", "likely_origin"]


def test_validate_accepts_a_well_formed_response() -> None:
    payload = _payload()
    assert validate_discrimination_response(payload) == payload


def test_validate_rejects_out_of_range_rating() -> None:
    with pytest.raises(InvalidDiscriminationResponseError):
        validate_discrimination_response(_payload(likely_origin=9))


def test_validate_rejects_missing_evidence() -> None:
    with pytest.raises(InvalidDiscriminationResponseError):
        validate_discrimination_response({"likely_origin": 3})


# -------------------------------------------------------------------- request


def test_request_never_reveals_the_items_true_label() -> None:
    """The core guarantee: is_generated must not leak into what the judge
    actually reads, for either a real or a generated item."""
    generated = _item(generated=True, text="Thanks, I'll handle it.")
    real = _item(generated=False, text="Thanks, I'll handle it.")
    req_a = build_discrimination_request(generated, "claude-opus-5")
    req_b = build_discrimination_request(real, "claude-opus-5")
    assert req_a.messages[0].content == req_b.messages[0].content


def test_request_item_text_stays_out_of_the_cached_system_prompt() -> None:
    item = _item(text="A very specific unique phrase xyzzy123")
    request = build_discrimination_request(item, "claude-opus-5")
    assert "xyzzy123" not in (request.system or "")
    assert "xyzzy123" in request.messages[0].content


def test_task_framing_never_says_generated_or_ai() -> None:
    """Same blinding-in-wording discipline as the rubric: the framing must
    work identically regardless of which item it's sent with."""
    request = build_discrimination_request(_item(), "claude-opus-5")
    text = (request.system or "").lower()
    assert "ai-generated" not in text
    assert "the model" not in text


def test_replicate_index_becomes_the_request_variant_field() -> None:
    r1 = build_discrimination_request(_item(), "claude-opus-5", replicate=1)
    r2 = build_discrimination_request(_item(), "claude-opus-5", replicate=2)
    assert r1.variant == 1
    assert r2.variant == 2


def test_request_marks_task_framing_as_cacheable() -> None:
    request = build_discrimination_request(_item(), "claude-opus-5")
    assert request.cache_system is True


# --------------------------------------------------------------- scoring


def test_run_discrimination_returns_one_result_per_valid_response(tmp_path: Path) -> None:
    client = _ScriptedClient([_response(_payload(1)), _response(_payload(5))])
    items = [_item("i1", text="first"), _item("i2", text="second")]
    results, summary = run_discrimination(
        items,
        client,
        model="claude-opus-5",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        run_id="r1",
    )
    assert len(results) == 2
    assert summary.n_scored == 2
    assert summary.n_invalid == 0
    assert {r.likely_origin for r in results} == {1, 5}


def test_run_discrimination_records_invalid_without_dropping_the_rest(tmp_path: Path) -> None:
    bad = {"evidence": "x", "likely_origin": 99}
    client = _ScriptedClient([_response(bad), _response(_payload())])
    items = [_item("i1", text="first message"), _item("i2", text="second message")]
    results, summary = run_discrimination(
        items,
        client,
        model="claude-opus-5",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        run_id="r1",
    )
    assert summary.n_invalid == 1
    assert summary.n_scored == 1
    assert len(results) == 1


def test_run_discrimination_handles_missing_parsed_payload(tmp_path: Path) -> None:
    client = _ScriptedClient([_response(None)])
    results, summary = run_discrimination(
        [_item("i1")],
        client,
        model="claude-opus-5",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        run_id="r1",
    )
    assert results == []
    assert summary.n_invalid == 1


def test_second_pass_is_served_entirely_from_cache(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    items = [_item("i1")]

    first_client = _ScriptedClient([_response(_payload())])
    run_discrimination(
        items,
        first_client,
        model="claude-opus-5",
        cache=cache,
        ledger=CostLedger(tmp_path / "l1.csv"),
        run_id="r1",
    )
    assert len(first_client.calls) == 1

    second_client = _ScriptedClient([])
    results, summary = run_discrimination(
        items,
        second_client,
        model="claude-opus-5",
        cache=cache,
        ledger=CostLedger(tmp_path / "l2.csv"),
        run_id="r2",
    )
    assert second_client.calls == []
    assert summary.n_from_cache == 1
    assert len(results) == 1


def test_stub_model_results_are_not_billed(tmp_path: Path) -> None:
    client = _ScriptedClient([_response(_payload(), model="stub-sim")])
    ledger_path = tmp_path / "ledger.csv"
    _, summary = run_discrimination(
        [_item("i1")],
        client,
        model="stub-sim",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(ledger_path),
        run_id="r1",
    )
    assert summary.total_cost_usd == 0.0
    assert CostLedger(ledger_path).total_usd() == 0.0


def test_local_model_results_are_not_billed(tmp_path: Path) -> None:
    """The identical mistake caught once in judge/run.py's billing check --
    missing is_local_model() would crash on an unpriced 'local/...' id."""
    client = _ScriptedClient([_response(_payload(), model="local/llama3.2:3b")])
    ledger_path = tmp_path / "ledger.csv"
    _, summary = run_discrimination(
        [_item("i1")],
        client,
        model="local/llama3.2:3b",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(ledger_path),
        run_id="r1",
    )
    assert summary.total_cost_usd == 0.0
    assert CostLedger(ledger_path).total_usd() == 0.0


def test_results_preserve_the_true_label_for_analysis(tmp_path: Path) -> None:
    """The one place is_generated is allowed to surface -- after scoring,
    never inside the prompt itself (checked separately above)."""
    client = _ScriptedClient([_response(_payload())])
    results, _ = run_discrimination(
        [_item("i1", generated=True)],
        client,
        model="claude-opus-5",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        run_id="r1",
    )
    assert results[0].is_generated is True
    assert results[0].source_id == "msg_42"


def test_results_to_rows_includes_likely_origin_and_true_label() -> None:
    from thesis.judge.discrimination import DiscriminationResult

    result = DiscriminationResult(
        item_id="i1",
        source_id="msg_42",
        is_generated=True,
        model="claude-opus-5",
        likely_origin=2,
        evidence="a bit too polished",
        from_cache=False,
    )
    rows = results_to_rows([result])
    assert rows[0]["likely_origin"] == 2
    assert rows[0]["is_generated"] is True


def test_summary_defaults_to_zero() -> None:
    summary = DiscriminationSummary()
    assert (summary.n_scored, summary.n_invalid, summary.total_cost_usd) == (0, 0, 0.0)
