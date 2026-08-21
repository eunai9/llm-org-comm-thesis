"""Judge module tests.

No live model calls: a scripted fake client returns controlled structured
payloads. What's tested is the design properties this module exists to
guarantee -- blind interleaving (no provenance leaks into a prompt), the
three phrasing variants actually differing, evidence-before-score in the
schema's own property order, and the same cache-first / honest-invalid
handling the simulator's run.py established.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from thesis.judge.prompt import (
    VARIANTS,
    JudgeItem,
    render_item_block,
    render_rubric_block,
)
from thesis.judge.rubric import (
    RUBRIC_ITEMS,
    SCORE_VALUES,
    InvalidJudgeResponseError,
    build_judge_schema,
    validate_judge_response,
)
from thesis.judge.run import (
    ScoringSummary,
    build_judge_request,
    results_to_rows,
    score_items,
)
from thesis.llm.base import CompletionRequest, CompletionResponse, Provider, Usage
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger


def _item(
    item_id: str = "i1", generated: bool = True, text: str = "Sure, I'll take care of it."
) -> JudgeItem:
    return JudgeItem(item_id=item_id, text=text, is_generated=generated, source_id="msg_42")


def _full_payload(score: int = 3) -> dict[str, Any]:
    return {
        item.key: {"evidence": f"quote for {item.key}", "score": score} for item in RUBRIC_ITEMS
    }


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
        text="{}", usage=Usage(input_tokens=100, output_tokens=50), model=model, parsed=payload
    )


# ------------------------------------------------------------------ rubric


def test_schema_declares_evidence_before_score() -> None:
    """Field order in the schema is the calibration mechanism, not decoration."""
    schema = build_judge_schema()
    for item in RUBRIC_ITEMS:
        keys = list(schema["properties"][item.key]["properties"])
        assert keys == ["evidence", "score"]


def test_schema_uses_an_enum_not_a_numeric_range() -> None:
    """minimum/maximum are not reliably enforced by structured-output APIs --
    this project already found that building the simulator's schema."""
    schema = build_judge_schema()
    for item in RUBRIC_ITEMS:
        spec = schema["properties"][item.key]["properties"]["score"]
        assert "minimum" not in spec
        assert "maximum" not in spec
        assert spec["enum"] == list(SCORE_VALUES)


def test_schema_forbids_extra_properties_at_every_level() -> None:
    schema = build_judge_schema()
    assert schema["additionalProperties"] is False
    for item in RUBRIC_ITEMS:
        assert schema["properties"][item.key]["additionalProperties"] is False


def test_six_items_across_two_named_groups() -> None:
    assert len(RUBRIC_ITEMS) == 6
    assert {item.group for item in RUBRIC_ITEMS} == {
        "empirical_fidelity",
        "communication_performance",
    }


def test_no_rubric_wording_says_generated_or_model() -> None:
    """The design constraint that makes blind interleaving possible: an item
    that says 'the generated message' could never be answered identically
    for a real one."""
    for item in RUBRIC_ITEMS:
        for text in (item.prompt, item.low_anchor, item.high_anchor):
            assert "generated" not in text.lower()
            assert "the model" not in text.lower()


def test_validate_accepts_a_well_formed_response() -> None:
    payload = _full_payload()
    assert validate_judge_response(payload) == payload


def test_validate_rejects_missing_items() -> None:
    payload = _full_payload()
    del payload["clarity"]
    with pytest.raises(InvalidJudgeResponseError, match="clarity"):
        validate_judge_response(payload)


def test_validate_rejects_out_of_range_score() -> None:
    payload = _full_payload()
    payload["clarity"]["score"] = 7
    with pytest.raises(InvalidJudgeResponseError):
        validate_judge_response(payload)


def test_validate_rejects_missing_evidence() -> None:
    payload = _full_payload()
    del payload["clarity"]["evidence"]
    with pytest.raises(InvalidJudgeResponseError):
        validate_judge_response(payload)


# ------------------------------------------------------------------ prompt


def test_all_three_variants_produce_different_text() -> None:
    """Consistency across variants is the thing being measured -- if they
    were identical there would be nothing to measure."""
    rendered = {v: render_rubric_block(v) for v in VARIANTS}
    assert len({rendered[v] for v in VARIANTS}) == 3


def test_every_variant_asks_all_six_items() -> None:
    for variant in VARIANTS:
        block = render_rubric_block(variant)
        for item in RUBRIC_ITEMS:
            assert item.key in block


def test_item_block_never_reveals_provenance() -> None:
    """The core blinding guarantee: is_generated must not leak into the text
    a judge actually reads."""
    generated = _item(generated=True, text="Thanks, I'll handle it.")
    real = _item(generated=False, text="Thanks, I'll handle it.")
    assert render_item_block(generated) == render_item_block(real)


def test_item_block_contains_no_id_or_source() -> None:
    item = _item(item_id="secret_id_123", text="Body text here.")
    block = render_item_block(item)
    assert "secret_id_123" not in block
    assert "msg_42" not in block


def test_reordered_variant_actually_reorders_items() -> None:
    from thesis.judge.prompt import _ITEM_ORDER

    assert _ITEM_ORDER["neutral"] != _ITEM_ORDER["reordered"]
    assert set(_ITEM_ORDER["neutral"]) == set(_ITEM_ORDER["reordered"])


# -------------------------------------------------------------------- request


def test_request_marks_rubric_as_cacheable() -> None:
    request = build_judge_request(_item(), "neutral", "claude-opus-5")
    assert request.cache_system is True
    assert request.system


def test_request_item_text_stays_out_of_the_cached_system_prompt() -> None:
    item = _item(text="A very specific unique phrase xyzzy123")
    request = build_judge_request(item, "neutral", "claude-opus-5")
    assert "xyzzy123" not in (request.system or "")
    assert "xyzzy123" in request.messages[0].content


def test_replicate_index_becomes_the_request_variant_field() -> None:
    """Distinguishes independent draws for a self-consistency check, the same
    role CompletionRequest.variant plays for the simulator's replicates."""
    r1 = build_judge_request(_item(), "neutral", "claude-opus-5", replicate=1)
    r2 = build_judge_request(_item(), "neutral", "claude-opus-5", replicate=2)
    assert r1.variant == 1
    assert r2.variant == 2


# --------------------------------------------------------------- scoring


def test_score_items_returns_one_result_per_valid_response(tmp_path: Path) -> None:
    client = _ScriptedClient([_response(_full_payload()), _response(_full_payload(score=5))])
    items = [_item("i1"), _item("i2")]
    results, summary = score_items(
        items,
        client,
        variant="neutral",
        model="claude-opus-5",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        run_id="r1",
    )
    assert len(results) == 2
    assert summary.n_scored == 2
    assert summary.n_invalid == 0


def test_score_items_records_invalid_without_dropping_the_rest(tmp_path: Path) -> None:
    """One malformed judge response must not discard every other score."""
    bad_payload = _full_payload()
    bad_payload["clarity"]["score"] = 99
    client = _ScriptedClient([_response(bad_payload), _response(_full_payload())])
    # Distinct text -- identical text would hash to the same cache key and the
    # second lookup would silently return the first (bad) cached response,
    # which is correct cache behavior but would defeat this specific test.
    items = [_item("i1", text="First message."), _item("i2", text="Second message.")]
    results, summary = score_items(
        items,
        client,
        variant="neutral",
        model="claude-opus-5",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        run_id="r1",
    )
    assert summary.n_invalid == 1
    assert summary.n_scored == 1
    assert len(results) == 1


def test_score_items_handles_missing_parsed_payload(tmp_path: Path) -> None:
    client = _ScriptedClient([_response(None)])
    results, summary = score_items(
        [_item("i1")],
        client,
        variant="neutral",
        model="claude-opus-5",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        run_id="r1",
    )
    assert results == []
    assert summary.n_invalid == 1


def test_second_pass_is_served_entirely_from_cache(tmp_path: Path) -> None:
    """Re-scoring must cost nothing, the same guarantee run.py gives the
    simulator's replicates."""
    cache = ResponseCache(tmp_path / "cache")
    items = [_item("i1")]

    first_client = _ScriptedClient([_response(_full_payload())])
    score_items(
        items,
        first_client,
        variant="neutral",
        model="claude-opus-5",
        cache=cache,
        ledger=CostLedger(tmp_path / "ledger.csv"),
        run_id="r1",
    )
    assert len(first_client.calls) == 1

    second_client = _ScriptedClient([])
    results, summary = score_items(
        items,
        second_client,
        variant="neutral",
        model="claude-opus-5",
        cache=cache,
        ledger=CostLedger(tmp_path / "ledger.csv"),
        run_id="r2",
    )
    assert second_client.calls == []
    assert summary.n_from_cache == 1
    assert len(results) == 1


def test_stub_model_scores_are_not_billed(tmp_path: Path) -> None:
    client = _ScriptedClient([_response(_full_payload(), model="stub-sim")])
    ledger_path = tmp_path / "ledger.csv"
    _, summary = score_items(
        [_item("i1")],
        client,
        variant="neutral",
        model="stub-sim",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(ledger_path),
        run_id="r1",
    )
    assert summary.total_cost_usd == 0.0
    assert CostLedger(ledger_path).total_usd() == 0.0


def test_results_preserve_provenance_for_analysis(tmp_path: Path) -> None:
    """The one place is_generated is allowed to surface -- after scoring, for
    the calling analysis code, never inside the prompt itself."""
    client = _ScriptedClient([_response(_full_payload())])
    results, _ = score_items(
        [_item("i1", generated=True)],
        client,
        variant="neutral",
        model="claude-opus-5",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        run_id="r1",
    )
    assert results[0].is_generated is True
    assert results[0].source_id == "msg_42"


def test_results_to_rows_flattens_every_score_and_evidence_field() -> None:
    from thesis.judge.run import JudgeResult

    result = JudgeResult(
        item_id="i1",
        source_id="msg_42",
        is_generated=True,
        variant="neutral",
        model="claude-opus-5",
        scores={"clarity": 4},
        evidence={"clarity": "clear ask"},
        from_cache=False,
    )
    rows = results_to_rows([result])
    assert rows[0]["score_clarity"] == 4
    assert rows[0]["evidence_clarity"] == "clear ask"


def test_scoring_summary_defaults_to_zero() -> None:
    summary = ScoringSummary()
    assert (summary.n_scored, summary.n_invalid, summary.total_cost_usd) == (0, 0, 0.0)
