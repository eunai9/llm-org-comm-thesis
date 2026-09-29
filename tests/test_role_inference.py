from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from thesis.analysis.role_inference import (
    ABSOLUTE_SCHEMA,
    PAIRED_SCHEMA,
    InvalidRoleInferenceResponseError,
    PairedItem,
    build_absolute_request,
    render_paired_block,
    run_role_inference_absolute,
    run_role_inference_paired,
    validate_absolute_response,
    validate_paired_response,
)
from thesis.judge.prompt import JudgeItem
from thesis.llm.base import CompletionRequest, CompletionResponse, Provider, Usage
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger


@dataclass
class _ScriptedClient:
    responses: list[CompletionResponse]
    provider: Provider = "ollama"

    def __post_init__(self) -> None:
        self.calls: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return self.responses.pop(0)


def _item(item_id: str = "i1", text: str = "Sure, I'll take care of it.") -> JudgeItem:
    return JudgeItem(item_id=item_id, text=text, is_generated=True, source_id="msg_1")


def _response(
    payload: dict[str, object] | None, model: str = "local/qwen2.5:3b"
) -> CompletionResponse:
    return CompletionResponse(
        text="{}", usage=Usage(input_tokens=80, output_tokens=20), model=model, parsed=payload
    )


def _cache(tmp_path: Path) -> ResponseCache:
    return ResponseCache(tmp_path / "cache")


def _ledger(tmp_path: Path) -> CostLedger:
    return CostLedger(tmp_path / "ledger.csv")


def test_schema_restricts_the_direction_field_to_the_three_values() -> None:
    assert set(ABSOLUTE_SCHEMA["properties"]["inferred_direction"]["enum"]) == {
        "up",
        "lateral",
        "down",
    }


def test_validate_accepts_a_well_formed_response() -> None:
    payload = {"evidence": "no directive language", "inferred_direction": "lateral"}
    assert validate_absolute_response(payload) == payload


def test_validate_rejects_a_direction_outside_the_enum() -> None:
    payload = {"evidence": "x", "inferred_direction": "sideways"}
    try:
        validate_absolute_response(payload)
        raise AssertionError("expected InvalidRoleInferenceResponseError")
    except InvalidRoleInferenceResponseError:
        pass


def test_request_never_contains_the_word_direction_labels() -> None:
    """The rendered prompt must be the reply and the task framing only --
    never a direction word, which would defeat blinding by suggesting the
    answer set outside the schema's own enum. The enum values are allowed
    to appear in the schema (a structural field, never shown as prose in
    the message text) but not in the system framing's free text."""
    request = build_absolute_request(_item(), "qwen2.5:3b")
    assert request.system is not None
    for leak_word in ("you are writing to", "reports into", "senior to you"):
        assert leak_word not in request.system.lower()


def test_run_scores_one_item_end_to_end(tmp_path: Path) -> None:
    client = _ScriptedClient(
        [_response({"evidence": "no directive language", "inferred_direction": "lateral"})]
    )
    results, summary = run_role_inference_absolute(
        [_item()],
        client,
        model="qwen2.5:3b",
        cache=_cache(tmp_path),
        ledger=_ledger(tmp_path),
        run_id="test-run",
    )
    assert summary.n_scored == 1
    assert summary.n_invalid == 0
    assert results[0].inferred_direction == "lateral"
    assert results[0].item_id == "i1"


def test_invalid_response_is_counted_not_raised(tmp_path: Path) -> None:
    client = _ScriptedClient([_response({"evidence": "x", "inferred_direction": "sideways"})])
    results, summary = run_role_inference_absolute(
        [_item()],
        client,
        model="qwen2.5:3b",
        cache=_cache(tmp_path),
        ledger=_ledger(tmp_path),
        run_id="test-run",
    )
    assert results == []
    assert summary.n_invalid == 1
    assert summary.n_scored == 0


def test_result_carries_the_true_label_the_judge_never_saw(tmp_path: Path) -> None:
    """is_generated and the caller's own true direction are tracked on the
    result object, never rendered into the prompt -- the same split
    JudgeItem/DiscriminationResult already use."""
    client = _ScriptedClient(
        [_response({"evidence": "no directive language", "inferred_direction": "down"})]
    )
    results, _ = run_role_inference_absolute(
        [_item()],
        client,
        model="qwen2.5:3b",
        cache=_cache(tmp_path),
        ledger=_ledger(tmp_path),
        run_id="test-run",
    )
    assert results[0].is_generated is True


def _paired_item(pair_id: str = "p1") -> PairedItem:
    return PairedItem(
        pair_id=pair_id,
        text_a="I need this by Friday.",
        text_b="Whenever you get a chance, could you take a look?",
        senior_slot="A",
    )


def test_paired_schema_restricts_the_answer_to_a_or_b() -> None:
    assert set(PAIRED_SCHEMA["properties"]["answer"]["enum"]) == {"A", "B"}


def test_paired_block_shows_both_replies_labeled_and_hides_the_answer() -> None:
    block = render_paired_block(_paired_item())
    assert "I need this by Friday." in block
    assert "Whenever you get a chance" in block
    assert "A" in block and "B" in block
    assert "senior_slot" not in block.lower()


def test_paired_validate_rejects_an_answer_outside_a_or_b() -> None:
    try:
        validate_paired_response({"evidence": "x", "answer": "C"})
        raise AssertionError("expected InvalidRoleInferenceResponseError")
    except InvalidRoleInferenceResponseError:
        pass


def test_paired_run_scores_one_pair_and_checks_it_against_the_true_slot(
    tmp_path: Path,
) -> None:
    client = _ScriptedClient([_response({"evidence": "more direct", "answer": "A"})])
    results, summary = run_role_inference_paired(
        [_paired_item()],
        client,
        model="qwen2.5:3b",
        cache=_cache(tmp_path),
        ledger=_ledger(tmp_path),
        run_id="test-run",
    )
    assert summary.n_scored == 1
    assert results[0].judged_correctly is True  # answered A, true senior slot is A
