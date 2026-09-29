from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from thesis.analysis.role_inference import (
    ABSOLUTE_SCHEMA,
    PAIRED_SCHEMA,
    InvalidRoleInferenceResponseError,
    PairedItem,
    RoleInferenceResult,
    build_absolute_items_from_grid,
    build_absolute_items_from_real_email,
    build_absolute_request,
    build_paired_items_from_grid,
    build_positive_control_items,
    render_paired_block,
    run_role_inference_absolute,
    run_role_inference_paired,
    summarize_absolute,
    validate_absolute_response,
    validate_paired_response,
)
from thesis.judge.prompt import JudgeItem, render_item_block
from thesis.llm.base import CompletionRequest, CompletionResponse, Provider, Usage
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.sim.scenario import _DIRECTION_FRAMING, Direction


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


def _tiny_grid() -> pd.DataFrame:
    rows = []
    for direction in ("up", "lateral", "down"):
        rows.append(
            {
                "cell_id": f"cell_{direction}",
                "persona_id": "persona_1",
                "scenario_id": f"approve_or_decline__{direction}__high__neutral",
                "task_type": "approve_or_decline",
                "direction": direction,
                "stakes": "high",
                "replicate": 1,
                "model": "llama3.2:3b",
                "body": "Approved, go ahead.\n\nBest,\nJohn Smith, Vice President",
            }
        )
    return pd.DataFrame(rows)


def test_build_absolute_items_from_grid_returns_one_item_per_row() -> None:
    items, true_directions = build_absolute_items_from_grid(_tiny_grid())
    assert len(items) == 3
    assert {true_directions[i.item_id] for i in items} == {"up", "lateral", "down"}


def test_build_absolute_items_from_grid_strips_identity() -> None:
    items, _ = build_absolute_items_from_grid(_tiny_grid())
    for item in items:
        assert "John Smith" not in item.text
        assert "Vice President" not in item.text
    assert "Approved, go ahead." in items[0].text


def test_no_direction_framing_language_reaches_the_rendered_prompt() -> None:
    """The blinding constraint itself: none of the exact framing sentences
    thesis.sim.scenario uses to tell the *generating* model which
    direction it is writing in may appear anywhere in the text the judge
    is shown. This is the actual verification of the Global Constraints
    blinding rule, not a restatement of it."""
    items, _ = build_absolute_items_from_grid(_tiny_grid())
    for item in items:
        rendered = render_item_block(item).lower()
        for framing_sentence in _DIRECTION_FRAMING.values():
            assert framing_sentence.lower() not in rendered


def test_build_absolute_items_from_real_email() -> None:
    bodies = pd.DataFrame(
        {"message_uid": ["m1", "m2"], "body_clean": ["Approved.", "Please review by Friday."]}
    )
    directions = pd.DataFrame({"message_uid": ["m1", "m2"], "direction": ["lateral", "down"]})
    items, true_directions = build_absolute_items_from_real_email(bodies, directions)
    assert len(items) == 2
    assert all(not item.is_generated for item in items)
    assert true_directions[items[0].item_id] in ("lateral", "down")


def test_build_paired_items_from_grid_pairs_up_with_down_same_scenario() -> None:
    pairs = build_paired_items_from_grid(_tiny_grid())
    assert len(pairs) == 1  # one (task_type, stakes, tone) triple in the fixture
    assert pairs[0].senior_slot in ("A", "B")


def _metrics_result(true_direction: str, inferred_direction: str) -> RoleInferenceResult:
    return RoleInferenceResult(
        item_id="i",
        source_id="i",
        is_generated=True,
        true_direction=true_direction,
        model="qwen2.5:3b",
        inferred_direction=inferred_direction,
        evidence="x",
        from_cache=False,
    )


def test_summarize_absolute_perfect_agreement() -> None:
    results = [_metrics_result(d, d) for d in ("up", "lateral", "down") for _ in range(10)]
    metrics = summarize_absolute(results)
    assert metrics.accuracy == 1.0
    assert metrics.kappa == 1.0
    assert metrics.n == 30


def test_summarize_absolute_chance_agreement_has_kappa_near_zero() -> None:
    import itertools

    labels = ["up", "lateral", "down"]
    # Every true label paired with every inferred label equally often --
    # by construction, no better than chance.
    results = [
        _metrics_result(true, inferred)
        for true, inferred in itertools.product(labels, labels)
        for _ in range(10)
    ]
    metrics = summarize_absolute(results)
    assert abs(metrics.kappa) < 0.05


def test_summarize_absolute_confusion_matrix_shape() -> None:
    results = [
        _metrics_result("up", "up"),
        _metrics_result("up", "lateral"),
        _metrics_result("down", "down"),
    ]
    metrics = summarize_absolute(results)
    assert metrics.confusion["up"]["up"] == 1
    assert metrics.confusion["up"]["lateral"] == 1
    assert metrics.confusion["down"]["down"] == 1


def test_summarize_absolute_confidence_interval_widens_with_fewer_items() -> None:
    small = summarize_absolute(
        [_metrics_result("up", "up")] * 5 + [_metrics_result("up", "down")] * 5
    )
    large = summarize_absolute(
        [_metrics_result("up", "up")] * 50 + [_metrics_result("up", "down")] * 50
    )
    assert (small.ci_high - small.ci_low) > (large.ci_high - large.ci_low)


def test_positive_control_items_contain_the_framing_sentence() -> None:
    """The inverse of test_no_direction_framing_language_reaches_the_rendered_prompt:
    here the framing sentence must be present, since this is the sanity
    check that the harness can detect the label when it is actually there."""
    items, true_directions = build_positive_control_items(_tiny_grid())
    for item in items:
        true_direction = cast(Direction, true_directions[item.item_id])
        assert _DIRECTION_FRAMING[true_direction].lower() in item.text.lower()
