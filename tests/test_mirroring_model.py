"""Tests for the model-based mirroring check.

No live model calls: a scripted fake client returns controlled payloads, the
same way test_judge.py does it. What is tested is the deterministic part --
how the prompt renders, what the schema enforces, how draws are averaged, and
how the AUC is computed. Whether the check actually works is an empirical
question the validation run answers, not something a test can assert.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from thesis.analysis.mirroring_model import (
    SCORE_VALUES,
    InvalidMirroringResponseError,
    MirroringItem,
    MirroringScore,
    ScoringSummary,
    auc_against,
    build_items,
    build_request,
    build_schema,
    compare_runs,
    render_item_block,
    score_items,
    self_consistency,
    truncate_words,
    validate_response,
)
from thesis.llm.base import CompletionRequest, CompletionResponse, Provider, Usage
from thesis.llm.cache import ResponseCache, cache_key
from thesis.llm.cost import CostLedger

STIMULUS = "Please send the signed contract to Richard before Friday."
REPLY = "Can you send the signed contract to Richard?"


def _item(item_id: str = "1") -> MirroringItem:
    return MirroringItem(item_id=item_id, stimulus=STIMULUS, reply=REPLY)


def _response(
    payload: dict[str, Any] | None, model: str = "local/qwen2.5:3b"
) -> CompletionResponse:
    return CompletionResponse(
        text="{}", usage=Usage(input_tokens=200, output_tokens=20), model=model, parsed=payload
    )


class _ScriptedClient:
    provider: Provider = "ollama"

    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return self._responses.pop(0)


def _scored(
    client: _ScriptedClient,
    tmp_path: Path,
    *,
    draws: int = 3,
    items: list[MirroringItem] | None = None,
) -> tuple[list[MirroringScore], ScoringSummary]:
    return score_items(
        items if items is not None else [_item()],
        client,
        model="qwen2.5:3b",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        run_id="test",
        draws=draws,
    )


# ----------------------------------------------------------------- prompt


def test_item_block_shows_both_messages() -> None:
    block = render_item_block(_item())
    assert STIMULUS in block
    assert REPLY in block


def test_item_block_ends_with_the_scale() -> None:
    """A 3B model reads the end of its prompt best, so the scale goes last."""
    block = render_item_block(_item())
    assert block.rstrip().endswith("Quote a short piece of the reply, then give the score.")


def test_item_block_states_every_anchor() -> None:
    block = render_item_block(_item())
    for level in SCORE_VALUES:
        assert f"{level} = " in block


def test_item_block_leaks_no_provenance() -> None:
    """The model is never told which reply was generated."""
    block = render_item_block(MirroringItem(item_id="gen_42", stimulus=STIMULUS, reply=REPLY))
    assert "gen_42" not in block
    assert "generated" not in block.lower()


def test_a_long_incoming_message_is_cut() -> None:
    long_stimulus = " ".join(["word"] * 900)
    block = render_item_block(MirroringItem(item_id="1", stimulus=long_stimulus, reply=REPLY))
    assert block.count("word") == 400


def test_truncate_leaves_a_short_message_alone() -> None:
    assert truncate_words("one two three") == "one two three"


# ----------------------------------------------------------------- schema


def test_schema_declares_evidence_before_score() -> None:
    """Field order is the calibration mechanism, the same as the judge rubric."""
    assert list(build_schema()["properties"]) == ["evidence", "score"]


def test_schema_uses_an_enum_not_a_numeric_range() -> None:
    spec = build_schema()["properties"]["score"]
    assert spec["enum"] == list(SCORE_VALUES)
    assert "minimum" not in spec
    assert "maximum" not in spec


def test_schema_forbids_extra_properties() -> None:
    assert build_schema()["additionalProperties"] is False


def test_valid_response_passes_through_unchanged() -> None:
    payload = {"evidence": "Can you send it?", "score": 5}
    assert validate_response(payload) == payload


@pytest.mark.parametrize(
    "payload",
    [
        {"score": 3},
        {"evidence": "x"},
        {"evidence": "x", "score": 0},
        {"evidence": "x", "score": 6},
        {"evidence": "x", "score": "high"},
        {"evidence": 3, "score": 3},
    ],
)
def test_a_malformed_response_is_rejected(payload: dict[str, Any]) -> None:
    with pytest.raises(InvalidMirroringResponseError):
        validate_response(payload)


# ------------------------------------------------------------------ draws


def test_each_draw_gets_its_own_cache_key() -> None:
    """Without this the cache serves draw 1 three times and the spread is zero."""
    keys = {
        cache_key(build_request(_item(), "qwen2.5:3b", draw=draw), "ollama") for draw in range(3)
    }
    assert len(keys) == 3


def test_the_draw_index_is_the_requests_variant_field() -> None:
    assert build_request(_item(), "qwen2.5:3b", draw=2).variant == 2


def test_scores_are_averaged_across_draws(tmp_path: Path) -> None:
    client = _ScriptedClient([_response({"evidence": "q", "score": score}) for score in (5, 3, 4)])
    results, summary = _scored(client, tmp_path)
    assert results[0].score == 4.0
    assert results[0].draws == (5, 3, 4)
    assert results[0].spread == 2.0
    assert summary.n_scored == 1


def test_an_invalid_draw_is_skipped_not_raised(tmp_path: Path) -> None:
    client = _ScriptedClient(
        [
            _response({"evidence": "q", "score": 5}),
            _response({"evidence": "q", "score": 9}),
            _response({"evidence": "q", "score": 3}),
        ]
    )
    results, summary = _scored(client, tmp_path)
    assert results[0].draws == (5, 3)
    assert summary.n_invalid == 1
    assert summary.n_scored == 1


def test_an_item_whose_draws_all_fail_is_dropped(tmp_path: Path) -> None:
    client = _ScriptedClient([_response(None) for _ in range(3)])
    results, summary = _scored(client, tmp_path)
    assert results == []
    assert summary.n_invalid == 3
    assert summary.n_scored == 0


def test_one_bad_item_does_not_discard_the_others(tmp_path: Path) -> None:
    """Two different replies, so the cache cannot serve one for the other."""
    client = _ScriptedClient([_response(None), _response({"evidence": "q", "score": 2})])
    items = [
        MirroringItem(item_id="1", stimulus=STIMULUS, reply=REPLY),
        MirroringItem(item_id="2", stimulus=STIMULUS, reply="Sent it this morning."),
    ]
    results, _ = _scored(client, tmp_path, draws=1, items=items)
    assert [result.item_id for result in results] == ["2"]


def test_a_second_run_is_served_from_cache(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    ledger = CostLedger(tmp_path / "ledger.csv")
    for expected_calls in (1, 0):
        client = _ScriptedClient([_response({"evidence": "q", "score": 4})])
        score_items(
            [_item()],
            client,
            model="qwen2.5:3b",
            cache=cache,
            ledger=ledger,
            run_id="test",
            draws=1,
        )
        assert len(client.calls) == expected_calls


def test_a_local_model_costs_nothing(tmp_path: Path) -> None:
    client = _ScriptedClient([_response({"evidence": "q", "score": 4})])
    ledger_path = tmp_path / "ledger.csv"
    _, summary = score_items(
        [_item()],
        client,
        model="qwen2.5:3b",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(ledger_path),
        run_id="test",
        draws=1,
    )
    assert summary.total_cost_usd == 0.0
    assert CostLedger(ledger_path).total_usd() == 0.0


# ------------------------------------------------------- self-consistency


def _result(item_id: str, draws: tuple[int, ...]) -> MirroringScore:
    return MirroringScore(
        item_id=item_id,
        score=sum(draws) / len(draws),
        draws=draws,
        evidence="q",
        spread=float(max(draws) - min(draws)),
    )


def test_identical_draws_agree_exactly() -> None:
    assert self_consistency([_result("1", (4, 4, 4))])["exact_agreement"] == 1.0


def test_disagreeing_draws_show_their_spread() -> None:
    consistency = self_consistency([_result("1", (1, 3, 5))])
    assert consistency["exact_agreement"] == 0.0
    assert consistency["mean_spread"] == 4.0
    assert consistency["within_one_point"] == 0.0


def test_a_single_draw_has_no_consistency_to_report() -> None:
    assert self_consistency([_result("1", (4,))])["n"] == 0


# --------------------------------------------------------------------- AUC


def test_a_perfect_check_scores_one() -> None:
    assert auc_against([5, 5, 1, 1], [True, True, False, False]) == 1.0


def test_a_check_pointing_the_wrong_way_scores_below_chance() -> None:
    assert auc_against([1, 1, 5, 5], [True, True, False, False]) == 0.0


def test_a_check_that_says_the_same_thing_every_time_scores_chance() -> None:
    """The degenerate case: no ranking means no separation."""
    assert auc_against([5, 5, 5, 5], [True, True, False, False]) == 0.5


def test_build_items_keeps_the_coding_sheets_own_item_numbers() -> None:
    frame = pd.DataFrame(
        {"item": [7, 8], "stimulus_text": [STIMULUS] * 2, "generated_reply": [REPLY] * 2}
    )
    assert [item.item_id for item in build_items(frame)] == ["7", "8"]


# -------------------------------------------------------------- comparison


def _pairs(cell_ids: list[str], replies: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": cell_ids,
            "stimulus_text": [STIMULUS] * len(cell_ids),
            "generated_reply": replies,
        }
    )


def test_comparison_pairs_on_cell_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The rows are in a different order in each run, and the pairing still holds."""
    monkeypatch.setattr("thesis.analysis.mirroring_model.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("thesis.analysis.mirroring_model.COST_LEDGER", tmp_path / "ledger.csv")
    before = _pairs(["a", "b"], ["Can you send it?", "Can you handle it?"])
    after = _pairs(["b", "a"], ["I handled it.", "I sent it this morning."])
    client = _ScriptedClient(
        [_response({"evidence": "q", "score": score}) for score in (5, 5, 1, 1)]
    )
    comparison = compare_runs(before, after, client, model="qwen2.5:3b", draws=1)
    assert comparison.n_paired == 2
    assert comparison.before == 5.0
    assert comparison.after == 1.0
    assert comparison.change == -4.0


def test_comparison_refuses_two_runs_with_no_shared_cells(tmp_path: Path) -> None:
    client = _ScriptedClient([])
    with pytest.raises(ValueError, match="not the same design"):
        compare_runs(
            _pairs(["a"], ["one"]), _pairs(["z"], ["two"]), client, model="qwen2.5:3b", draws=1
        )
