"""Tests for the draw-to-draw stability measures.

No model calls: both inputs are frames of replies that were already
generated. What is tested is the counting -- agreement, the cross-table,
kappa -- and that the two draws are paired on persona and message rather
than on row order, which is the mistake that would make any of these
numbers meaningless.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from thesis.analysis.draw_stability import (
    decision_stability,
    imperative_ratios,
    merge_draws,
    plot,
)


def _frame(
    decisions: list[str],
    *,
    role: str = "sim_local",
    replies: list[str] | None = None,
) -> pd.DataFrame:
    n = len(decisions)
    return pd.DataFrame(
        {
            "cell_id": [f"{role}__real_t{i}__uid{i}" for i in range(n)],
            "decision": decisions,
            "stimulus_text": ["Please send me the report." for _ in range(n)],
            "generated_reply": replies if replies is not None else ["I will send it."] * n,
        }
    )


def test_identical_draws_agree_completely() -> None:
    decisions = ["accept", "defer", "accept", "defer"]
    result = decision_stability(_frame(decisions), _frame(decisions))
    assert result.n == 4
    assert result.n_agree == 4
    assert result.share_agree == 1.0
    assert result.kappa == 1.0


def test_agreement_counts_only_matching_decisions() -> None:
    result = decision_stability(
        _frame(["accept", "accept", "defer", "defer"]),
        _frame(["accept", "defer", "defer", "accept"]),
    )
    assert result.n_agree == 2
    assert result.share_agree == 0.5


def test_cross_table_rows_are_the_first_draw() -> None:
    """Row = what draw 1 said, column = what draw 2 said. Reading it the
    other way round would reverse every swap the section describes."""
    result = decision_stability(
        _frame(["accept", "accept", "defer"]),
        _frame(["accept", "defer", "defer"]),
    )
    assert result.counts["accept"]["defer"] == 1
    assert result.counts["accept"]["accept"] == 1
    assert result.counts["defer"]["accept"] == 0
    assert result.counts["defer"]["defer"] == 1


def test_expected_agreement_comes_from_the_two_sets_of_totals() -> None:
    """Two accepts and two defers on each side. Two unrelated draws with
    those totals would match half the time, so kappa reads against 0.5."""
    result = decision_stability(
        _frame(["accept", "defer", "accept", "defer"]),
        _frame(["accept", "defer", "accept", "defer"]),
    )
    assert result.share_expected == 0.5


def test_kappa_is_near_zero_when_the_draws_are_unrelated() -> None:
    """Half the decisions match here, but so would half of two coin flips
    with these totals. Kappa is what says that plain agreement does not."""
    result = decision_stability(
        _frame(["accept", "accept", "defer", "defer"]),
        _frame(["accept", "defer", "accept", "defer"]),
    )
    assert result.share_agree == 0.5
    assert abs(result.kappa) < 0.01


def test_draws_are_paired_by_message_not_by_row_order() -> None:
    first = _frame(["accept", "defer", "escalate"])
    shuffled = _frame(["accept", "defer", "escalate"]).iloc[::-1].reset_index(drop=True)
    assert decision_stability(first, shuffled).n_agree == 3


def test_pairing_ignores_the_role_label() -> None:
    first = _frame(["accept", "defer"], role="sim_local")
    second = _frame(["accept", "defer"], role="sim_nvidia")
    assert len(merge_draws(first, second)) == 2


def test_no_shared_message_is_an_error() -> None:
    first = _frame(["accept"])
    second = _frame(["accept"])
    second["cell_id"] = ["sim_local__real_other__uid9"]
    try:
        merge_draws(first, second)
    except ValueError as exc:
        assert "not the same design" in str(exc)
    else:
        raise AssertionError("expected a ValueError for two unrelated runs")


def test_imperative_ratio_is_one_for_an_order_and_zero_for_a_statement() -> None:
    ratios = imperative_ratios(["a", "b"], ["Send me the file.", "The file is attached."])
    assert ratios.tolist() == [1.0, 0.0]


def test_plot_writes_a_figure(tmp_path: Path) -> None:
    result = decision_stability(_frame(["accept", "defer"]), _frame(["accept", "accept"]))
    path = plot(result, path=tmp_path / "test_retest_decision_agreement.png")
    assert path.is_file()
