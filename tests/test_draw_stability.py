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
import pytest

from thesis.analysis.draw_stability import (
    _measure_reliability,
    decision_stability,
    decision_stability_n,
    grid_draw_reliability,
    imperative_ratios,
    merge_draws,
    plot,
    spearman_brown,
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


# ---------------------------------------------------- grid-shaped reliability


def _reply_features_two_draws(
    imperative_ratio_1: list[float],
    imperative_ratio_2: list[float],
    hedge_rate_1: list[float],
    hedge_rate_2: list[float],
    decision_1: list[str],
    decision_2: list[str],
    *,
    shuffle_draw2: bool = False,
) -> pd.DataFrame:
    """A grid's `extract_q1_reply_features` shape, narrowed to the columns
    `grid_draw_reliability` needs: `cell_id` ending in `__r{replicate}`, and
    `replicate` itself."""
    n = len(imperative_ratio_1)
    draw1 = [
        {
            "cell_id": f"c{i}__r1",
            "replicate": 1,
            "decision": decision_1[i],
            "imperative_ratio": imperative_ratio_1[i],
            "hedge_rate": hedge_rate_1[i],
        }
        for i in range(n)
    ]
    draw2 = [
        {
            "cell_id": f"c{i}__r2",
            "replicate": 2,
            "decision": decision_2[i],
            "imperative_ratio": imperative_ratio_2[i],
            "hedge_rate": hedge_rate_2[i],
        }
        for i in range(n)
    ]
    if shuffle_draw2:
        draw2 = draw2[::-1]
    return pd.DataFrame(draw1 + draw2)


def _reply_features_n_draws(
    imperative_ratio: list[list[float]],
    hedge_rate: list[list[float]],
    decision: list[list[str]],
) -> pd.DataFrame:
    """Same shape as :func:`_reply_features_two_draws`, for any number of
    draws -- one list per draw, each holding one value per cell."""
    n_draws = len(imperative_ratio)
    n_cells = len(imperative_ratio[0])
    rows = [
        {
            "cell_id": f"c{i}__r{draw + 1}",
            "replicate": draw + 1,
            "decision": decision[draw][i],
            "imperative_ratio": imperative_ratio[draw][i],
            "hedge_rate": hedge_rate[draw][i],
        }
        for draw in range(n_draws)
        for i in range(n_cells)
    ]
    return pd.DataFrame(rows)


def test_grid_draw_reliability_is_perfect_when_draws_match() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    decisions = ["accept", "accept", "defer", "defer", "escalate"]
    frame = _reply_features_two_draws(values, values, values, values, decisions, decisions)

    result = grid_draw_reliability(frame)

    assert result.n_cells == 5
    assert result.imperative_ratio.pearson == 1.0
    assert result.imperative_ratio.icc == 1.0
    assert result.hedge_rate.pearson == 1.0
    assert result.decision.share_agree == 1.0
    assert result.decision.kappa == 1.0


def test_grid_draw_reliability_pairs_cells_not_row_order() -> None:
    """The merge is on cell identity, not row position -- draw 2 shuffled
    must give the same numbers as draw 2 in order."""
    values1 = [0.1, 0.3, 0.5, 0.2, 0.4]
    values2 = [0.2, 0.1, 0.6, 0.3, 0.5]
    decisions = ["accept"] * 5
    ordered = _reply_features_two_draws(values1, values2, values1, values2, decisions, decisions)
    shuffled = _reply_features_two_draws(
        values1, values2, values1, values2, decisions, decisions, shuffle_draw2=True
    )

    assert grid_draw_reliability(ordered).imperative_ratio.pearson == pytest.approx(
        grid_draw_reliability(shuffled).imperative_ratio.pearson
    )


def test_grid_draw_reliability_reports_spearman_brown_projections() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    decisions = ["accept"] * 5
    frame = _reply_features_two_draws(values, values, values, values, decisions, decisions)

    result = grid_draw_reliability(frame)

    assert result.imperative_ratio.spearman_brown_k2 == spearman_brown(
        result.imperative_ratio.pearson, 2
    )
    assert result.imperative_ratio.spearman_brown_k3 == spearman_brown(
        result.imperative_ratio.pearson, 3
    )


def test_grid_draw_reliability_requires_at_least_two_draws() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["c0__r1"],
            "replicate": [1],
            "decision": ["accept"],
            "imperative_ratio": [0.1],
            "hedge_rate": [0.0],
        }
    )
    with pytest.raises(ValueError, match="at least two draws"):
        grid_draw_reliability(frame)


def test_grid_draw_reliability_accepts_three_draws() -> None:
    """The crash found running the main Q1 grid's third draw overnight
    (PROGRESS_llms.md, Sep 23): grid_draw_reliability used to require
    exactly draws 1 and 2 and raised on anything else."""
    imperative_ratio = [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.05, 0.3, 0.25]]
    hedge_rate = [[0.0, 0.1, 0.2]] * 3
    decision = [["accept", "defer", "escalate"]] * 3
    frame = _reply_features_n_draws(imperative_ratio, hedge_rate, decision)

    result = grid_draw_reliability(frame)

    assert result.n_cells == 3
    assert result.k_draws == 3


def test_grid_draw_reliability_two_draw_numbers_are_unchanged_by_generalizing_to_n() -> None:
    """Golden values captured from the exactly-two-draws code before it was
    generalized to N draws -- a regression check that generalizing the
    formula did not change what it computes for the case every earlier
    section of the progress log already reported numbers for."""
    values1 = [0.1, 0.3, 0.5, 0.2, 0.4]
    values2 = [0.2, 0.1, 0.6, 0.3, 0.5]
    decisions = ["accept"] * 5
    frame = _reply_features_two_draws(values1, values2, values1, values2, decisions, decisions)

    result = grid_draw_reliability(frame)

    assert result.k_draws == 2
    assert result.imperative_ratio.icc == pytest.approx(0.758, abs=1e-3)
    assert result.imperative_ratio.pearson == pytest.approx(0.762, abs=1e-3)
    assert result.imperative_ratio.spearman == pytest.approx(0.7, abs=1e-3)


# --------------------------------------------------------- N-draw internals


def test_measure_reliability_pearson_is_the_mean_pairwise_correlation() -> None:
    """With 3 draws there is no single pair left to report -- pearson and
    spearman become the mean over every pair, computed here independently
    of the function under test."""
    d1 = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    d2 = pd.Series([0.2, 0.1, 0.6, 0.3, 0.5])
    d3 = pd.Series([0.15, 0.25, 0.2, 0.45, 0.4])
    expected_pearson = (d1.corr(d2) + d1.corr(d3) + d2.corr(d3)) / 3
    expected_spearman = (
        d1.corr(d2, method="spearman")
        + d1.corr(d3, method="spearman")
        + d2.corr(d3, method="spearman")
    ) / 3

    result = _measure_reliability("x", [d1, d2, d3])

    assert result.pearson == pytest.approx(round(expected_pearson, 3), abs=1e-3)
    assert result.spearman == pytest.approx(round(expected_spearman, 3), abs=1e-3)


def test_decision_stability_n_delegates_to_cohens_kappa_at_two_draws() -> None:
    """At exactly two draws, the N-draw path must match Cohen's kappa
    exactly, not approximate it through Fleiss' formula -- so every
    already-published two-draw kappa in the progress log stays correct."""
    d1 = pd.Series(["accept", "defer", "accept", "escalate", "defer"])
    d2 = pd.Series(["accept", "accept", "accept", "escalate", "escalate"])

    result = decision_stability_n([d1, d2])

    assert result.kappa == pytest.approx(0.412, abs=1e-3)
    assert result.share_agree == pytest.approx(0.6, abs=1e-3)


def test_decision_stability_n_is_one_for_full_agreement_across_three_draws() -> None:
    """Every item's three draws agree with each other, but different items
    use different categories, so this is not the degenerate all-one-category
    case (which leaves kappa undefined)."""
    d1 = pd.Series(["accept", "defer", "escalate"])
    d2 = pd.Series(["accept", "defer", "escalate"])
    d3 = pd.Series(["accept", "defer", "escalate"])

    result = decision_stability_n([d1, d2, d3])

    assert result.kappa == pytest.approx(1.0, abs=1e-6)
    assert result.share_agree == pytest.approx(1.0, abs=1e-6)
    assert result.n_agree == 3


def test_decision_stability_n_matches_fleiss_kappa_by_hand() -> None:
    """4 items, 3 raters, 2 categories -- worked by hand from Fleiss'
    formula: p_A=7/12, p_B=5/12, P_bar=2/3, P_e=37/72, kappa=11/35."""
    d1 = pd.Series(["A", "A", "A", "B"])
    d2 = pd.Series(["A", "A", "A", "B"])
    d3 = pd.Series(["A", "B", "B", "B"])

    result = decision_stability_n([d1, d2, d3])

    assert result.kappa == pytest.approx(11 / 35, abs=1e-3)
    assert result.n_agree == 2  # items 1 and 4, all three raters agree
    assert result.share_agree == pytest.approx(0.5, abs=1e-6)
    assert result.share_expected == pytest.approx(13 / 48, abs=1e-3)


def test_spearman_brown_is_identity_at_one_draw() -> None:
    assert spearman_brown(0.42, 1) == pytest.approx(0.42)


def test_spearman_brown_matches_section_50s_projection() -> None:
    """Section 50 measured r=0.15 between two draws of orders per reply, and
    reported that two draws would reach about 0.26 and three about 0.35."""
    assert spearman_brown(0.15, 2) == pytest.approx(0.26, abs=0.005)
    assert spearman_brown(0.15, 3) == pytest.approx(0.35, abs=0.005)
