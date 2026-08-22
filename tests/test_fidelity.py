"""Fidelity-statistics tests.

No model calls, no judge output required: every function here takes plain
scores or text as input, so what's tested is the statistical logic itself --
against hand-computable cases wherever possible, not just "it runs".
"""

from __future__ import annotations

import pytest
from scipy import stats as scipy_stats

from thesis.analysis.fidelity import (
    DEFAULT_TOST_MARGIN,
    DegenerateLabelsError,
    UnequalLengthError,
    discrimination_auc,
    model_free_discrimination_auc,
    paired_wilcoxon,
    tost_equivalence,
)

# ------------------------------------------------------------- input checks


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(UnequalLengthError):
        paired_wilcoxon([1, 2, 3], [1, 2], "clarity")


def test_empty_input_is_rejected() -> None:
    with pytest.raises(UnequalLengthError):
        paired_wilcoxon([], [], "clarity")


# --------------------------------------------------------------- wilcoxon


def test_wilcoxon_matches_scipy_directly() -> None:
    """Not a reimplementation risk: confirms this module calls scipy the way
    it claims to, on a real (non-degenerate) paired dataset."""
    real = [4, 5, 3, 4, 5, 2, 4]
    generated = [3, 3, 3, 2, 4, 2, 3]
    expected = scipy_stats.wilcoxon(real, generated)

    result = paired_wilcoxon(real, generated, "clarity")
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.n_pairs == 7


def test_wilcoxon_median_difference_direction() -> None:
    """Positive means real scored higher -- the documented convention."""
    result = paired_wilcoxon([5, 5, 5], [3, 3, 3], "clarity")
    assert result.median_difference == pytest.approx(2.0)


def test_wilcoxon_handles_perfect_agreement_without_crashing() -> None:
    """scipy raises on an all-zero difference vector; a real, meaningful
    result (perfect agreement) must be reported, not crash the analysis."""
    result = paired_wilcoxon([3, 4, 5], [3, 4, 5], "clarity")
    assert result.median_difference == 0.0
    assert result.p_value == 1.0


# ------------------------------------------------------------------- TOST


def test_tost_mean_difference_is_hand_computable() -> None:
    real = [4.0, 4.0, 4.0, 4.0, 4.0]
    generated = [3.9, 3.9, 3.9, 3.9, 3.9]
    result = tost_equivalence(real, generated, "clarity", margin=0.4)
    assert result.mean_difference == pytest.approx(0.1)


def test_tost_declares_equivalence_for_a_tiny_well_supported_difference() -> None:
    """Five pairs, differing by a consistent 0.1 against a margin of 0.4:
    comfortably inside, and with near-zero variance, should read as
    equivalent."""
    real = [4.0, 4.1, 3.9, 4.0, 4.0]
    generated = [3.9, 4.0, 3.8, 3.9, 3.9]
    result = tost_equivalence(real, generated, "clarity", margin=0.4)
    assert result.equivalent is True


def test_tost_does_not_declare_equivalence_for_a_difference_beyond_the_margin() -> None:
    real = [5.0, 5.0, 5.0, 5.0, 5.0]
    generated = [2.0, 2.0, 2.0, 2.0, 2.0]
    result = tost_equivalence(real, generated, "clarity", margin=0.4)
    assert result.equivalent is False


def test_tost_zero_variance_inside_margin_is_equivalent_without_a_t_test() -> None:
    """Every pair has an identical difference: the mean is a certainty, not
    an estimate -- there is nothing to run a t-test on, but the equivalence
    question still has a definite answer."""
    result = tost_equivalence([4.0, 4.0, 4.0], [3.9, 3.9, 3.9], "clarity", margin=0.4)
    assert result.equivalent is True


def test_tost_zero_variance_outside_margin_is_not_equivalent() -> None:
    result = tost_equivalence([5.0, 5.0, 5.0], [1.0, 1.0, 1.0], "clarity", margin=0.4)
    assert result.equivalent is False


def test_default_margin_matches_the_planned_value() -> None:
    """0.4 rubric points was frozen in the plan before any result existed."""
    assert DEFAULT_TOST_MARGIN == 0.4


def test_tost_requires_both_one_sided_tests_to_reject() -> None:
    """A difference that clears the lower bound but not the upper one must
    not be reported as equivalent -- the core TOST logic, not an edge case."""
    real = [4.5, 4.5, 4.5, 4.5, 4.5]
    generated = [3.9, 3.9, 3.9, 3.9, 3.9]
    result = tost_equivalence(real, generated, "clarity", margin=0.4)
    assert result.p_upper > result.alpha
    assert result.equivalent is False


# --------------------------------------------------------- discrimination AUC


def test_discrimination_auc_matches_sklearn_directly() -> None:
    from sklearn.metrics import roc_auc_score

    is_generated = [True, True, False, False, True, False]
    likely_origin = [2, 3, 4, 5, 1, 3]
    expected = roc_auc_score([not g for g in is_generated], likely_origin)
    assert discrimination_auc(is_generated, likely_origin) == pytest.approx(expected)


def test_discrimination_auc_is_one_when_ratings_perfectly_separate_classes() -> None:
    is_generated = [True, True, False, False]
    likely_origin = [1, 2, 4, 5]  # generated always rated lower
    assert discrimination_auc(is_generated, likely_origin) == pytest.approx(1.0)


def test_discrimination_auc_is_half_when_ratings_carry_no_information() -> None:
    """AUC ~ 0.5 is the strong-fidelity result the plan names explicitly:
    the judge's rating is no better than chance at recovering true origin."""
    is_generated = [True, False, True, False]
    likely_origin = [3, 3, 3, 3]
    assert discrimination_auc(is_generated, likely_origin) == pytest.approx(0.5)


def test_discrimination_auc_rejects_a_single_class() -> None:
    """An AUC is undefined with only one true label present -- there is no
    ranking task to score, and silently returning a number here would
    misrepresent 'could not be computed' as a real result."""
    with pytest.raises(DegenerateLabelsError):
        discrimination_auc([True, True, True], [1, 2, 3])


# ------------------------------------------------------- model-free AUC


def test_model_free_auc_separates_clearly_distinct_vocabularies() -> None:
    """Not testing sklearn itself -- confirming this function's own plumbing
    (TF-IDF -> logistic regression -> stratified CV) produces a high AUC on
    text that is trivially separable by vocabulary alone."""
    real = [f"quarterly earnings report number {i}" for i in range(8)]
    generated = [f"puppy kitten rainbow sunshine {i}" for i in range(8)]
    auc = model_free_discrimination_auc(real, generated, n_folds=4)
    assert auc > 0.9


def test_model_free_auc_rejects_a_single_class() -> None:
    with pytest.raises(DegenerateLabelsError):
        model_free_discrimination_auc(["a", "b", "c"], [])


def test_model_free_auc_rejects_too_few_examples_for_cross_validation() -> None:
    """Fewer than 2 examples of the rarer class cannot be split into any
    number of folds; must fail clearly rather than let sklearn raise an
    opaque internal error."""
    with pytest.raises(DegenerateLabelsError):
        model_free_discrimination_auc(["only one real example"], ["gen a", "gen b", "gen c"])


def test_model_free_auc_is_deterministic_given_a_seed() -> None:
    real = [f"real message content {i}" for i in range(6)]
    generated = [f"synthetic generated text {i}" for i in range(6)]
    a = model_free_discrimination_auc(real, generated, n_folds=3, seed=1)
    b = model_free_discrimination_auc(real, generated, n_folds=3, seed=1)
    assert a == b
