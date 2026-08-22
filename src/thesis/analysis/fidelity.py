"""Q2 fidelity statistics: are generated responses similar to real ones?

Four techniques, each answering a different version of that question, over
output the judge package already knows how to produce:

- **Paired Wilcoxon** -- per rubric dimension, is there a systematic
  difference between a real reply and the generated reply to the *same*
  stimulus?
- **TOST equivalence** -- the same paired difference, but framed to let the
  data *positively* support similarity, not just fail to reject a null.
- **Discrimination AUC** -- from judge.discrimination's output: can the judge,
  shown one message with no other signal, directly guess its origin?
- **Model-free AUC** -- the same question with no LLM at all, so there is a
  fidelity number that does not depend on trusting the judge to be
  introspective about its own guessing.

**Why paired, not independent-groups, comparison is the design.** A real
reply and a generated reply to the *same* incoming message differ only in
who (or what) wrote the reply -- the stimulus, and therefore most of the
variance unrelated to authorship, is held constant. Comparing independent
groups instead would mean a chunk of the observed difference could just be
different stimuli being different, which is a confound this design avoids
by construction. The pairing is created upstream, by generating a reply to
the exact same message a real ``S_real_eval`` reply answers -- this module
only consumes already-paired scores, it does not create the pairing.

**Failing to reject a null is not evidence of similarity.** A paired
Wilcoxon with a large p-value only says the data did not detect a
difference -- it could just as easily mean the sample was too small to
detect a real one. TOST inverts the logic: it tests whether the difference
can be bounded *within* a pre-specified margin, which is the only way to
make a positive equivalence claim rather than an absence-of-evidence one.
The margin (:data:`DEFAULT_TOST_MARGIN`, ±0.4 rubric points) is fixed before
looking at any result, for the same reason every other frozen threshold in
this project is frozen in advance: choosing it after seeing the data would
let the margin be tuned to whatever answer is convenient.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

DEFAULT_TOST_MARGIN = 0.4
DEFAULT_ALPHA = 0.05


class UnequalLengthError(ValueError):
    """Raised when paired scores do not have matching lengths.

    A silent length mismatch would zip the shorter sequence against only
    part of the longer one, quietly comparing the wrong items to each other
    -- exactly the kind of bug that would not show up as a crash.
    """


def _check_paired(real: Sequence[float], generated: Sequence[float]) -> int:
    if len(real) != len(generated):
        msg = (
            f"paired scores must have equal length: {len(real)} real vs {len(generated)} generated"
        )
        raise UnequalLengthError(msg)
    if len(real) == 0:
        msg = "no paired scores to compare"
        raise UnequalLengthError(msg)
    return len(real)


@dataclass(frozen=True, slots=True)
class WilcoxonResult:
    """One dimension's paired Wilcoxon signed-rank test."""

    dimension: str
    n_pairs: int
    statistic: float
    p_value: float
    median_difference: float


def paired_wilcoxon(
    real_scores: Sequence[float], generated_scores: Sequence[float], dimension: str
) -> WilcoxonResult:
    """Is there a systematic difference between paired real and generated scores?

    ``median_difference`` is real minus generated: positive means real scored
    higher on this dimension. Ties (identical paired scores) are dropped by
    scipy's default "wilcox" zero-method, the standard handling for this test.
    """
    n_pairs = _check_paired(real_scores, generated_scores)
    real_arr = np.asarray(real_scores, dtype=float)
    gen_arr = np.asarray(generated_scores, dtype=float)
    differences = real_arr - gen_arr

    if np.all(differences == 0):
        # scipy raises on an all-zero difference vector rather than return a
        # p-value; a real result here (perfect agreement on every pair) is
        # meaningful and should be reported, not crash the analysis.
        return WilcoxonResult(
            dimension=dimension, n_pairs=n_pairs, statistic=0.0, p_value=1.0, median_difference=0.0
        )

    result = stats.wilcoxon(real_arr, gen_arr)
    return WilcoxonResult(
        dimension=dimension,
        n_pairs=n_pairs,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        median_difference=float(np.median(differences)),
    )


@dataclass(frozen=True, slots=True)
class TOSTResult:
    """One dimension's two-one-sided-tests equivalence result."""

    dimension: str
    n_pairs: int
    margin: float
    mean_difference: float
    p_lower: float
    p_upper: float
    alpha: float

    @property
    def equivalent(self) -> bool:
        """Equivalence is claimed only when *both* one-sided tests reject
        their null -- the defining logic of TOST. Either alone is not
        sufficient: rejecting only the lower bound says nothing about the
        upper one."""
        return self.p_lower < self.alpha and self.p_upper < self.alpha


def tost_equivalence(
    real_scores: Sequence[float],
    generated_scores: Sequence[float],
    dimension: str,
    *,
    margin: float = DEFAULT_TOST_MARGIN,
    alpha: float = DEFAULT_ALPHA,
) -> TOSTResult:
    """Two one-sided tests: is the paired mean difference within ``margin``?

    Built directly from a paired t-test rather than a dedicated TOST package,
    since the procedure is exactly two one-sided paired t-tests against the
    lower and upper equivalence bounds:

    - lower test: H0 is "true difference <= -margin", rejected when the
      difference is significantly greater than -margin.
    - upper test: H0 is "true difference >= +margin", rejected when the
      difference is significantly less than +margin.

    Both rejecting is what licenses the positive claim "the difference is
    within [-margin, +margin]", which is the equivalence claim itself.
    """
    n_pairs = _check_paired(real_scores, generated_scores)
    differences = np.asarray(real_scores, dtype=float) - np.asarray(generated_scores, dtype=float)
    mean_diff = float(np.mean(differences))
    se = float(np.std(differences, ddof=1) / np.sqrt(n_pairs)) if n_pairs > 1 else 0.0
    df = n_pairs - 1

    if se == 0.0:
        # Every pair had an identical difference: the mean is a certainty,
        # not an estimate, so it either sits inside the margin or it does
        # not, with nothing to test statistically.
        inside = -margin < mean_diff < margin
        p_lower = 0.0 if inside else 1.0
        p_upper = 0.0 if inside else 1.0
    else:
        t_lower = (mean_diff - (-margin)) / se
        t_upper = (mean_diff - margin) / se
        p_lower = float(1 - stats.t.cdf(t_lower, df))
        p_upper = float(stats.t.cdf(t_upper, df))

    return TOSTResult(
        dimension=dimension,
        n_pairs=n_pairs,
        margin=margin,
        mean_difference=mean_diff,
        p_lower=p_lower,
        p_upper=p_upper,
        alpha=alpha,
    )


class DegenerateLabelsError(ValueError):
    """Raised when every item shares one true label.

    An AUC is undefined with only one class present -- there is no ranking
    task to score. Reporting one anyway (as some libraries silently do, via
    a NaN or a default) would misrepresent "could not be computed" as "was
    computed and happened to be uninformative".
    """


def discrimination_auc(is_generated: Sequence[bool], likely_origin: Sequence[float]) -> float:
    """AUC for the judge's own discrimination ratings.

    ``likely_origin`` is scored high when the judge believes a message is
    real (see judge.discrimination), so the positive class here is "real"
    (``not is_generated``) -- an AUC near 0.5 means the rating carries no
    information about true origin (strong fidelity evidence), near 1.0 means
    the judge can reliably tell them apart.
    """
    n = _check_paired(is_generated, likely_origin)
    is_real = [not g for g in is_generated]
    if len(set(is_real)) < 2:
        msg = f"all {n} item(s) share one true label; AUC is undefined"
        raise DegenerateLabelsError(msg)

    return float(roc_auc_score(is_real, likely_origin))


def model_free_discrimination_auc(
    real_texts: Sequence[str],
    generated_texts: Sequence[str],
    *,
    n_folds: int = 5,
    seed: int = 20260807,
) -> float:
    """TF-IDF + logistic regression separating real from generated text.

    No LLM anywhere in this function -- the point is a fidelity number that
    does not depend on trusting the judge to accurately introspect about its
    own guessing. Cross-validated (default 5-fold, per the plan) rather than
    a single train/test split, since a single split's AUC on a small corpus
    is itself a noisy estimate of the thing being measured.
    """
    texts = list(real_texts) + list(generated_texts)
    labels = [0] * len(real_texts) + [1] * len(generated_texts)
    if len(set(labels)) < 2:
        msg = "need at least one real and one generated text"
        raise DegenerateLabelsError(msg)

    features = TfidfVectorizer(stop_words="english").fit_transform(texts)
    n_per_class = min(len(real_texts), len(generated_texts))
    folds = min(n_folds, n_per_class)
    if folds < 2:
        msg = f"need at least 2 examples of the rarer class for cross-validation, got {n_per_class}"
        raise DegenerateLabelsError(msg)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        LogisticRegression(max_iter=1000), features, labels, cv=cv, scoring="roc_auc"
    )
    return float(np.mean(scores))
