"""Q1: does hierarchical direction shape what a persona writes?

The design's manipulation is ``direction`` (up / lateral / down), and the
plan's own statistical warning is explicit and non-negotiable: *"power_score
is per-message, nested within sender... treating N messages as N independent
observations of ~150 people is a serious error and a guaranteed question."*
The same nesting exists here in miniature -- only 10 personas, each
contributing many generated replies -- so a plain t-test or chi-square
across all replies would silently treat "the same persona asked ten
questions" as ten independent people, inflating apparent significance.
Every function in this module accounts for that clustering explicitly,
never by ignoring it.

**Two outcome types, two different tools, one honest limitation.**

- A continuous linguistic feature (imperative_ratio, hedge_rate, ...) gets a
  proper mixed-effects model: ``feature ~ direction`` with a random intercept
  per persona (:func:`fit_direction_mixed_model`), exactly the
  ``statsmodels.MixedLM`` design the plan names directly.
- ``decision`` (accept/decline/defer/escalate/none) is categorical, and
  statsmodels has no clean support for a clustered multinomial model.
  :func:`direction_decision_association` is therefore a plain chi-square
  test of independence -- a real limitation, not a hidden one: it does not
  account for clustering by persona, so a significant result here is
  suggestive, not confirmatory, until a properly clustered categorical model
  exists. Stated once, here, rather than left for a reader to discover.

**Lateral is the reference level, not chosen by accident.** Lateral was
built into the design specifically as the no-power-difference baseline (see
``scenario.py``), so every coefficient in the mixed model is interpreted
against it: "writing up vs. a peer" and "writing down vs. a peer" are the
two contrasts that actually answer Q1, not an arbitrary alphabetical
reference statsmodels would otherwise pick on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from numpy.linalg import LinAlgError
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

# statsmodels' default L-BFGS optimizer can throw numpy.linalg.LinAlgError
# ("Singular matrix") when the true random-intercept variance is at or near
# the boundary of zero -- i.e. persona genuinely explains ~none of an
# outcome's variance, which is a real, expected result for some features
# (hedge_rate showed exactly this in the first Q1 pilot), not a data bug.
# Powell and Nelder-Mead are derivative-free and don't hit that singularity,
# so they're a robust fallback rather than the primary choice (they're
# slower and less precise when the optimization surface is well-behaved).
_OPTIMIZER_FALLBACKS: tuple[str, ...] = ("lbfgs", "powell", "nm")


class InsufficientDataError(ValueError):
    """Raised when there is not enough data to fit the requested model.

    A mixed model needs multiple groups and multiple observations per group
    to estimate a random-intercept variance at all; failing loudly here is
    better than statsmodels silently returning a degenerate or
    non-converged fit that looks like a real result.
    """


@dataclass(frozen=True, slots=True)
class MixedModelResult:
    """One fitted ``outcome ~ direction`` model, random intercept per persona."""

    outcome: str
    reference_level: str
    n_observations: int
    n_groups: int
    coefficients: dict[str, float]
    p_values: dict[str, float]
    group_variance: float
    converged: bool

    def contrast(self, level: str) -> tuple[float, float]:
        """Coefficient and p-value for one direction level vs. the reference.

        Raises KeyError with the available contrasts listed, rather than a
        bare statsmodels-style parameter name mismatch, if ``level`` was
        never a direction observed in the data.
        """
        key = f"direction[T.{level}]"
        if key not in self.coefficients:
            available = sorted(k for k in self.coefficients if k.startswith("direction["))
            msg = f"no contrast for {level!r}; available: {available}"
            raise KeyError(msg)
        return self.coefficients[key], self.p_values[key]


def fit_direction_mixed_model(
    df: pd.DataFrame,
    outcome_col: str,
    *,
    direction_col: str = "direction",
    cluster_col: str = "persona_id",
    reference: str = "lateral",
) -> MixedModelResult:
    """Fit ``outcome ~ direction`` with a random intercept per ``cluster_col``.

    ``reference`` sets which direction level every coefficient is measured
    against -- defaults to "lateral", the design's own no-power-difference
    baseline, not statsmodels' default alphabetical choice.
    """
    working = df[[outcome_col, direction_col, cluster_col]].dropna()
    n_groups = working[cluster_col].nunique()
    if len(working) < 3 or n_groups < 2:
        msg = (
            f"need at least 2 groups and 3 observations to fit a mixed model; "
            f"got {len(working)} observation(s) across {n_groups} group(s)"
        )
        raise InsufficientDataError(msg)

    levels = [reference, *sorted(lv for lv in working[direction_col].unique() if lv != reference)]
    working = working.assign(
        **{direction_col: pd.Categorical(working[direction_col], categories=levels, ordered=False)}
    )

    formula = f"{outcome_col} ~ C({direction_col}, Treatment(reference='{reference}'))"
    model = MixedLM.from_formula(formula, groups=working[cluster_col], data=working)

    fit = None
    last_error: Exception | None = None
    for method in _OPTIMIZER_FALLBACKS:
        try:
            fit = model.fit(reml=True, method=method)
        except (LinAlgError, ValueError) as exc:
            last_error = exc
            continue
        if fit.converged:
            break
    if fit is None:
        msg = f"no optimizer converged for {outcome_col} ~ {direction_col}: {last_error}"
        raise InsufficientDataError(msg)

    # statsmodels/patsy names a fixed-effect parameter after the full formula
    # term, e.g. "C(direction, Treatment(reference='lateral'))[T.up]" -- the
    # actual varying level is the unquoted suffix after "[T.", not (as a
    # first attempt here wrongly assumed) something matched by searching for
    # the level's name quoted in the string, which only ever matches the
    # reference level, present in every term. Renamed to the plain
    # "direction[T.up]" form contrast() expects, so callers never have to
    # know the formula's exact patsy spelling.
    coefficients: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for name, coef in fit.params.items():
        if name in ("Intercept", "Group Var"):
            coefficients[name] = float(coef)
            p_values[name] = float(fit.pvalues.get(name, float("nan")))
            continue
        if "[T." in name and name.endswith("]"):
            level = name.rsplit("[T.", 1)[1][:-1]
            clean_name = f"direction[T.{level}]"
            coefficients[clean_name] = float(coef)
            p_values[clean_name] = float(fit.pvalues.get(name, float("nan")))

    return MixedModelResult(
        outcome=outcome_col,
        reference_level=reference,
        n_observations=len(working),
        n_groups=n_groups,
        coefficients=coefficients,
        p_values=p_values,
        group_variance=float(fit.cov_re.iloc[0, 0]),
        converged=bool(fit.converged),
    )


@dataclass(frozen=True, slots=True)
class AssociationResult:
    """A chi-square test of independence between direction and decision.

    Deliberately does not claim to account for clustering by persona -- see
    the module docstring for why a clustered categorical model was not
    built. Treat a significant result here as suggestive, not confirmatory.
    """

    statistic: float
    p_value: float
    degrees_of_freedom: int
    n_observations: int
    contingency_table: pd.DataFrame


def direction_decision_association(
    df: pd.DataFrame, *, direction_col: str = "direction", decision_col: str = "decision"
) -> AssociationResult:
    """Chi-square test: is decision independent of direction?

    Not clustered by persona -- see the module and class docstrings. A
    useful, honestly-limited first pass, not the final word on Q1's
    decision-attitude claim.
    """
    working = df[[direction_col, decision_col]].dropna()
    if len(working) == 0:
        msg = "no observations to test"
        raise InsufficientDataError(msg)

    table = pd.crosstab(working[direction_col], working[decision_col])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    return AssociationResult(
        statistic=float(chi2),
        p_value=float(p_value),
        degrees_of_freedom=int(dof),
        n_observations=len(working),
        contingency_table=table,
    )


def summarize_by_direction(
    df: pd.DataFrame, outcome_cols: list[str], *, direction_col: str = "direction"
) -> pd.DataFrame:
    """Mean of each outcome by direction -- the plain descriptive table a
    reader checks the model's coefficients against."""
    return df.groupby(direction_col)[outcome_cols].agg(["mean", "count"])
