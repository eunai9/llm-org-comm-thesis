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

**Direction and tone, fit separately, can each only report a main effect.**
Two calls to :func:`fit_direction_mixed_model` -- one with ``direction`` as
the factor, one with ``tone`` -- can show "hierarchy matters" and "incoming
tone doesn't," but neither can show whether tone's (null) effect is uniform
across direction, or concentrated in one corner of the grid (e.g. an
assertive message only changes the reply when writing up, not laterally or
down). :func:`fit_interaction_model` fits both factors and their product
in one model, so the interaction term itself -- not just the two main
effects -- gets a coefficient and a p-value.

**A per-reply rate is the wrong unit when replies are one sentence long.**
``imperative_ratio`` divides imperative sentences by total sentences, which
is a sensible continuous measure for a normal email and close to
meaningless for a one-sentence reply, where it can only be 0 or 1 -- a
coarse binary in a design built to detect an effect the size of a fraction
of that gap. :func:`fit_sentence_level_model` fits the data at the grain it
actually has: one binary observation per *sentence* (is this sentence
imperative?) in a logistic mixed model with a random intercept per persona,
rather than dividing a small integer by a smaller one first and losing most
of the sample's information to rounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy import stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
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


def _fit_with_fallback(model: MixedLM, *, label: str) -> Any:
    """Try each optimizer in :data:`_OPTIMIZER_FALLBACKS` in turn, keeping
    the first one that both runs and converges. Shared by every model in
    this module rather than duplicated per function."""
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
        msg = f"no optimizer converged for {label}: {last_error}"
        raise InsufficientDataError(msg)
    return fit


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
    fit = _fit_with_fallback(model, label=f"{outcome_col} ~ {direction_col}")

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
class SentenceModelResult:
    """One fitted ``outcome ~ direction`` logistic mixed model, one row per
    sentence, random intercept per persona.

    Fitted by variational Bayes (:mod:`statsmodels.genmod.bayes_mixed_glm`),
    not maximum likelihood -- ``coefficients`` are posterior means on the
    logit scale, and ``p_values`` are an approximate two-sided Wald test
    from the posterior mean and SD treated as asymptotically normal. That
    is a standard way to summarize a VB fit, but it is not the same
    calibrated quantity :class:`MixedModelResult`'s p-values are; read a
    contrast here as directional evidence; a value near a conventional
    cutoff should not be leaned on as heavily as the linear models' figures.
    """

    outcome: str
    reference_level: str
    n_observations: int
    n_groups: int
    coefficients: dict[str, float]
    posterior_sd: dict[str, float]
    p_values: dict[str, float]
    group_sd: float

    def contrast(self, level: str) -> tuple[float, float]:
        """Posterior mean and approximate p-value for one direction level
        vs. the reference, on the logit scale."""
        key = f"direction[T.{level}]"
        if key not in self.coefficients:
            available = sorted(k for k in self.coefficients if k.startswith("direction["))
            msg = f"no contrast for {level!r}; available: {available}"
            raise KeyError(msg)
        return self.coefficients[key], self.p_values[key]


def fit_sentence_level_model(
    df: pd.DataFrame,
    outcome_col: str,
    *,
    direction_col: str = "direction",
    cluster_col: str = "persona_id",
    reference: str = "lateral",
) -> SentenceModelResult:
    """Fit ``outcome ~ direction`` on one row per sentence, ``outcome_col``
    a 0/1 (or boolean) column, with a random intercept per ``cluster_col``.

    Use this instead of :func:`fit_direction_mixed_model` when the outcome
    is a per-sentence rate computed over very short text -- see the module
    docstring for why a rate over one or two sentences is too coarse an
    instrument for the effect size this design looks for. The input frame
    here has one row per *sentence*, not per reply: build it with
    :func:`thesis.data.features.extract_sentence_features` joined back onto
    each sentence's reply-level ``direction``/``cluster_col``.
    """
    working = df[[outcome_col, direction_col, cluster_col]].dropna()
    n_groups = working[cluster_col].nunique()
    if len(working) < 3 or n_groups < 2:
        msg = (
            f"need at least 2 groups and 3 observations to fit a mixed model; "
            f"got {len(working)} observation(s) across {n_groups} group(s)"
        )
        raise InsufficientDataError(msg)

    outcome_values = set(working[outcome_col].unique())
    if not outcome_values <= {0, 1}:  # True/False collapse into 1/0 in a set
        msg = f"{outcome_col!r} must be binary (0/1 or bool); got values {sorted(outcome_values)}"
        raise ValueError(msg)

    levels = [reference, *sorted(lv for lv in working[direction_col].unique() if lv != reference)]
    working = working.assign(
        **{
            direction_col: pd.Categorical(working[direction_col], categories=levels, ordered=False),
            outcome_col: working[outcome_col].astype(int),
        }
    )

    formula = f"{outcome_col} ~ C({direction_col}, Treatment(reference='{reference}'))"
    model = BinomialBayesMixedGLM.from_formula(
        formula, {cluster_col: f"0 + C({cluster_col})"}, data=working
    )
    fit = model.fit_vb()

    # Mirrors fit_direction_mixed_model's own renaming: statsmodels/patsy
    # names a fixed-effect parameter after the full formula term, e.g.
    # "C(direction, Treatment(reference='lateral'))[T.up]" -- stripped down
    # to the plain "direction[T.up]" form contrast() expects.
    coefficients: dict[str, float] = {}
    posterior_sd: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for name, mean, sd in zip(model.exog_names, fit.fe_mean, fit.fe_sd, strict=True):
        if name == "Intercept":
            clean_name = name
        else:
            level = name.rsplit("[T.", 1)[1][:-1]
            clean_name = f"direction[T.{level}]"
        z = mean / sd if sd > 0 else float("inf")
        coefficients[clean_name] = float(mean)
        posterior_sd[clean_name] = float(sd)
        p_values[clean_name] = float(2 * (1 - stats.norm.cdf(abs(z))))

    return SentenceModelResult(
        outcome=outcome_col,
        reference_level=reference,
        n_observations=len(working),
        n_groups=n_groups,
        coefficients=coefficients,
        posterior_sd=posterior_sd,
        p_values=p_values,
        group_sd=float(np.exp(fit.vcp_mean[0])),
    )


# Splits a raw patsy interaction term into its per-factor pieces on patsy's
# *own* ":" separator, never on a ":" that happens to sit inside a factor
# level's own text. Patsy always writes that separator immediately after one
# term's closing "]" and immediately before the next term's "C(" -- e.g.
# "...[T.up]:C(tone, ...)..." -- so anchoring the split there (rather than
# splitting on every literal ":", as a first version of this function did)
# correctly leaves a level like "llama3.2:3b" (an Ollama model id, not a
# hyphenated word -- the judge-swap generator/judge factors use raw model
# ids as their levels) untouched: that colon is never preceded by "]" and
# followed by "C(", so it never matches this boundary.
_INTERACTION_TERM_BOUNDARY = re.compile(r"(?<=\]):(?=C\()")


def _clean_interaction_term(name: str, factor1_col: str, factor2_col: str) -> str:
    """Turn one raw patsy parameter name into the plain ``factor[T.level]``
    form :class:`InteractionModelResult` uses, joining both sides with ``:``
    for an interaction term. Which factor a bare term belongs to is read off
    which column name appears inside its ``C(...)`` wrapper, since patsy's
    own spelling gives no shorter way to tell them apart."""
    parts = []
    for term in _INTERACTION_TERM_BOUNDARY.split(name):
        if f"C({factor1_col}," in term:
            col = factor1_col
        elif f"C({factor2_col}," in term:
            col = factor2_col
        else:
            msg = f"unrecognized patsy term {term!r}"
            raise ValueError(msg)
        level = term.rsplit("[T.", 1)[1][:-1]
        parts.append(f"{col}[T.{level}]")
    return ":".join(parts)


@dataclass(frozen=True, slots=True)
class InteractionModelResult:
    """One fitted ``outcome ~ factor1 * factor2`` model, random intercept
    per persona. Unlike two separate :func:`fit_direction_mixed_model` runs,
    this can show whether one factor's effect depends on the other."""

    outcome: str
    factor1: str
    factor2: str
    reference1: str
    reference2: str
    n_observations: int
    n_groups: int
    coefficients: dict[str, float]
    p_values: dict[str, float]
    group_variance: float
    converged: bool

    def main_effect(self, factor: str, level: str) -> tuple[float, float]:
        """Coefficient and p-value for one factor's level vs. its reference,
        holding the *other* factor at its own reference level -- not an
        average over the other factor's levels. See :meth:`interaction` for
        how that level combines with a specific level of the other factor.
        """
        key = f"{factor}[T.{level}]"
        if key not in self.coefficients:
            available = sorted(
                k for k in self.coefficients if k.startswith(f"{factor}[") and ":" not in k
            )
            msg = f"no main effect for {factor}={level!r}; available: {available}"
            raise KeyError(msg)
        return self.coefficients[key], self.p_values[key]

    def interaction(self, level1: str, level2: str) -> tuple[float, float]:
        """Coefficient and p-value for the ``factor1=level1, factor2=level2``
        interaction term -- how much that specific combination departs from
        what the two main effects alone would predict."""
        key = f"{self.factor1}[T.{level1}]:{self.factor2}[T.{level2}]"
        if key not in self.coefficients:
            available = sorted(k for k in self.coefficients if ":" in k)
            msg = (
                f"no interaction term for {self.factor1}={level1!r}, "
                f"{self.factor2}={level2!r}; available: {available}"
            )
            raise KeyError(msg)
        return self.coefficients[key], self.p_values[key]


def fit_interaction_model(
    df: pd.DataFrame,
    outcome_col: str,
    *,
    factor1_col: str = "direction",
    factor2_col: str = "tone",
    cluster_col: str = "persona_id",
    reference1: str = "lateral",
    reference2: str = "neutral",
) -> InteractionModelResult:
    """Fit ``outcome ~ factor1 * factor2`` with a random intercept per
    ``cluster_col``.

    Two separate calls to :func:`fit_direction_mixed_model` can each only
    report a main effect; this fits both factors and their interaction in
    one model, so a coefficient exists for "does factor1's effect change
    depending on factor2" rather than just "does factor1 matter" and "does
    factor2 matter" in isolation.
    """
    working = df[[outcome_col, factor1_col, factor2_col, cluster_col]].dropna()
    n_groups = working[cluster_col].nunique()
    if len(working) < 3 or n_groups < 2:
        msg = (
            f"need at least 2 groups and 3 observations to fit a mixed model; "
            f"got {len(working)} observation(s) across {n_groups} group(s)"
        )
        raise InsufficientDataError(msg)

    levels1 = [reference1, *sorted(lv for lv in working[factor1_col].unique() if lv != reference1)]
    levels2 = [reference2, *sorted(lv for lv in working[factor2_col].unique() if lv != reference2)]
    working = working.assign(
        **{
            factor1_col: pd.Categorical(working[factor1_col], categories=levels1, ordered=False),
            factor2_col: pd.Categorical(working[factor2_col], categories=levels2, ordered=False),
        }
    )

    formula = (
        f"{outcome_col} ~ C({factor1_col}, Treatment(reference='{reference1}')) "
        f"* C({factor2_col}, Treatment(reference='{reference2}'))"
    )
    model = MixedLM.from_formula(formula, groups=working[cluster_col], data=working)
    fit = _fit_with_fallback(model, label=f"{outcome_col} ~ {factor1_col} * {factor2_col}")

    coefficients: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for name, coef in fit.params.items():
        if name in ("Intercept", "Group Var"):
            coefficients[name] = float(coef)
            p_values[name] = float(fit.pvalues.get(name, float("nan")))
            continue
        clean_name = _clean_interaction_term(name, factor1_col, factor2_col)
        coefficients[clean_name] = float(coef)
        p_values[clean_name] = float(fit.pvalues.get(name, float("nan")))

    return InteractionModelResult(
        outcome=outcome_col,
        factor1=factor1_col,
        factor2=factor2_col,
        reference1=reference1,
        reference2=reference2,
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
