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

**Every factor above is categorical. One question needs a continuous one.**
A dose-response design states a number in the prompt and asks how much the
output moves. Coding five target lengths as five unordered levels throws
away the ordering and gives four contrasts instead of one slope.
:func:`fit_dose_response_model` fits the dose as a continuous predictor, so
the answer is a single number. With both sides logged that number is an
elasticity: 1 means the output tracks the instruction exactly, 0 means the
instruction does nothing.

**Real email needs controls the simulator does not.** In the simulator every
persona writes in all three directions. In real email direction depends on
the writer's own rank: a junior employee cannot write down. So the direction
and sentence models take optional ``covariates`` (off by default).
:func:`fit_direction_fixed_effects` goes further and uses one dummy per
writer, so each contrast comes only from differences inside one writer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy.linalg import LinAlgError
from scipy import sparse, stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.regression.mixed_linear_model import MixedLM

# statsmodels' default L-BFGS optimizer can throw numpy.linalg.LinAlgError
# ("Singular matrix") when the true random-intercept variance is at or near
# the boundary of zero -- i.e. persona genuinely explains ~none of an
# outcome's variance, which is a real, expected result for some features
# (hedge_rate showed exactly this in the first Q1 pilot), not a data bug.
# Powell and Nelder-Mead are derivative-free and don't hit that singularity.
_OPTIMIZER_FALLBACKS: tuple[str, ...] = ("lbfgs", "powell", "nm")


def _fit_with_fallback(model: MixedLM, *, label: str) -> Any:
    """Fit with every optimizer in :data:`_OPTIMIZER_FALLBACKS` and keep the
    converged fit with the highest finite log-likelihood. Ties keep the
    earlier optimizer. If none qualifies, return the last fit that ran.

    Keeping the first fit that reports convergence is not safe. On the
    real-email Q1 data (PROGRESS.md section 48) L-BFGS reported convergence
    at a broken point: intercept and group variance both exactly zero, and
    an infinite log-likelihood. Powell and Nelder-Mead agreed on a finite,
    sensible fit. So an infinite log-likelihood counts as a failed fit.
    """
    best: Any = None
    last_fit: Any = None
    last_error: Exception | None = None
    for method in _OPTIMIZER_FALLBACKS:
        try:
            fit = model.fit(reml=True, method=method)
        except (LinAlgError, ValueError) as exc:
            last_error = exc
            continue
        last_fit = fit
        usable = fit.converged and np.isfinite(fit.llf)
        if usable and (best is None or fit.llf > best.llf):
            best = fit
    if best is not None:
        return best
    if last_fit is None:
        msg = f"no optimizer converged for {label}: {last_error}"
        raise InsufficientDataError(msg)
    return last_fit


def _direction_term(name: str, direction_col: str) -> str | None:
    """The plain ``direction[T.level]`` name for a patsy direction term, or
    None if ``name`` belongs to another term, such as a covariate.

    Matching on the ``C(direction_col,`` prefix matters once covariates exist.
    A covariate's own ``rank[T.3]`` term also contains ``[T.``, and must not
    be read as a direction contrast.
    """
    if not name.startswith(f"C({direction_col},"):
        return None
    level = name.rsplit("[T.", 1)[1][:-1]
    return f"direction[T.{level}]"


def _covariate_formula(covariates: Sequence[str]) -> str:
    """Covariates enter the formula as they are. A string or categorical
    column becomes dummies, a numeric column a slope."""
    return "".join(f" + {covariate}" for covariate in covariates)


def _indicator_matrix(values: pd.Series) -> sparse.csr_matrix:
    """One sparse 0/1 column per distinct value, in sorted order, as patsy's
    ``0 + C(col)`` would give densely."""
    codes, uniques = pd.factorize(values, sort=True)
    n_rows = len(codes)
    return sparse.csr_matrix(
        (np.ones(n_rows), (np.arange(n_rows), codes)), shape=(n_rows, len(uniques))
    )


# A draw's own index, appended by thesis.sim.grid.expand to every cell id
# ("...__r1", "...__r2", ...). Stripping it recovers the cell's identity --
# what two draws of the same cell have in common.
_REPLICATE_SUFFIX = re.compile(r"__r\d+$")


def cell_id_without_replicate(cell_ids: pd.Series) -> pd.Series:
    """A cell's identity: its ``cell_id`` with the trailing ``__r{replicate}``
    draw index removed.

    Two draws of the same cell get different ``cell_id`` values -- see
    :func:`thesis.sim.grid.expand`, which appends the suffix. This recovers
    what the two draws share, so they can be grouped back into one cell.
    """
    return cell_ids.str.replace(_REPLICATE_SUFFIX, "", regex=True)


def aggregate_replicates(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    cell_id_col: str = "cell_id",
    keep_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """One row per cell, each of ``value_cols`` averaged across its draws.

    Fitting a model on one row per draw would treat two draws of the same
    cell as two independent observations, which understates how uncertain
    the estimate really is. This collapses the draws first, so a model
    fitted on the result sees one row per cell, as the design intends.

    ``keep_cols`` (e.g. persona_id, direction) must be constant within a
    cell; the first row's value is kept. The result also carries
    ``n_draws``, how many rows each cell had before averaging.
    """
    working = df.assign(**{cell_id_col: cell_id_without_replicate(df[cell_id_col])})
    grouped = working.groupby(cell_id_col, as_index=False).agg(
        {**{col: "mean" for col in value_cols}, **{col: "first" for col in keep_cols}}
    )
    n_draws = working.groupby(cell_id_col).size().rename("n_draws")
    return grouped.merge(n_draws, on=cell_id_col)


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
    covariates: Sequence[str] = (),
) -> MixedModelResult:
    """Fit ``outcome ~ direction`` with a random intercept per ``cluster_col``.

    ``reference`` sets which direction level every coefficient is measured
    against -- defaults to "lateral", the design's own no-power-difference
    baseline, not statsmodels' default alphabetical choice.

    ``covariates`` are extra columns added as fixed effects. The default is
    none, which fits exactly the model this function always fitted. Their
    coefficients are kept under patsy's own names, e.g. ``rank[T.3]``.
    """
    working = df[[outcome_col, direction_col, cluster_col, *covariates]].dropna()
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

    formula = (
        f"{outcome_col} ~ C({direction_col}, Treatment(reference='{reference}'))"
        f"{_covariate_formula(covariates)}"
    )
    model = MixedLM.from_formula(formula, groups=working[cluster_col], data=working)
    fit = _fit_with_fallback(model, label=f"{outcome_col} ~ {direction_col}")

    # statsmodels/patsy names a fixed-effect parameter after the full formula
    # term, e.g. "C(direction, Treatment(reference='lateral'))[T.up]" -- the
    # actual varying level is the unquoted suffix after "[T.", not (as a
    # first attempt here wrongly assumed) something matched by searching for
    # the level's name quoted in the string, which only ever matches the
    # reference level, present in every term. Renamed to the plain
    # "direction[T.up]" form contrast() expects, so callers never have to
    # know the formula's exact patsy spelling. Covariate terms keep their
    # patsy names.
    coefficients: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for name, coef in fit.params.items():
        clean_name = _direction_term(name, direction_col) or name
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
    # SD of the second, nested random intercept (e.g. email within sender).
    # None when the model was fitted without one.
    nested_sd: float | None = None

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
    covariates: Sequence[str] = (),
    nested_col: str | None = None,
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

    ``covariates`` are extra fixed effects, as in
    :func:`fit_direction_mixed_model`. ``nested_col`` adds a second random
    intercept, for example one per email inside each sender. Its matrix is
    built sparse, because one dummy column per email is too large to hold
    dense. Both default to off, which fits exactly the model this function
    always fitted.
    """
    extra = [nested_col] if nested_col is not None else []
    working = df[[outcome_col, direction_col, cluster_col, *covariates, *extra]].dropna()
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

    formula = (
        f"{outcome_col} ~ C({direction_col}, Treatment(reference='{reference}'))"
        f"{_covariate_formula(covariates)}"
    )
    model = BinomialBayesMixedGLM.from_formula(
        formula, {cluster_col: f"0 + C({cluster_col})"}, data=working
    )
    if nested_col is not None:
        # Same fixed effects, rebuilt with a sparse matrix holding both
        # random intercepts: cluster first, nested second.
        blocks = [_indicator_matrix(working[col]) for col in (cluster_col, nested_col)]
        model = BinomialBayesMixedGLM(
            model.endog,
            pd.DataFrame(model.exog, columns=model.exog_names),
            sparse.hstack(blocks, format="csr"),
            np.concatenate([np.full(b.shape[1], i) for i, b in enumerate(blocks)]),
            vcp_names=[cluster_col, nested_col],
        )
        # BFGS keeps a dense matrix over every parameter, and one random
        # effect per email means thousands of them. L-BFGS-B gives the same
        # estimates to three decimals on the test fixture, about 80x faster.
        fit = model.fit_vb(fit_method="L-BFGS-B")
    else:
        fit = model.fit_vb()

    # Mirrors fit_direction_mixed_model's own renaming: statsmodels/patsy
    # names a fixed-effect parameter after the full formula term, e.g.
    # "C(direction, Treatment(reference='lateral'))[T.up]" -- stripped down
    # to the plain "direction[T.up]" form contrast() expects.
    coefficients: dict[str, float] = {}
    posterior_sd: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for name, mean, sd in zip(model.exog_names, fit.fe_mean, fit.fe_sd, strict=True):
        clean_name = _direction_term(name, direction_col) or name
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
        nested_sd=float(np.exp(fit.vcp_mean[1])) if nested_col is not None else None,
    )


@dataclass(frozen=True, slots=True)
class FixedEffectsResult:
    """One ``outcome ~ direction + cluster dummies`` fit, with standard
    errors clustered by the same column.

    The dummies absorb everything constant within a cluster, such as a
    sender's rank. So each direction contrast uses only differences inside
    one cluster.
    """

    outcome: str
    reference_level: str
    family: str
    n_observations: int
    n_groups: int
    n_groups_dropped: int
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    p_values: dict[str, float]

    def contrast(self, level: str) -> tuple[float, float]:
        """Coefficient and p-value for one direction level vs. the reference."""
        key = f"direction[T.{level}]"
        if key not in self.coefficients:
            msg = f"no contrast for {level!r}; available: {sorted(self.coefficients)}"
            raise KeyError(msg)
        return self.coefficients[key], self.p_values[key]


def fit_direction_fixed_effects(
    df: pd.DataFrame,
    outcome_col: str,
    *,
    direction_col: str = "direction",
    cluster_col: str = "persona_id",
    reference: str = "lateral",
    family: Literal["linear", "logistic"] = "linear",
) -> FixedEffectsResult:
    """Fit ``outcome ~ direction + C(cluster)`` with cluster-robust errors.

    This is the within-cluster check next to the mixed models. A random
    intercept assumes the cluster effect is unrelated to direction. That
    fails when direction depends on who the cluster is, as a sender's rank
    limits which directions they can write in. Cluster dummies make no such
    assumption.

    ``family="logistic"`` fits a logit with the same dummies. A cluster whose
    outcome never varies has an infinite dummy and says nothing about
    direction, so it is dropped first and counted in ``n_groups_dropped``.
    """
    working = df[[outcome_col, direction_col, cluster_col]].dropna()
    n_dropped = 0
    if family == "logistic":
        working = working.assign(**{outcome_col: working[outcome_col].astype(int)})
        cluster_mean = working.groupby(cluster_col)[outcome_col].transform("mean")
        varies = (cluster_mean > 0) & (cluster_mean < 1)
        n_dropped = working.loc[~varies, cluster_col].nunique()
        working = working[varies]

    n_groups = working[cluster_col].nunique()
    if len(working) < 3 or n_groups < 2:
        msg = (
            f"need at least 2 groups and 3 observations to fit a fixed-effects model; "
            f"got {len(working)} observation(s) across {n_groups} group(s)"
        )
        raise InsufficientDataError(msg)

    levels = [reference, *sorted(lv for lv in working[direction_col].unique() if lv != reference)]
    working = working.assign(
        **{
            direction_col: pd.Categorical(working[direction_col], categories=levels),
            cluster_col: working[cluster_col].astype(str),
        }
    )
    formula = (
        f"{outcome_col} ~ C({direction_col}, Treatment(reference='{reference}'))"
        f" + C({cluster_col})"
    )
    groups = pd.factorize(working[cluster_col])[0]
    if family == "logistic":
        model = smf.glm(formula, data=working, family=sm.families.Binomial())
    else:
        model = smf.ols(formula, data=working)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})

    coefficients: dict[str, float] = {}
    std_errors: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for name in fit.params.index:
        clean_name = _direction_term(name, direction_col)
        if clean_name is None:
            continue
        coefficients[clean_name] = float(fit.params[name])
        std_errors[clean_name] = float(fit.bse[name])
        p_values[clean_name] = float(fit.pvalues[name])

    return FixedEffectsResult(
        outcome=outcome_col,
        reference_level=reference,
        family=family,
        n_observations=len(working),
        n_groups=n_groups,
        n_groups_dropped=int(n_dropped),
        coefficients=coefficients,
        std_errors=std_errors,
        p_values=p_values,
    )


@dataclass(frozen=True, slots=True)
class DoseResponseResult:
    """One fitted ``outcome ~ dose`` model, random intercept per item.

    ``slope`` is how much the outcome moves when the dose moves by one unit.
    Log both sides before fitting and the slope is an elasticity: 1 means the
    outcome tracks the dose exactly, 0 means the dose does nothing, and 0.2
    means a doubling of the dose buys about a 15% rise in the outcome.

    ``conf_low`` and ``conf_high`` are the 95% interval for the slope. They
    are reported because the headline claim is usually "the slope is near
    zero", and a point estimate alone cannot say how near.
    """

    outcome: str
    dose: str
    n_observations: int
    n_groups: int
    intercept: float
    slope: float
    slope_se: float
    p_value: float
    conf_low: float
    conf_high: float
    group_variance: float
    converged: bool


def fit_dose_response_model(
    df: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    *,
    cluster_col: str = "cell_id",
) -> DoseResponseResult:
    """Fit ``outcome ~ dose`` with the dose continuous and a random intercept
    per ``cluster_col``.

    Use this when the manipulated factor is a number the design chose, not a
    category. Every other model in this module treats its factor as unordered
    levels, which costs one contrast per level and never yields a slope.

    ``cluster_col`` defaults to ``cell_id`` rather than ``persona_id`` because
    the repeated unit in a dose-response run is the item: the same stimulus is
    answered once per dose level, so the item is what observations pair on.
    """
    working = df[[outcome_col, dose_col, cluster_col]].dropna()
    n_groups = working[cluster_col].nunique()
    n_doses = working[dose_col].nunique()
    if len(working) < 3 or n_groups < 2:
        msg = (
            f"need at least 2 groups and 3 observations to fit a mixed model; "
            f"got {len(working)} observation(s) across {n_groups} group(s)"
        )
        raise InsufficientDataError(msg)
    if n_doses < 2:
        msg = f"need at least 2 distinct {dose_col!r} values to estimate a slope; got {n_doses}"
        raise InsufficientDataError(msg)

    formula = f"{outcome_col} ~ {dose_col}"
    model = MixedLM.from_formula(formula, groups=working[cluster_col], data=working)
    fit = _fit_with_fallback(model, label=formula)

    interval = fit.conf_int()
    return DoseResponseResult(
        outcome=outcome_col,
        dose=dose_col,
        n_observations=len(working),
        n_groups=n_groups,
        intercept=float(fit.params["Intercept"]),
        slope=float(fit.params[dose_col]),
        slope_se=float(fit.bse[dose_col]),
        p_value=float(fit.pvalues[dose_col]),
        conf_low=float(interval.loc[dose_col].iloc[0]),
        conf_high=float(interval.loc[dose_col].iloc[1]),
        group_variance=float(fit.cov_re.iloc[0, 0]),
        converged=bool(fit.converged),
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
    std_errors: dict[str, float]
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

    def main_effect_std_error(self, factor: str, level: str) -> float:
        """How precisely one main effect is measured, on the outcome's own
        scale. Smaller is better.

        A separate method rather than a third element in
        :meth:`main_effect`'s tuple, so every existing caller keeps working.
        """
        self.main_effect(factor, level)
        return self.std_errors[f"{factor}[T.{level}]"]

    def interaction_std_error(self, level1: str, level2: str) -> float:
        """How precisely the interaction term is measured. Smaller is better."""
        self.interaction(level1, level2)
        return self.std_errors[f"{self.factor1}[T.{level1}]:{self.factor2}[T.{level2}]"]


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
    std_errors: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for name, coef in fit.params.items():
        if name in ("Intercept", "Group Var"):
            coefficients[name] = float(coef)
            std_errors[name] = float(fit.bse.get(name, float("nan")))
            p_values[name] = float(fit.pvalues.get(name, float("nan")))
            continue
        clean_name = _clean_interaction_term(name, factor1_col, factor2_col)
        coefficients[clean_name] = float(coef)
        std_errors[clean_name] = float(fit.bse.get(name, float("nan")))
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
        std_errors=std_errors,
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
