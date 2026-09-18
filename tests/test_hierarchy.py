"""Q1 hierarchy-analysis tests.

Every mixed-model test injects a *known* effect into synthetic data and
checks the fitted coefficient recovers it -- not just that the function
runs. This is what caught a real bug during development: the first
implementation's coefficient-name parsing matched the wrong substring in
statsmodels' patsy-generated parameter names and silently produced an
empty coefficient dict, which a "does it run without crashing" test alone
would never have caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.linalg import LinAlgError

from thesis.analysis.hierarchy import (
    AssociationResult,
    DoseResponseResult,
    FixedEffectsResult,
    InsufficientDataError,
    InteractionModelResult,
    MixedModelResult,
    SentenceModelResult,
    _fit_with_fallback,
    aggregate_replicates,
    cell_id_without_replicate,
    direction_decision_association,
    fit_direction_fixed_effects,
    fit_direction_mixed_model,
    fit_dose_response_model,
    fit_interaction_model,
    fit_sentence_level_model,
    summarize_by_direction,
)


def _clustered_data(
    effects: dict[str, float], *, n_personas: int = 8, n_per_cell: int = 20, seed: int = 1
) -> pd.DataFrame:
    """Synthetic data with a known per-direction effect and real
    persona-level clustering (so the mixed model has something to
    estimate a random intercept from)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_personas):
        persona_id = f"p{i}"
        persona_offset = rng.normal(0, 0.3)
        for direction, effect in effects.items():
            for _ in range(n_per_cell):
                rows.append(
                    {
                        "persona_id": persona_id,
                        "direction": direction,
                        "outcome": 2.0 + effect + persona_offset + rng.normal(0, 0.15),
                    }
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------- mixed model


def test_recovers_a_known_injected_effect() -> None:
    """The core regression test for the coefficient-parsing bug: a large,
    known effect must come back close to its true value, not silently
    missing from the result."""
    df = _clustered_data({"lateral": 0.0, "up": 0.5, "down": 0.05})
    result = fit_direction_mixed_model(df, "outcome", reference="lateral")

    up_coef, up_p = result.contrast("up")
    down_coef, _down_p = result.contrast("down")

    assert up_coef == pytest.approx(0.5, abs=0.15)
    assert up_p < 0.01
    assert down_coef == pytest.approx(0.05, abs=0.15)


def test_reference_level_has_no_contrast_of_its_own() -> None:
    """Lateral is the baseline the other two are measured against -- it
    should not appear as its own contrast key."""
    df = _clustered_data({"lateral": 0.0, "up": 0.3, "down": 0.1})
    result = fit_direction_mixed_model(df, "outcome", reference="lateral")
    assert "direction[T.lateral]" not in result.coefficients


def test_contrast_raises_a_clear_error_for_an_unobserved_level() -> None:
    df = _clustered_data({"lateral": 0.0, "up": 0.2})
    result = fit_direction_mixed_model(df, "outcome", reference="lateral")
    with pytest.raises(KeyError, match="down"):
        result.contrast("down")


def test_reference_level_is_configurable() -> None:
    """Confirms the reference is genuinely respected, not hardcoded to
    'lateral' despite the parameter -- switching it changes which
    contrasts exist."""
    df = _clustered_data({"lateral": 0.0, "up": 0.3, "down": 0.1})
    result = fit_direction_mixed_model(df, "outcome", reference="up")
    assert result.reference_level == "up"
    assert "direction[T.lateral]" in result.coefficients
    assert "direction[T.up]" not in result.coefficients


def test_group_variance_is_reported() -> None:
    """The random-intercept variance itself: whether clustering by persona
    actually matters is part of the result, not just the fixed effects."""
    df = _clustered_data({"lateral": 0.0, "up": 0.1})
    result = fit_direction_mixed_model(df, "outcome")
    assert result.group_variance >= 0.0


def test_fits_when_group_variance_is_at_the_zero_boundary() -> None:
    """A regression test for a real crash: when persona genuinely explains
    ~none of an outcome's variance, statsmodels' default L-BFGS optimizer can
    throw numpy.linalg.LinAlgError deep inside its own score computation,
    because the true random-intercept variance sits right at the boundary of
    zero. This is a real, expected result (hedge_rate showed exactly this in
    the first Q1 pilot on live data), not a data bug, so the fit must
    succeed via a fallback optimizer rather than propagate the crash."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(10):
        persona_id = f"p{i}"
        for direction in ("up", "lateral", "down"):
            for _ in range(8):
                rows.append(
                    {
                        "persona_id": persona_id,
                        "direction": direction,
                        # No persona_offset term at all -- zero real
                        # clustering by construction, the boundary case.
                        "outcome": rng.choice([0.0, 0.5, 1.0]),
                    }
                )
    df = pd.DataFrame(rows)
    result = fit_direction_mixed_model(df, "outcome", reference="lateral")
    assert result.group_variance == pytest.approx(0.0, abs=0.05)


def test_rejects_too_few_groups() -> None:
    df = _clustered_data({"lateral": 0.0, "up": 0.1}, n_personas=1)
    with pytest.raises(InsufficientDataError):
        fit_direction_mixed_model(df, "outcome")


def test_rejects_empty_data() -> None:
    empty = pd.DataFrame({"outcome": [], "direction": [], "persona_id": []})
    with pytest.raises(InsufficientDataError):
        fit_direction_mixed_model(empty, "outcome")


def test_drops_rows_with_missing_values_rather_than_erroring() -> None:
    df = _clustered_data({"lateral": 0.0, "up": 0.3}, n_personas=6, n_per_cell=10)
    df.loc[0, "outcome"] = None
    result = fit_direction_mixed_model(df, "outcome")
    assert result.n_observations == len(df) - 1


def test_custom_column_names_are_respected() -> None:
    df = _clustered_data({"lateral": 0.0, "up": 0.3})
    df = df.rename(columns={"outcome": "hedge_rate", "persona_id": "who", "direction": "dir"})
    result = fit_direction_mixed_model(
        df, "hedge_rate", direction_col="dir", cluster_col="who", reference="lateral"
    )
    assert result.outcome == "hedge_rate"
    assert result.contrast("up")[0] == pytest.approx(0.3, abs=0.2)


# --------------------------------------------------------- sentence-level


def _sentence_data(
    effects: dict[str, float], *, n_personas: int = 8, n_per_cell: int = 40, seed: int = 7
) -> pd.DataFrame:
    """Synthetic one-row-per-sentence binary data. ``effects`` are on the
    logit scale, added to a per-persona random offset before drawing a
    Bernoulli outcome -- the sentence-level analogue of ``_clustered_data``."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_personas):
        persona_id = f"p{i}"
        persona_offset = rng.normal(0, 0.4)
        for direction, effect in effects.items():
            prob = 1.0 / (1.0 + np.exp(-(persona_offset + effect)))
            for _ in range(n_per_cell):
                rows.append(
                    {
                        "persona_id": persona_id,
                        "direction": direction,
                        "is_imperative": int(rng.random() < prob),
                    }
                )
    return pd.DataFrame(rows)


def test_sentence_model_recovers_a_known_injected_effect() -> None:
    """The sentence-level analogue of the module's core regression test: a
    large, known logit-scale effect must come back close to its true value
    and clearly distinguishable from a near-null one, not lost in the
    approximation VB fitting makes."""
    df = _sentence_data({"lateral": -0.3, "up": 1.5, "down": 0.0})
    result = fit_sentence_level_model(df, "is_imperative", reference="lateral")

    up_coef, up_p = result.contrast("up")
    down_coef, _down_p = result.contrast("down")

    assert up_coef == pytest.approx(1.5, abs=0.6)
    assert up_p < 0.01
    assert down_coef == pytest.approx(0.0, abs=0.6)


def test_sentence_model_accepts_boolean_outcome() -> None:
    df = _sentence_data({"lateral": -0.3, "up": 1.5})
    df["is_imperative"] = df["is_imperative"].astype(bool)
    result = fit_sentence_level_model(df, "is_imperative", reference="lateral")
    assert result.contrast("up")[0] == pytest.approx(1.5, abs=0.6)


def test_sentence_model_rejects_a_non_binary_outcome() -> None:
    df = _sentence_data({"lateral": 0.0, "up": 0.5})
    df["is_imperative"] = df["is_imperative"] + 1  # now {1, 2}, not {0, 1}
    with pytest.raises(ValueError, match="binary"):
        fit_sentence_level_model(df, "is_imperative")


def test_sentence_model_reference_level_has_no_contrast_of_its_own() -> None:
    df = _sentence_data({"lateral": 0.0, "up": 0.8, "down": 0.3})
    result = fit_sentence_level_model(df, "is_imperative", reference="lateral")
    assert "direction[T.lateral]" not in result.coefficients


def test_sentence_model_contrast_raises_a_clear_error_for_an_unobserved_level() -> None:
    df = _sentence_data({"lateral": 0.0, "up": 0.5})
    result = fit_sentence_level_model(df, "is_imperative", reference="lateral")
    with pytest.raises(KeyError, match="down"):
        result.contrast("down")


def test_sentence_model_reference_level_is_configurable() -> None:
    df = _sentence_data({"lateral": 0.0, "up": 0.6, "down": 0.2})
    result = fit_sentence_level_model(df, "is_imperative", reference="up")
    assert result.reference_level == "up"
    assert "direction[T.lateral]" in result.coefficients
    assert "direction[T.up]" not in result.coefficients


def test_sentence_model_group_sd_is_reported() -> None:
    df = _sentence_data({"lateral": 0.0, "up": 0.4})
    result = fit_sentence_level_model(df, "is_imperative")
    assert result.group_sd >= 0.0


def test_sentence_model_rejects_too_few_groups() -> None:
    df = _sentence_data({"lateral": 0.0, "up": 0.5}, n_personas=1)
    with pytest.raises(InsufficientDataError):
        fit_sentence_level_model(df, "is_imperative")


def test_sentence_model_rejects_empty_data() -> None:
    empty = pd.DataFrame({"is_imperative": [], "direction": [], "persona_id": []})
    with pytest.raises(InsufficientDataError):
        fit_sentence_level_model(empty, "is_imperative")


def test_sentence_model_result_is_frozen() -> None:
    df = _sentence_data({"lateral": 0.0, "up": 0.5})
    result = fit_sentence_level_model(df, "is_imperative")
    assert isinstance(result, SentenceModelResult)
    with pytest.raises(AttributeError):
        result.n_observations = 999  # type: ignore[misc]


def _two_factor_data(
    cell_effects: dict[tuple[str, str], float],
    *,
    factor1_levels: tuple[str, ...] = ("lateral", "up", "down"),
    factor2_levels: tuple[str, ...] = ("neutral", "deferential", "warm", "assertive"),
    n_personas: int = 8,
    n_per_cell: int = 15,
    seed: int = 5,
) -> pd.DataFrame:
    """Synthetic data crossing two factors, with real persona-level
    clustering and a per-(factor1, factor2)-cell effect. Cells not named in
    ``cell_effects`` default to 0.0, so an effect placed at a single cell
    tests the interaction term specifically, not either main effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_personas):
        persona_id = f"p{i}"
        persona_offset = rng.normal(0, 0.2)
        for level1 in factor1_levels:
            for level2 in factor2_levels:
                effect = cell_effects.get((level1, level2), 0.0)
                for _ in range(n_per_cell):
                    rows.append(
                        {
                            "persona_id": persona_id,
                            "direction": level1,
                            "tone": level2,
                            "outcome": 2.0 + effect + persona_offset + rng.normal(0, 0.15),
                        }
                    )
    return pd.DataFrame(rows)


# ------------------------------------------------------------ interaction


def test_recovers_a_known_injected_interaction_effect() -> None:
    """A real effect placed at exactly one (direction, tone) cell should
    show up in the interaction term, not in either main effect -- treatment
    coding puts a main effect's coefficient at the *other* factor's
    reference level, which this injected effect never touches."""
    df = _two_factor_data({("up", "assertive"): 0.6})
    result = fit_interaction_model(df, "outcome", reference1="lateral", reference2="neutral")

    interaction_coef, interaction_p = result.interaction("up", "assertive")
    assert interaction_coef == pytest.approx(0.6, abs=0.2)
    assert interaction_p < 0.01

    up_coef, _ = result.main_effect("direction", "up")
    assertive_coef, _ = result.main_effect("tone", "assertive")
    assert up_coef == pytest.approx(0.0, abs=0.2)
    assert assertive_coef == pytest.approx(0.0, abs=0.2)


def test_main_effect_is_recovered_when_uniform_across_the_other_factor() -> None:
    """A direction effect that holds at every tone level should show up as
    a main effect, with no real interaction term."""
    df = _two_factor_data(
        {("up", level2): 0.4 for level2 in ("neutral", "deferential", "warm", "assertive")}
    )
    result = fit_interaction_model(df, "outcome", reference1="lateral", reference2="neutral")
    up_coef, up_p = result.main_effect("direction", "up")
    assert up_coef == pytest.approx(0.4, abs=0.2)
    assert up_p < 0.01

    interaction_coef, _ = result.interaction("up", "assertive")
    assert interaction_coef == pytest.approx(0.0, abs=0.2)


def test_main_effect_raises_a_clear_error_for_an_unobserved_level() -> None:
    df = _two_factor_data({("up", "assertive"): 0.5})
    result = fit_interaction_model(df, "outcome")
    with pytest.raises(KeyError, match="direction"):
        result.main_effect("direction", "nonexistent")


def test_interaction_raises_a_clear_error_for_an_unobserved_combination() -> None:
    df = _two_factor_data({("up", "assertive"): 0.5})
    result = fit_interaction_model(df, "outcome")
    with pytest.raises(KeyError, match="warm"):
        result.interaction("down", "nonexistent-tone-level")


def test_interaction_model_rejects_too_few_groups() -> None:
    df = _two_factor_data({("up", "assertive"): 0.5}, n_personas=1)
    with pytest.raises(InsufficientDataError):
        fit_interaction_model(df, "outcome")


def test_interaction_model_result_is_frozen() -> None:
    df = _two_factor_data({("up", "assertive"): 0.5})
    result = fit_interaction_model(df, "outcome")
    assert isinstance(result, InteractionModelResult)
    with pytest.raises(AttributeError):
        result.n_observations = 999  # type: ignore[misc]


def test_interaction_model_custom_factor_names_are_respected() -> None:
    df = _two_factor_data({("up", "assertive"): 0.5})
    df = df.rename(columns={"direction": "dir", "tone": "incoming_tone"})
    result = fit_interaction_model(
        df,
        "outcome",
        factor1_col="dir",
        factor2_col="incoming_tone",
        reference1="lateral",
        reference2="neutral",
    )
    coef, _ = result.interaction("up", "assertive")
    assert coef == pytest.approx(0.5, abs=0.2)


def test_interaction_model_handles_a_colon_inside_a_factor_level() -> None:
    """A factor level containing a colon -- e.g. an Ollama model id like
    ``llama3.2:3b`` -- must not be confused with patsy's own ``:``
    interaction-term separator. This is a regression test for a real crash
    (``analysis.judge_swap``'s generator/judge factors are raw model ids):
    ``_clean_interaction_term`` used to split a raw parameter name on every
    literal ``:``, which shredded a level's own colon along with patsy's,
    and raised an ``IndexError`` trying to read a ``[T.`` marker out of a
    fragment that never had one."""
    df = _two_factor_data(
        {("qwen2.5:3b", "qwen2.5:3b"): 0.5},
        factor1_levels=("llama3.2:3b", "qwen2.5:3b"),
        factor2_levels=("llama3.2:3b", "qwen2.5:3b"),
    )
    result = fit_interaction_model(
        df, "outcome", reference1="llama3.2:3b", reference2="llama3.2:3b"
    )

    interaction_coef, interaction_p = result.interaction("qwen2.5:3b", "qwen2.5:3b")
    assert interaction_coef == pytest.approx(0.5, abs=0.2)
    assert interaction_p < 0.01

    main_coef, _ = result.main_effect("direction", "qwen2.5:3b")
    assert main_coef == pytest.approx(0.0, abs=0.2)


# ------------------------------------------------------------ association


def test_association_detects_a_strong_relationship() -> None:
    """Direction perfectly determines decision by construction; the
    association test must find that, not report independence."""
    rows = []
    mapping = {"up": "escalate", "lateral": "accept", "down": "decline"}
    for direction, decision in mapping.items():
        for _ in range(30):
            rows.append({"direction": direction, "decision": decision})
    df = pd.DataFrame(rows)

    result = direction_decision_association(df)
    assert result.p_value < 0.001
    assert result.n_observations == 90


def test_association_finds_no_relationship_when_none_exists() -> None:
    rng = np.random.default_rng(0)
    decisions = ["accept", "decline", "defer"]
    directions = ["up", "lateral", "down"]
    rows = [
        {"direction": rng.choice(directions), "decision": rng.choice(decisions)} for _ in range(300)
    ]
    df = pd.DataFrame(rows)
    result = direction_decision_association(df)
    assert result.p_value > 0.05


def test_association_rejects_empty_data() -> None:
    empty = pd.DataFrame({"direction": [], "decision": []})
    with pytest.raises(InsufficientDataError):
        direction_decision_association(empty)


def test_association_contingency_table_has_expected_shape() -> None:
    df = pd.DataFrame(
        {
            "direction": ["up", "up", "down", "lateral"],
            "decision": ["accept", "decline", "accept", "accept"],
        }
    )
    result = direction_decision_association(df)
    assert set(result.contingency_table.index) == {"up", "down", "lateral"}
    assert isinstance(result, AssociationResult)


# --------------------------------------------------------------- summary


def test_summarize_by_direction_reports_mean_and_count() -> None:
    df = _clustered_data({"lateral": 0.0, "up": 1.0}, n_personas=4, n_per_cell=5)
    summary = summarize_by_direction(df, ["outcome"])
    assert summary.loc["up", ("outcome", "count")] == 20
    assert summary.loc["up", ("outcome", "mean")] > summary.loc["lateral", ("outcome", "mean")]


def test_mixed_model_result_is_frozen() -> None:
    df = _clustered_data({"lateral": 0.0, "up": 0.2})
    result = fit_direction_mixed_model(df, "outcome")
    assert isinstance(result, MixedModelResult)
    with pytest.raises(AttributeError):
        result.n_observations = 999  # type: ignore[misc]


# --------------------------------------------------------- dose-response


def _dose_data(
    slope: float,
    *,
    n_items: int = 30,
    doses: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    seed: int = 5,
) -> pd.DataFrame:
    """Synthetic data with a known slope on a continuous dose, clustered by item."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_items):
        cell_id = f"item{index}"
        offset = rng.normal(0.0, 0.3)
        for dose in doses:
            rows.append(
                {
                    "cell_id": cell_id,
                    "dose": dose,
                    "outcome": 0.5 + slope * dose + offset + rng.normal(0.0, 0.2),
                }
            )
    return pd.DataFrame(rows)


def test_dose_response_recovers_a_known_slope() -> None:
    """A slope injected at 0.4 must come back near 0.4, with an interval that
    contains it -- the same standard every other model here is held to."""
    result = fit_dose_response_model(_dose_data(0.4), "outcome", "dose")

    assert isinstance(result, DoseResponseResult)
    assert result.converged
    assert result.slope == pytest.approx(0.4, abs=0.03)
    assert result.conf_low < 0.4 < result.conf_high
    assert result.p_value < 0.001
    assert result.n_observations == 30 * 4
    assert result.n_groups == 30


def test_dose_response_recovers_a_zero_slope() -> None:
    """A dose that does nothing must not produce an effect."""
    result = fit_dose_response_model(_dose_data(0.0), "outcome", "dose")
    assert result.slope == pytest.approx(0.0, abs=0.03)
    assert result.p_value > 0.05


def test_dose_response_recovers_a_negative_slope() -> None:
    result = fit_dose_response_model(_dose_data(-0.25), "outcome", "dose")
    assert result.slope == pytest.approx(-0.25, abs=0.03)


def test_dose_response_needs_two_dose_levels() -> None:
    """One dose level cannot identify a slope, and must fail loudly."""
    with pytest.raises(InsufficientDataError, match="distinct"):
        fit_dose_response_model(_dose_data(0.4, doses=(2.0,)), "outcome", "dose")


def test_dose_response_needs_enough_data() -> None:
    tiny = pd.DataFrame({"cell_id": ["a"], "dose": [1.0], "outcome": [0.5]})
    with pytest.raises(InsufficientDataError):
        fit_dose_response_model(tiny, "outcome", "dose")


def test_dose_response_result_is_frozen() -> None:
    result = fit_dose_response_model(_dose_data(0.4), "outcome", "dose")
    with pytest.raises(AttributeError):
        result.slope = 999.0  # type: ignore[misc]


# ------------------------------------------- covariates and fixed effects

# Which directions a sender of each rank writes in. Rank 1 cannot write down
# and rank 3 cannot write up, as in real email. So direction and rank are
# confounded: most "up" rows come from low-rank senders.
_RANK_PLAN: dict[int, list[str]] = {
    1: ["up", "up", "up", "lateral"],
    2: ["up", "lateral", "lateral", "down"],
    3: ["lateral", "down", "down", "down"],
}
_DIRECTION_EFFECT: dict[str, float] = {"up": 0.3, "lateral": 0.0, "down": -0.2}


def _rank_confounded_data(*, n_senders: int = 600, seed: int = 1) -> pd.DataFrame:
    """Per-email data where rank moves the outcome a lot (1.5 per step), and
    each sender has only four noisy emails. A random intercept alone then
    shrinks each sender toward the mean and leaves part of the rank
    difference in the direction contrasts."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_senders):
        rank = 1 + i % 3
        offset = rng.normal(0.0, 0.1)
        for direction in _RANK_PLAN[rank]:
            rows.append(
                {
                    "who": f"s{i}",
                    "rank": str(rank),
                    "direction": direction,
                    "outcome": 1.5 * (rank - 1)
                    + _DIRECTION_EFFECT[direction]
                    + offset
                    + rng.normal(0.0, 1.0),
                }
            )
    return pd.DataFrame(rows)


def test_rank_covariate_recovers_a_known_effect_under_confounding() -> None:
    """With rank as a covariate the injected +0.3 / -0.2 come back. Without
    it they do not, which shows the covariate is doing the work."""
    df = _rank_confounded_data()
    naive = fit_direction_mixed_model(df, "outcome", cluster_col="who")
    adjusted = fit_direction_mixed_model(df, "outcome", cluster_col="who", covariates=["rank"])

    assert adjusted.contrast("up")[0] == pytest.approx(0.3, abs=0.15)
    assert adjusted.contrast("up")[1] < 0.01
    assert adjusted.contrast("down")[0] == pytest.approx(-0.2, abs=0.15)
    assert naive.contrast("up")[0] < 0.1
    assert naive.contrast("down")[0] > 0.0


def test_covariate_terms_are_not_read_as_direction_contrasts() -> None:
    """A covariate's own ``rank[T.2]`` term also contains ``[T.``. It must
    keep its name and never overwrite a direction contrast."""
    df = _rank_confounded_data(n_senders=60)
    result = fit_direction_mixed_model(df, "outcome", cluster_col="who", covariates=["rank"])
    direction_keys = sorted(k for k in result.coefficients if k.startswith("direction["))
    assert direction_keys == ["direction[T.down]", "direction[T.up]"]
    assert "rank[T.2]" in result.coefficients
    assert "rank[T.3]" in result.coefficients


def test_no_covariates_gives_the_same_terms_as_before() -> None:
    df = _clustered_data({"lateral": 0.0, "up": 0.5, "down": 0.05})
    result = fit_direction_mixed_model(df, "outcome")
    assert set(result.coefficients) == {
        "Intercept",
        "Group Var",
        "direction[T.down]",
        "direction[T.up]",
    }


def test_fixed_effects_recover_a_known_effect_under_confounding() -> None:
    """Sender dummies absorb rank, so no covariate is needed."""
    result = fit_direction_fixed_effects(_rank_confounded_data(), "outcome", cluster_col="who")
    assert isinstance(result, FixedEffectsResult)
    assert result.contrast("up")[0] == pytest.approx(0.3, abs=0.15)
    assert result.contrast("up")[1] < 0.01
    assert result.contrast("down")[0] == pytest.approx(-0.2, abs=0.15)
    assert set(result.coefficients) == {"direction[T.down]", "direction[T.up]"}
    assert result.n_groups == 600


def _rank_confounded_sentences(*, n_senders: int = 90, seed: int = 11) -> pd.DataFrame:
    """One row per sentence, six sentences per email, twelve emails per
    sender. Logit effects: up +1.0, down -0.8, rank +0.8 per step. Each email
    has its own offset, so sentences inside one email are correlated."""
    effects = {"up": 1.0, "lateral": 0.0, "down": -0.8}
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_senders):
        rank = 1 + i % 3
        sender_offset = rng.normal(0.0, 0.3)
        for k, direction in enumerate(_RANK_PLAN[rank] * 3):
            email_offset = rng.normal(0.0, 0.5)
            logit = -0.5 + 0.8 * (rank - 1) + effects[direction] + sender_offset + email_offset
            prob = 1.0 / (1.0 + np.exp(-logit))
            for _ in range(6):
                rows.append(
                    {
                        "who": f"s{i}",
                        "email": f"s{i}e{k}",
                        "rank": str(rank),
                        "direction": direction,
                        "is_imperative": int(rng.random() < prob),
                    }
                )
    return pd.DataFrame(rows)


def test_sentence_model_with_covariate_and_nested_intercept_recovers_the_effect() -> None:
    df = _rank_confounded_sentences()
    result = fit_sentence_level_model(
        df, "is_imperative", cluster_col="who", covariates=["rank"], nested_col="email"
    )
    assert result.contrast("up")[0] == pytest.approx(1.0, abs=0.35)
    assert result.contrast("up")[1] < 0.01
    assert result.contrast("down")[0] == pytest.approx(-0.8, abs=0.35)
    assert result.nested_sd is not None and result.nested_sd > 0
    assert "rank[T.3]" in result.coefficients


def test_sentence_model_has_no_nested_sd_by_default() -> None:
    result = fit_sentence_level_model(_sentence_data({"lateral": -0.3, "up": 1.5}), "is_imperative")
    assert result.nested_sd is None


def test_logistic_fixed_effects_recover_the_effect_and_drop_constant_senders() -> None:
    """A sender whose sentences are never imperative has an infinite dummy.
    It must be dropped and counted, not break the fit."""
    df = _rank_confounded_sentences()
    constant = df[df["who"] == "s0"].assign(who="never", is_imperative=0)
    result = fit_direction_fixed_effects(
        pd.concat([df, constant]), "is_imperative", cluster_col="who", family="logistic"
    )
    assert result.contrast("up")[0] == pytest.approx(1.0, abs=0.4)
    assert result.contrast("up")[1] < 0.01
    assert result.contrast("down")[0] == pytest.approx(-0.8, abs=0.4)
    assert result.n_groups_dropped == 1
    assert result.n_groups == 90


def test_fixed_effects_contrast_raises_a_clear_error_for_an_unobserved_level() -> None:
    result = fit_direction_fixed_effects(
        _rank_confounded_data(n_senders=60), "outcome", cluster_col="who"
    )
    with pytest.raises(KeyError, match="no contrast"):
        result.contrast("sideways")


def test_fixed_effects_rejects_too_few_groups() -> None:
    df = _rank_confounded_data(n_senders=60)
    with pytest.raises(InsufficientDataError):
        fit_direction_fixed_effects(df[df["who"] == "s0"], "outcome", cluster_col="who")


# ------------------------------------------------------ optimizer choice


class _FakeFit:
    def __init__(self, method: str, converged: bool, llf: float) -> None:
        self.method = method
        self.converged = converged
        self.llf = llf


class _FakeModel:
    """Stands in for MixedLM. Each optimizer either returns a fit with the
    given convergence flag and log-likelihood, or raises."""

    def __init__(self, outcomes: dict[str, tuple[bool, float] | Exception]) -> None:
        self.outcomes = outcomes

    def fit(self, *, reml: bool, method: str) -> _FakeFit:
        outcome = self.outcomes[method]
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeFit(method, *outcome)


def test_fit_keeps_the_converged_optimizer_with_the_highest_likelihood() -> None:
    """L-BFGS can report convergence at a worse point. The better fit from
    another optimizer must win, as on the real-email Q1 data."""
    model = _FakeModel({"lbfgs": (True, -10.0), "powell": (True, -5.0), "nm": (True, -5.0)})
    assert _fit_with_fallback(model, label="x").method == "powell"


def test_fit_rejects_an_infinite_likelihood_that_claims_convergence() -> None:
    """The real-email failure: L-BFGS said converged, with llf = inf."""
    model = _FakeModel(
        {"lbfgs": (True, float("inf")), "powell": (True, 297.8), "nm": (True, 297.8)}
    )
    assert _fit_with_fallback(model, label="x").method == "powell"


def test_fit_ignores_a_better_likelihood_that_did_not_converge() -> None:
    model = _FakeModel({"lbfgs": (True, -10.0), "powell": (False, 0.0), "nm": (True, -12.0)})
    assert _fit_with_fallback(model, label="x").method == "lbfgs"


def test_fit_skips_an_optimizer_that_raises() -> None:
    model = _FakeModel(
        {"lbfgs": LinAlgError("singular"), "powell": (True, -3.0), "nm": (True, -4.0)}
    )
    assert _fit_with_fallback(model, label="x").method == "powell"


def test_fit_returns_the_last_fit_when_none_converges() -> None:
    model = _FakeModel({"lbfgs": (False, -1.0), "powell": (False, -2.0), "nm": (False, -3.0)})
    assert _fit_with_fallback(model, label="x").method == "nm"


def test_fit_raises_when_every_optimizer_raises() -> None:
    model = _FakeModel({m: ValueError("bad") for m in ("lbfgs", "powell", "nm")})
    with pytest.raises(InsufficientDataError, match="no optimizer"):
        _fit_with_fallback(model, label="x")


# --------------------------------------------------------- replicate handling


def test_cell_id_without_replicate_strips_the_draw_suffix() -> None:
    ids = pd.Series(["sim_q1__p0__scen__r1", "sim_q1__p0__scen__r2"])
    assert cell_id_without_replicate(ids).tolist() == ["sim_q1__p0__scen"] * 2


def test_cell_id_without_replicate_only_strips_the_trailing_suffix() -> None:
    """A persona or scenario id that happens to contain ``r1`` elsewhere in
    the string must survive -- only the trailing draw index goes."""
    ids = pd.Series(["sim_q1__rank1_p__scen__r1"])
    assert cell_id_without_replicate(ids).tolist() == ["sim_q1__rank1_p__scen"]


def test_aggregate_replicates_averages_the_value_columns() -> None:
    df = pd.DataFrame(
        {
            "cell_id": ["c__r1", "c__r2"],
            "persona_id": ["p0", "p0"],
            "direction": ["up", "up"],
            "imperative_ratio": [0.2, 0.6],
        }
    )
    result = aggregate_replicates(df, ["imperative_ratio"], keep_cols=["persona_id", "direction"])
    assert len(result) == 1
    assert result["imperative_ratio"].iloc[0] == pytest.approx(0.4)
    assert result["persona_id"].iloc[0] == "p0"
    assert result["n_draws"].iloc[0] == 2


def test_aggregate_replicates_keeps_cells_separate() -> None:
    df = pd.DataFrame(
        {
            "cell_id": ["a__r1", "a__r2", "b__r1", "b__r2"],
            "persona_id": ["p0", "p0", "p0", "p0"],
            "imperative_ratio": [0.0, 0.0, 1.0, 1.0],
        }
    )
    result = aggregate_replicates(df, ["imperative_ratio"], keep_cols=["persona_id"])
    assert len(result) == 2
    assert set(result["imperative_ratio"]) == {0.0, 1.0}


def _persona_cell_sentences(
    effects: dict[str, float],
    *,
    n_personas: int = 16,
    cells_per_persona: int = 12,
    sentences_per_draw: int = 3,
    n_draws: int = 2,
    seed: int = 5,
) -> pd.DataFrame:
    """Sentences nested in cells nested in personas -- the shape
    ``run_q1_analysis`` builds for a multi-draw Q1 grid: each cell is
    answered ``n_draws`` times, and each reply contributes
    ``sentences_per_draw`` sentences that are not independent of each
    other."""
    rng = np.random.default_rng(seed)
    directions = list(effects)
    rows = []
    for p in range(n_personas):
        persona_offset = rng.normal(0.0, 0.3)
        for c in range(cells_per_persona):
            direction = directions[c % len(directions)]
            cell_offset = rng.normal(0.0, 0.4)
            logit = -0.3 + effects[direction] + persona_offset + cell_offset
            prob = 1.0 / (1.0 + np.exp(-logit))
            for _draw in range(1, n_draws + 1):
                for _ in range(sentences_per_draw):
                    rows.append(
                        {
                            "persona_id": f"p{p}",
                            "cell_base_id": f"p{p}_c{c}",
                            "direction": direction,
                            "is_imperative": int(rng.random() < prob),
                        }
                    )
    return pd.DataFrame(rows)


def test_sentence_model_recovers_effect_with_cells_nested_in_persona() -> None:
    """The two-grouping-level shape a multi-draw Q1 grid needs: a random
    intercept per persona and a second one per cell, since two draws of one
    cell give two replies whose sentences are not independent of each
    other. A known effect must still come back close to its true value."""
    df = _persona_cell_sentences({"lateral": 0.0, "up": 1.2, "down": -0.9})
    result = fit_sentence_level_model(df, "is_imperative", nested_col="cell_base_id")

    assert result.contrast("up")[0] == pytest.approx(1.2, abs=0.35)
    assert result.contrast("up")[1] < 0.01
    assert result.contrast("down")[0] == pytest.approx(-0.9, abs=0.35)
    assert result.contrast("down")[1] < 0.01
    assert result.nested_sd is not None and result.nested_sd > 0
