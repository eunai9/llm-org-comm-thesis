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

from thesis.analysis.hierarchy import (
    AssociationResult,
    InsufficientDataError,
    MixedModelResult,
    direction_decision_association,
    fit_direction_mixed_model,
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
