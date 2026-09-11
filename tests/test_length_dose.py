"""Tests for the length dose-response experiment.

Only the deterministic parts are tested. The model calls are not: they need a
running Ollama server and they are the part this design treats as the
measurement, not as code under test.

The elasticity test injects a known slope and checks it comes back, the same
practice every other mixed model in this project is tested under.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from thesis.analysis.hierarchy import InsufficientDataError
from thesis.analysis.length_dose import (
    TARGET_HIT_FRACTION,
    TARGET_LEVELS,
    check_pairing,
    fit_elasticity,
    format_report,
    override_target_length,
    paired_frame,
    run_analysis,
    spread_test,
    summarize_conditions,
    summary_dict,
    word_count,
)
from thesis.sim.persona import Persona, PersonaStyle, render_persona_block


def _persona(persona_id: str, mean_tokens: float) -> Persona:
    return Persona(
        persona_id=persona_id,
        seniority_rank=3,
        rank_label="Director",
        department="Trading",
        style=PersonaStyle(
            mean_tokens=mean_tokens,
            mean_recipients=2.0,
            imperative_ratio=0.3,
            hedge_rate=0.1,
            deference_rate=0.05,
            question_ratio=0.2,
        ),
        n_people=7,
        n_messages=500,
        derivation="cell",
    )


def _dose_frame(
    slope: float,
    *,
    n_items: int = 40,
    targets: tuple[int, ...] = TARGET_LEVELS,
    seed: int = 7,
    item_sd: float = 0.25,
    noise_sd: float = 0.2,
) -> pd.DataFrame:
    """Replies whose length follows a known elasticity on the instructed target."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_items):
        cell_id = f"cell{index}"
        offset = rng.normal(0.0, item_sd)
        for target in targets:
            log_words = 1.0 + slope * math.log(target) + offset + rng.normal(0.0, noise_sd)
            rows.append(
                {
                    "cell_id": cell_id,
                    "instructed_target": target,
                    "reply_words": max(1, round(math.exp(log_words))),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------ the design


def test_the_grid_spans_a_20x_range_and_includes_the_models_own_length() -> None:
    """20 is in the grid so the instruction can be tested in both directions."""
    assert TARGET_LEVELS == (20, 50, 100, 200, 400)
    assert list(TARGET_LEVELS) == sorted(TARGET_LEVELS)
    assert len(set(TARGET_LEVELS)) == len(TARGET_LEVELS)
    assert max(TARGET_LEVELS) / min(TARGET_LEVELS) == 20


# --------------------------------------------------------- the override


def test_override_changes_only_the_stated_length() -> None:
    base = [_persona("p1", 62.0), _persona("p2", 93.0)]
    overridden = override_target_length(base, 200)

    assert [p.style.mean_tokens for p in overridden] == [200.0, 200.0]
    for before, after in zip(base, overridden, strict=True):
        assert after.persona_id == before.persona_id
        assert after.seniority_rank == before.seniority_rank
        assert after.department == before.department
        assert after.style.imperative_ratio == before.style.imperative_ratio
        assert after.style.hedge_rate == before.style.hedge_rate
        assert after.style.mean_recipients == before.style.mean_recipients


def test_override_leaves_the_originals_alone() -> None:
    """The personas are frozen; the override must return copies, not mutate."""
    base = [_persona("p1", 62.0)]
    override_target_length(base, 400)
    assert base[0].style.mean_tokens == 62.0


def test_override_reaches_the_prompt_text() -> None:
    """The whole experiment rests on this one rendered line changing."""
    overridden = override_target_length([_persona("p1", 62.0)], 400)
    assert "Typical message length: about 400 words." in render_persona_block(overridden[0])


def test_every_condition_states_the_same_target() -> None:
    """Within a condition the dose must be one number, not ten persona numbers."""
    base = [_persona("p1", 62.0), _persona("p2", 93.0), _persona("p3", 70.0)]
    for target in TARGET_LEVELS:
        assert {p.style.mean_tokens for p in override_target_length(base, target)} == {
            float(target)
        }


# ----------------------------------------------------------- the pairing


def test_pairing_holds_when_every_item_appears_in_every_condition() -> None:
    report = check_pairing(_dose_frame(0.1), TARGET_LEVELS)
    assert report.n_items_in_every_condition == 40
    assert report.n_items_dropped == 0
    assert report.fully_paired


def test_pairing_flags_an_item_missing_from_one_condition() -> None:
    frame = _dose_frame(0.1)
    dropped = frame[~((frame["cell_id"] == "cell0") & (frame["instructed_target"] == 400))]
    report = check_pairing(dropped, TARGET_LEVELS)

    assert report.n_items_any == 40
    assert report.n_items_in_every_condition == 39
    assert report.n_items_dropped == 1
    assert not report.fully_paired


def test_paired_frame_keeps_only_complete_items() -> None:
    frame = _dose_frame(0.1)
    dropped = frame[~((frame["cell_id"] == "cell0") & (frame["instructed_target"] == 400))]
    kept = paired_frame(dropped, TARGET_LEVELS)

    assert "cell0" not in set(kept["cell_id"])
    assert len(kept) == 39 * len(TARGET_LEVELS)


# -------------------------------------------------------- the elasticity


def test_elasticity_recovers_a_known_slope() -> None:
    """The headline number. A slope injected at 0.45 must come back near 0.45."""
    result = fit_elasticity(_dose_frame(0.45))

    assert result.converged
    assert result.slope == pytest.approx(0.45, abs=0.05)
    assert result.p_value < 0.001
    assert result.conf_low < 0.45 < result.conf_high
    assert result.n_observations == 40 * len(TARGET_LEVELS)
    assert result.n_groups == 40


def test_elasticity_recovers_a_flat_response() -> None:
    """A model that ignores the instruction gives a slope near zero, and the
    interval must contain zero rather than the fit inventing an effect."""
    result = fit_elasticity(_dose_frame(0.0))

    assert result.slope == pytest.approx(0.0, abs=0.05)
    assert result.conf_low < 0.0 < result.conf_high
    assert result.p_value > 0.05


def test_elasticity_needs_more_than_one_dose() -> None:
    frame = _dose_frame(0.3, targets=(100,))
    with pytest.raises(InsufficientDataError, match="distinct"):
        fit_elasticity(frame)


def test_elasticity_drops_empty_replies_rather_than_taking_log_of_zero() -> None:
    frame = _dose_frame(0.3)
    frame.loc[0, "reply_words"] = 0
    result = fit_elasticity(frame)

    assert result.n_observations == 40 * len(TARGET_LEVELS) - 1


# ----------------------------------------------------------- the summary


def test_word_count_is_whitespace_separated() -> None:
    assert word_count("send the file to legal") == 5
    assert word_count("") == 0


def test_share_reaching_target_uses_the_eighty_percent_bar() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c", "d"],
            "instructed_target": [100, 100, 100, 100],
            # 80 is exactly the bar, 79 is just under it.
            "reply_words": [120, 80, 79, 10],
        }
    )
    row = summarize_conditions(frame).iloc[0]

    assert TARGET_HIT_FRACTION == 0.80
    assert row["share_reaching_target"] == pytest.approx(0.5)
    assert row["n_replies"] == 4
    assert row["median_words"] == pytest.approx(79.5)


def test_summary_has_one_row_per_condition_in_order() -> None:
    table = summarize_conditions(_dose_frame(0.2))
    assert list(table["instructed_target"]) == list(TARGET_LEVELS)
    assert (table["n_replies"] == 40).all()


def test_spread_test_sees_a_widening_spread() -> None:
    """A flat mean with a growing spread is a real result, so it gets a test."""
    rng = np.random.default_rng(3)
    rows = []
    for target, sd in ((20, 2.0), (400, 40.0)):
        for index in range(60):
            rows.append(
                {
                    "cell_id": f"cell{index}",
                    "instructed_target": target,
                    "reply_words": max(1, round(50 + rng.normal(0.0, sd))),
                }
            )
    statistic, p_value = spread_test(pd.DataFrame(rows))

    assert statistic > 0
    assert p_value < 0.01


def test_spread_test_needs_two_conditions() -> None:
    with pytest.raises(InsufficientDataError, match="2 conditions"):
        spread_test(_dose_frame(0.1, targets=(100,)))


# ------------------------------------------------------- the whole report


def test_run_analysis_reports_the_slope_and_the_table() -> None:
    result = run_analysis(_dose_frame(0.45), model="llama3.2:3b")

    assert result.pairing.fully_paired
    assert len(result.per_condition) == len(TARGET_LEVELS)
    assert result.elasticity.slope == pytest.approx(0.45, abs=0.05)

    summary = summary_dict(result)
    assert summary["model"] == "llama3.2:3b"
    assert summary["targets"] == list(TARGET_LEVELS)
    assert summary["fully_paired"] is True

    report = format_report(result)
    assert "Elasticity" in report
    assert "instructed_target" in report
