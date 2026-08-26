"""Tests for the PROGRESS.md judge figures.

Rendering is not asserted pixel-by-pixel -- that would test matplotlib, and
would break on every legitimate styling change. What is asserted is the
contract around it: a file actually appears, and the argument shapes that
would silently mislabel a chart are rejected loudly instead. A figure that
pairs the wrong score with the wrong dimension is worse than no figure,
because it looks authoritative.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis.analysis.plots import (
    plot_discrimination_auc,
    plot_factor_interaction,
    plot_paired_dimensions,
)


def test_paired_dimensions_writes_a_png(tmp_path: Path) -> None:
    out = plot_paired_dimensions(
        ("clarity", "role_consistency"),
        real_scores=(4.7, 4.58),
        generated_scores=(4.72, 4.62),
        path=tmp_path / "paired.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_paired_dimensions_creates_missing_parent_directories(tmp_path: Path) -> None:
    out = plot_paired_dimensions(
        ("clarity",),
        real_scores=(4.7,),
        generated_scores=(4.6,),
        path=tmp_path / "nested" / "deeper" / "paired.png",
    )
    assert out.exists()


def test_paired_dimensions_rejects_mismatched_lengths(tmp_path: Path) -> None:
    """The failure mode this guards against is silent mislabelling: a short
    score list zipped against a longer dimension list would draw real
    numbers under the wrong names."""
    with pytest.raises(ValueError, match="same length"):
        plot_paired_dimensions(
            ("clarity", "role_consistency"),
            real_scores=(4.7,),
            generated_scores=(4.72, 4.62),
            path=tmp_path / "bad.png",
        )


def test_interaction_plot_writes_a_png(tmp_path: Path) -> None:
    out = plot_factor_interaction(
        ("llama-generated", "qwen-generated"),
        {"llama judging": (3.7, 4.3), "qwen judging": (2.7, 3.7)},
        path=tmp_path / "interaction.png",
        title="t",
    )
    assert out.exists()


def test_interaction_plot_rejects_more_than_two_series(tmp_path: Path) -> None:
    """Two categorical hues are what the palette validates for adjacent
    pairs here; a third silently introduced would not be checked."""
    with pytest.raises(ValueError, match="1 or 2 series"):
        plot_factor_interaction(
            ("a", "b"),
            {"one": (1.0, 2.0), "two": (2.0, 3.0), "three": (3.0, 4.0)},
            path=tmp_path / "bad.png",
            title="t",
        )


def test_interaction_plot_rejects_a_series_of_the_wrong_length(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="values but there are"):
        plot_factor_interaction(
            ("a", "b", "c"),
            {"one": (1.0, 2.0)},
            path=tmp_path / "bad.png",
            title="t",
        )


def test_discrimination_auc_writes_a_png(tmp_path: Path) -> None:
    out = plot_discrimination_auc(
        ("format-matched", "body only"), (0.719, 0.841), path=tmp_path / "auc.png"
    )
    assert out.exists()


def test_discrimination_auc_rejects_values_below_chance(tmp_path: Path) -> None:
    """The bars are anchored at 0.5, so a sub-chance value would render as a
    negative-width bar pointing the wrong way rather than as a visible
    error."""
    with pytest.raises(ValueError, match="below chance"):
        plot_discrimination_auc(("bad",), (0.42,), path=tmp_path / "bad.png")


def test_discrimination_auc_rejects_mismatched_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_discrimination_auc(("a", "b"), (0.7,), path=tmp_path / "bad.png")
