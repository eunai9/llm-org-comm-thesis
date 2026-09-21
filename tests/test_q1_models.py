"""Tests for the Q1 model comparison: argument parsing, the real-email row,
the interval arithmetic, the table, and the figure. No model is called and no
grid is refit; the fit itself is covered by tests/test_q1.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from thesis.analysis.plots import plot_effect_intervals
from thesis.analysis.q1_models import (
    PRIMARY,
    Z_95,
    format_table,
    parse_grid_arg,
    plot_models_vs_real,
    real_row,
)

REAL_MANIFEST = {"simulator_vs_real": {PRIMARY: {"real": 0.253, "real_p": 0.0004}}}


def test_parse_grid_arg_splits_label_and_path() -> None:
    label, path = parse_grid_arg("gpt-oss-20b=data/interim/grid.parquet")
    assert label == "gpt-oss-20b"
    assert path == Path("data/interim/grid.parquet")


def test_parse_grid_arg_keeps_an_equals_sign_in_the_path() -> None:
    label, path = parse_grid_arg("a=b=c")
    assert (label, path) == ("a", Path("b=c"))


@pytest.mark.parametrize("bad", ["no-separator", "=path-only", "label-only="])
def test_parse_grid_arg_rejects_a_missing_part(bad: str) -> None:
    with pytest.raises(ValueError, match="LABEL=PATH"):
        parse_grid_arg(bad)


def test_real_row_interval_is_centered_on_the_coefficient() -> None:
    row = real_row(REAL_MANIFEST)
    assert row["coefficient"] == pytest.approx(0.253)
    assert row["ci_high"] - row["coefficient"] == pytest.approx(Z_95 * row["se"], abs=1e-3)
    assert row["coefficient"] - row["ci_low"] == pytest.approx(Z_95 * row["se"], abs=1e-3)


def _model(coefficient: float, p: float) -> dict[str, object]:
    return {
        "n_replies": 238,
        "n_sentences": 719,
        "primary": {
            "coefficient": coefficient,
            "p": p,
            "ci_low": coefficient - 0.25,
            "ci_high": coefficient + 0.25,
        },
        "vs_real": {PRIMARY: {"difference_p": 0.4}},
    }


def test_format_table_has_one_row_per_model_plus_real() -> None:
    manifest = {
        "real": real_row(REAL_MANIFEST),
        "models": {"model a": _model(0.14, 0.296), "model b": _model(0.44, 0.0)},
    }
    lines = format_table(manifest).splitlines()
    assert len(lines) == 4
    assert lines[1].startswith("real email")
    assert lines[2].startswith("model a")
    assert "+0.140" in lines[2]


def test_plot_effect_intervals_writes_a_file(tmp_path: Path) -> None:
    out = plot_effect_intervals(
        ["real", "model"], [0.25, 0.1], [0.15, -0.1], [0.35, 0.3], tmp_path / "f.png", title="t"
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_effect_intervals_rejects_unequal_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_effect_intervals(["a", "b"], [0.1], [0.0], [0.2], tmp_path / "f.png", title="t")


def test_plot_models_vs_real_puts_real_email_first(tmp_path: Path) -> None:
    real = real_row(REAL_MANIFEST)
    out = plot_models_vs_real(real, {"model a": _model(0.14, 0.296)}, tmp_path / "m.png")
    assert out.exists()
