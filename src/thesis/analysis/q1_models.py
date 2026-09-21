"""Q1 across models: does each model show the writing-down effect that real email shows?

Reads any number of saved Q1 grids, one per model, and puts their direction
contrasts next to real email (PROGRESS.md section 48). No model is called.

Each grid is refit with :func:`thesis.analysis.q1.run_q1_analysis`, so a model
here is measured exactly as the single-model report measures it. The output is
one manifest and one figure, named by the caller so a later run with more
models never overwrites the numbers an earlier section cited.

    python -m thesis.analysis.q1_models \\
        --grid "DeepSeek V4 Flash=data/interim/q1_direction_grid_deepseek.parquet" \\
        --grid "gpt-oss-20b=data/interim/q1_direction_grid_gpt_oss_20b.parquet" \\
        --figure-prefix q1_models_ --manifest outputs/manifests/q1_models.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from thesis.analysis.plots import plot_effect_intervals
from thesis.analysis.q1 import (
    REAL_MANIFEST_PATH,
    Q1Grid,
    Q1Result,
    compare_with_real,
    design_of_frame,
    grid_contrasts,
    run_q1_analysis,
    sentence_levels_by_direction,
)
from thesis.analysis.q1_real import implied_se
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import DOCS_FIGURES_DIR

log = get_logger(__name__)

# 95% two-sided normal interval.
Z_95 = 1.96
PRIMARY = "is_imperative:down"


def parse_grid_arg(text: str) -> tuple[str, Path]:
    """Split a ``LABEL=PATH`` argument."""
    label, sep, path = text.partition("=")
    if not sep or not label or not path:
        msg = f"expected LABEL=PATH, got {text!r}"
        raise ValueError(msg)
    return label, Path(path)


def load_grid(path: Path) -> Q1Grid:
    """A saved grid file as a :class:`Q1Grid`, with nothing generated."""
    frame = pd.read_parquet(path)
    return Q1Grid(
        frame=frame,
        run_id="from-file",
        model=str(frame["model"].iloc[0]),
        n_cells=len(frame),
        n_from_cache=len(frame),
        n_generated=0,
        design=design_of_frame(frame),
    )


def summarize_model(result: Q1Result, real_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """One model's row: size, levels, contrasts and the comparison with real email."""
    contrasts = grid_contrasts(result)
    coefficient, p_value = contrasts[PRIMARY]
    se = implied_se(coefficient, p_value)
    n_replies = len(result.reply_features)
    n_sentences = len(result.sentence_features)
    return {
        "model": result.grid.model,
        "n_replies": n_replies,
        "n_sentences": n_sentences,
        "sentences_per_reply": round(n_sentences / n_replies, 2),
        "levels_by_direction": sentence_levels_by_direction(result.sentence_features),
        "primary": {
            "contrast": PRIMARY,
            "coefficient": round(coefficient, 4),
            "p": round(p_value, 4),
            "se": round(se, 4),
            "ci_low": round(coefficient - Z_95 * se, 4),
            "ci_high": round(coefficient + Z_95 * se, 4),
        },
        "vs_real": compare_with_real(contrasts, real_manifest),
    }


def real_row(real_manifest: Mapping[str, Any]) -> dict[str, float]:
    """The real-email benchmark for the primary contrast, with its interval."""
    real = real_manifest["simulator_vs_real"][PRIMARY]
    coefficient, p_value = float(real["real"]), float(real["real_p"])
    se = implied_se(coefficient, p_value)
    return {
        "coefficient": round(coefficient, 4),
        "p": p_value,
        "se": round(se, 4),
        "ci_low": round(coefficient - Z_95 * se, 4),
        "ci_high": round(coefficient + Z_95 * se, 4),
    }


def plot_models_vs_real(
    real: Mapping[str, float], models: Mapping[str, Mapping[str, Any]], path: Path
) -> Path:
    """The writing-down effect per model, each with its 95% interval, under real email."""
    labels = ["real Enron email", *models]
    rows = [real, *(m["primary"] for m in models.values())]
    return plot_effect_intervals(
        labels,
        [float(r["coefficient"]) for r in rows],
        [float(r["ci_low"]) for r in rows],
        [float(r["ci_high"]) for r in rows],
        path,
        title="Writing down: how much more often a sentence gives an order",
        subtitle="Difference from writing to a peer, on the logit scale, with a 95% interval.",
        x_label="effect of writing down (logit scale)",
    )


def build_manifest(
    grids: Sequence[tuple[str, Path]], real_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Fit every grid and collect the numbers one section needs."""
    models: dict[str, dict[str, Any]] = {}
    for label, path in grids:
        log.info("fitting %s from %s", label, path)
        models[label] = summarize_model(run_q1_analysis(load_grid(path)), real_manifest)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "real": real_row(real_manifest),
        "real_levels_by_direction": real_manifest["real_levels_by_direction"],
        "models": models,
    }


def format_table(manifest: Mapping[str, Any]) -> str:
    """Plain-text table of the primary contrast for every model."""
    real = manifest["real"]
    lines = [
        f"{'':22s}{'replies':>9s}{'sentences':>11s}{'writing down':>14s}{'p':>8s}"
        f"{'95% interval':>18s}{'vs real p':>11s}",
        f"{'real email':22s}{'':>9s}{'':>11s}{real['coefficient']:>+14.3f}{real['p']:>8.3f}"
        f"{f'[{real["ci_low"]:+.2f}, {real["ci_high"]:+.2f}]':>18s}{'':>11s}",
    ]
    for label, m in manifest["models"].items():
        p = m["primary"]
        lines.append(
            f"{label:22s}{m['n_replies']:>9d}{m['n_sentences']:>11d}{p['coefficient']:>+14.3f}"
            f"{p['p']:>8.3f}{f'[{p["ci_low"]:+.2f}, {p["ci_high"]:+.2f}]':>18s}"
            f"{m['vs_real'][PRIMARY]['difference_p']:>11.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="A saved Q1 grid and the name to show for it. Repeat once per model.",
    )
    parser.add_argument(
        "--figure-prefix",
        required=True,
        help="Prefix for the figure file, written under docs/figures/.",
    )
    parser.add_argument("--manifest", required=True, help="Where to write the manifest JSON.")
    args = parser.parse_args()

    configure_logging()
    if not REAL_MANIFEST_PATH.exists():
        msg = f"no real-email benchmark at {REAL_MANIFEST_PATH}; run thesis.analysis.q1_real first"
        raise FileNotFoundError(msg)
    real_manifest = json.loads(REAL_MANIFEST_PATH.read_text(encoding="utf-8"))

    manifest = build_manifest([parse_grid_arg(g) for g in args.grid], real_manifest)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    figure = plot_models_vs_real(
        manifest["real"],
        manifest["models"],
        DOCS_FIGURES_DIR / f"{args.figure_prefix}writing_down.png",
    )
    print(format_table(manifest))
    log.info("wrote %s and %s", manifest_path, figure)


if __name__ == "__main__":
    main()
