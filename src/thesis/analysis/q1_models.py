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

from thesis.analysis.hierarchy import (
    FixedEffectsResult,
    SentenceModelResult,
    fit_direction_fixed_effects,
)
from thesis.analysis.plots import plot_effect_intervals
from thesis.analysis.q1 import (
    REAL_MANIFEST_PATH,
    Q1Grid,
    Q1Result,
    compare_with_real,
    design_of_frame,
    grid_contrasts,
    prompt_hash_of_frame,
    run_q1_analysis,
    sentence_levels_by_direction,
)
from thesis.analysis.q1_real import (
    CONTRASTS,
    REFERENCE,
    SENDER_COL,
    SENTENCE_OUTCOME,
    SENTENCES_PATH,
    fit_sentence_model,
    implied_se,
    robust_contrasts,
    truncate_sentences,
)
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
    """A saved grid file as a :class:`Q1Grid`.

    ``n_from_cache``/``n_generated`` come from the ``from_cache`` column
    every grid row already carries (:func:`thesis.sim.run._result_row`),
    not a stand-in -- a grid loaded from a file used to hard-code
    ``n_from_cache=len(frame), n_generated=0`` regardless of the run's real
    split, which is wrong for any grid that generated anything this run.
    A grid saved before that column existed reports both as unknown (0/0)
    rather than guessing.
    """
    frame = pd.read_parquet(path)
    if "from_cache" in frame.columns:
        n_from_cache = int(frame["from_cache"].sum())
        n_generated = len(frame) - n_from_cache
    else:
        n_from_cache = 0
        n_generated = 0
    return Q1Grid(
        frame=frame,
        run_id="from-file",
        model=str(frame["model"].iloc[0]),
        n_cells=len(frame),
        n_from_cache=n_from_cache,
        n_generated=n_generated,
        prompt_text_hash=prompt_hash_of_frame(frame),
        design=design_of_frame(frame),
    )


def summarize_model(
    result: Q1Result,
    real_manifest: Mapping[str, Any],
    *,
    real_sentences: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """One model's row: size, levels, contrasts and the comparison with real email.

    ``real_sentences`` is optional: when given (real email's per-sentence
    table, ``q1_real.SENTENCES_PATH``), the row also gets
    ``vs_real_length_matched``, the same comparison cut to this model's own
    reply length (see :func:`compare_with_length_matched_real`) -- a second,
    independent check next to ``vs_real``'s full-length one, not a
    replacement for it.

    ``PRIMARY`` is a sentence-level ``is_imperative`` contrast, so
    ``result.sentence_model``'s variational-Bayes (VB) fit is not the only
    one available for it -- the persona-clustered fixed-effects fit
    (``result.sentence_model_persona_fe``) is the honest cross-check. The VB
    fit's p-value comes from a posterior SD that understates uncertainty on
    a large effect (found checking the gpt-oss-120b result,
    PROGRESS_llms.md's Q1 section), so the reported
    ``coefficient``/``p``/``se``/interval come from the clustered fit, not
    the VB one. The VB numbers stay in ``coefficient_vb``/``p_vb`` for
    comparison, not dropped.

    For a multi-draw grid, the pooled fixed-effects fit
    (``sentence_model_persona_fe_pooled``) is used instead of the draw-1-only
    one, matching ``grid_contrasts``'s own preference for the pooled fit --
    ``coefficient_vb``/``p_vb`` (from ``contrasts``, via ``grid_contrasts``)
    already come from the pooled fit for such a grid, so using the
    draw-1-only persona-FE fit here mixed a draw-1 number with pooled ones in
    the same row (found checking the Oct 4 full-design comparison against
    the Sep 23 PROGRESS_llms.md entry, where the two disagreed).

    ``vs_real`` has the same VB problem one level up: it used to compare the
    VB sentence model's ``is_imperative`` contrast against real email's own
    VB fit (``q1_real.py``'s ``"strict"`` version), the same understated
    pair ``primary`` above stops using. ``vs_real`` now uses the clustered
    fit on both sides for ``is_imperative`` instead -- this row's own
    ``persona_fe`` on the model side, and real email's
    ``strict_sender_fixed_effects`` fit (``q1_real.robust_contrasts``) on
    the real side. The other two outcomes (``imperative_ratio``,
    ``hedge_rate``) never had this problem -- they are plain mixed models,
    not VB -- so they pass through unchanged. The old VB-vs-VB comparison
    stays available under ``vs_real_vb``, not dropped.
    """
    contrasts = grid_contrasts(result)
    vb_coefficient, p_vb = contrasts[PRIMARY]
    level = PRIMARY.split(":")[1]
    persona_fe = result.sentence_model_persona_fe_pooled or result.sentence_model_persona_fe
    coefficient, p_value = persona_fe.contrast(level)
    se = persona_fe.std_errors[f"direction[T.{level}]"]
    n_replies = len(result.reply_features)
    n_sentences = len(result.sentence_features)
    row: dict[str, Any] = {
        "model": result.grid.model,
        "prompt_text_hash": result.grid.prompt_text_hash,
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
            "coefficient_vb": round(vb_coefficient, 4),
            "p_vb": round(p_vb, 4),
        },
        "vs_real": compare_with_real(
            _robust_is_imperative(contrasts, persona_fe), _robust_real_manifest(real_manifest)
        ),
        "vs_real_vb": compare_with_real(contrasts, real_manifest),
    }
    if real_sentences is not None:
        row["vs_real_length_matched"] = compare_with_length_matched_real(result, real_sentences)
    return row


def _robust_is_imperative(
    contrasts: Mapping[str, tuple[float, float]], persona_fe: FixedEffectsResult
) -> dict[str, tuple[float, float]]:
    """``contrasts`` with its ``is_imperative`` entries replaced by the
    persona-clustered fixed-effects fit's own contrasts, the same fit
    ``primary`` already reports -- everything else (the two outcomes that
    were never VB-fitted) passes through unchanged."""
    return {
        **contrasts,
        **{f"is_imperative:{level}": persona_fe.contrast(level) for level in CONTRASTS},
    }


def _robust_real_manifest(real_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """``real_manifest`` with its ``is_imperative`` entries in
    ``simulator_vs_real`` replaced by the sender-clustered fixed-effects
    fit (:func:`thesis.analysis.q1_real.robust_contrasts`), everything else
    unchanged."""
    robust = robust_contrasts(real_manifest)
    return {
        "simulator_vs_real": {
            **real_manifest["simulator_vs_real"],
            **{
                key: {"real": coefficient, "real_p": p_value}
                for key, (coefficient, p_value) in robust.items()
                if key.startswith("is_imperative:")
            },
        }
    }


def compare_with_length_matched_real(
    result: Q1Result, real_sentences: pd.DataFrame
) -> dict[str, Any]:
    """The ``is_imperative`` contrast against real email cut to this model's
    own mean reply length, instead of real email's full length.

    A model with short replies gets an inflated ``is_imperative`` share for
    the same underlying behavior -- one order out of two sentences is 50%,
    the same order out of five is 20% -- so comparing it against real
    email's full-length replies confounds direction with reply length. This
    cuts real email down to this model's own mean sentences per reply,
    rounded to the nearest whole sentence (minimum 1), before fitting the
    same real-email model ``q1_real.py`` already fits. Same z-test
    ``compare_with_real`` already uses, with a length-matched real side
    instead of the manifest's full-length one.

    Both sides use the sender/persona-clustered fixed-effects fit, not the
    VB sentence model -- the VB fit's posterior SD understates uncertainty
    on this outcome (PROGRESS_llms.md, Oct 4: a sender-clustered robust SE
    came out about twice the VB-implied one on the full-length version of
    this same comparison). The VB-based version is still returned under
    ``"vb"``, not dropped, the way ``primary`` keeps ``coefficient_vb``.
    """
    max_sentences = max(1, round(len(result.sentence_features) / len(result.reply_features)))
    truncated = truncate_sentences(real_sentences, max_sentences)
    matched = fit_direction_fixed_effects(
        truncated, SENTENCE_OUTCOME, cluster_col=SENDER_COL, reference=REFERENCE, family="logistic"
    )
    matched_vb = fit_sentence_model(truncated, control_rank=True, per_email=True)

    persona_fe = result.sentence_model_persona_fe_pooled or result.sentence_model_persona_fe
    model_contrasts = {f"is_imperative:{level}": persona_fe.contrast(level) for level in CONTRASTS}
    vb_model_contrasts = {
        k: v for k, v in grid_contrasts(result).items() if k.startswith("is_imperative:")
    }

    def _synthetic_real(fit: FixedEffectsResult | SentenceModelResult) -> dict[str, Any]:
        return {
            "simulator_vs_real": {
                f"is_imperative:{level}": {
                    "real": fit.contrast(level)[0],
                    "real_p": fit.contrast(level)[1],
                }
                for level in CONTRASTS
            }
        }

    comparison = compare_with_real(model_contrasts, _synthetic_real(matched))
    comparison_vb = compare_with_real(vb_model_contrasts, _synthetic_real(matched_vb))
    return {"max_sentences": max_sentences, "vb": comparison_vb, **comparison}


def real_row(real_manifest: Mapping[str, Any]) -> dict[str, float]:
    """The real-email benchmark for the primary contrast, with its interval.

    Uses the sender-clustered fixed-effects fit
    (:func:`thesis.analysis.q1_real.robust_contrasts`) when the manifest
    carries one, since ``PRIMARY`` is ``is_imperative:down`` and that
    outcome's VB fit understates its own uncertainty (PROGRESS_llms.md,
    Oct 4). Falls back to ``simulator_vs_real``'s VB contrast for a
    manifest built before this fit existed, or a hand-built test fixture
    that only sets ``simulator_vs_real``.
    """
    robust = robust_contrasts(real_manifest)
    if PRIMARY in robust:
        coefficient, p_value = robust[PRIMARY]
    else:
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
    grids: Sequence[tuple[str, Path]],
    real_manifest: Mapping[str, Any],
    *,
    real_sentences: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fit every grid and collect the numbers one section needs."""
    models: dict[str, dict[str, Any]] = {}
    for label, path in grids:
        log.info("fitting %s from %s", label, path)
        models[label] = summarize_model(
            run_q1_analysis(load_grid(path)), real_manifest, real_sentences=real_sentences
        )
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
    real_sentences = pd.read_parquet(SENTENCES_PATH) if SENTENCES_PATH.exists() else None

    manifest = build_manifest(
        [parse_grid_arg(g) for g in args.grid], real_manifest, real_sentences=real_sentences
    )
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
