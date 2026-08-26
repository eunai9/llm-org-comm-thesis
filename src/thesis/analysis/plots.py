"""Figures for the judge results reported in PROGRESS.md.

PROGRESS.md is a plain-language log meant to be read by a person, and the
judge results in it are all comparisons *across settings* -- real versus
generated on each rubric dimension, one model family judging another,
separability under successively fairer comparisons. A grid of numbers makes
the reader reconstruct the pattern mentally; a figure shows the direction
and size of the change directly, which is where the finding actually lives.

**Form follows the question, one chart each.**

- :func:`plot_paired_dimensions` -- a dumbbell (paired-dot) plot, because the
  quantity of interest is the *gap* between two scores on the same dimension,
  and a connecting segment encodes that gap as length. Grouped bars would
  make the reader compare bar tops across a gutter instead.
- :func:`plot_factor_interaction` -- an interaction plot, because whether two
  factors interact is exactly whether the lines are parallel; that is the
  one visual question the judge-swap design exists to ask.
- :func:`plot_discrimination_auc` -- horizontal bars against an explicit
  chance line, because an AUC is meaningless without 0.5 in the frame.

**Colors come from the data-viz reference palette unchanged** (categorical
slots 1 and 2, blue/orange), whose colorblind separation is validated at the
source. Nothing here re-steps or substitutes a hue.

Figures are written as PNG with the light chart surface baked in, rather
than transparent: they are embedded in a Markdown file rendered on a
background this code does not control, and a transparent ground would put
dark ink on a dark page for any reader using a dark theme.

Regenerate everything with ``python -m thesis.analysis.plots``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display in WSL; must be set before pyplot is imported.

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from thesis.paths import DOCS_FIGURES_DIR

# From references/palette.md, light mode, used as published.
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

_FONT = ["DejaVu Sans", "sans-serif"]


def _style_axes(ax: Axes) -> None:
    """Recessive chrome: hairline grid, no box, muted ticks."""
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_SECONDARY)


def _finish(fig: Figure, ax: Axes, title: str, subtitle: str, path: Path) -> Path:
    ax.set_title(title, color=INK_PRIMARY, fontsize=12.5, fontweight="600", loc="left", pad=18)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            color=INK_SECONDARY,
            fontsize=9.5,
            va="bottom",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_paired_dimensions(
    dimensions: Sequence[str],
    real_scores: Sequence[float],
    generated_scores: Sequence[float],
    path: Path,
    *,
    title: str = "Judge scores: real vs generated reply to the same message",
    subtitle: str = "",
) -> Path:
    """Dumbbell plot of two paired score series across rubric dimensions.

    Ordered by gap size so the dimensions that differ most sit together
    rather than scattered through the axis.
    """
    if not (len(dimensions) == len(real_scores) == len(generated_scores)):
        msg = (
            f"dimensions, real_scores and generated_scores must be the same length; "
            f"got {len(dimensions)}, {len(real_scores)}, {len(generated_scores)}"
        )
        raise ValueError(msg)

    order = sorted(range(len(dimensions)), key=lambda i: real_scores[i] - generated_scores[i])
    labels = [dimensions[i].replace("_", " ") for i in order]
    real = [real_scores[i] for i in order]
    gen = [generated_scores[i] for i in order]
    y = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(labels) + 1.9))
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)

    for yi, r, g in zip(y, real, gen, strict=True):
        ax.plot([g, r], [yi, yi], color=BASELINE, linewidth=2, zorder=1, solid_capstyle="round")
    # 2px surface ring so the two dots stay separable where they nearly overlap.
    ax.scatter(gen, y, s=95, color=SERIES_2, zorder=3, edgecolors=SURFACE, linewidths=2)
    ax.scatter(real, y, s=95, color=SERIES_1, zorder=3, edgecolors=SURFACE, linewidths=2)

    ax.set_yticks(y, labels)
    ax.set_xlabel("mean rubric score (1-5)", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylim(-0.7, len(labels) - 0.3)

    # Direct-label one row rather than every dot, and pick the row with the
    # widest gap: on a narrow row the two labels would print on top of each
    # other, which is exactly what happened when this labelled the first row.
    anchor = max(y, key=lambda i: abs(real[i] - gen[i]))
    ax.annotate(
        "real",
        (real[anchor], anchor),
        textcoords="offset points",
        xytext=(0, 14),
        ha="center",
        color=SERIES_1,
        fontsize=9,
        fontweight="600",
    )
    ax.annotate(
        "generated",
        (gen[anchor], anchor),
        textcoords="offset points",
        xytext=(0, 14),
        ha="center",
        color=SERIES_2,
        fontsize=9,
        fontweight="600",
    )
    return _finish(fig, ax, title, subtitle, path)


def plot_factor_interaction(
    x_levels: Sequence[str],
    series: dict[str, Sequence[float]],
    path: Path,
    *,
    title: str,
    subtitle: str = "",
    x_label: str = "",
    y_label: str = "mean rubric score (1-5)",
) -> Path:
    """Interaction plot: one line per series across the levels of one factor.

    Non-parallel lines are the interaction. Capped at two series, which is
    what the judge-swap design produces and what keeps the parallelism
    readable.
    """
    if not 1 <= len(series) <= 2:
        msg = f"expected 1 or 2 series for an interaction plot, got {len(series)}"
        raise ValueError(msg)
    for name, values in series.items():
        if len(values) != len(x_levels):
            msg = f"series {name!r} has {len(values)} values but there are {len(x_levels)} levels"
            raise ValueError(msg)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    _style_axes(ax)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)

    x = list(range(len(x_levels)))
    for color, (name, values) in zip((SERIES_1, SERIES_2), series.items(), strict=False):
        ax.plot(
            x,
            values,
            color=color,
            linewidth=2,
            marker="o",
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=name,
            zorder=3,
        )
        ax.annotate(
            name,
            (x[-1], values[-1]),
            textcoords="offset points",
            xytext=(10, 0),
            va="center",
            color=color,
            fontsize=9.5,
            fontweight="600",
        )

    ax.set_xticks(x, list(x_levels))
    ax.set_xlim(-0.35, len(x_levels) - 0.35)
    ax.set_xlabel(x_label, color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel(y_label, color=INK_SECONDARY, fontsize=9.5)
    # Upper left: the direct labels already occupy the right-hand margin, and
    # both series rise left-to-right, so this corner is the empty one.
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return _finish(fig, ax, title, subtitle, path)


def plot_discrimination_auc(
    labels: Sequence[str],
    aucs: Sequence[float],
    path: Path,
    *,
    title: str = "Model-free discrimination: can a word-count classifier tell them apart?",
    subtitle: str = "",
) -> Path:
    """Horizontal bars measuring distance above chance.

    **Anchored at 0.5, not 0.** An AUC of 0 is not "no separation" -- 0.5
    is, and a bar drawn from zero encodes half its length as meaning that
    does not exist, making every result look alike. The usual
    bars-must-start-at-zero rule is about not truncating a scale whose zero
    is meaningful; here chance *is* the meaningful origin, and the axis
    starts there explicitly rather than quietly.

    A single series, so no legend -- the title names what is plotted. Every
    bar is directly labelled: at four values that reads as a value list
    rather than clutter.
    """
    if len(labels) != len(aucs):
        msg = f"labels and aucs must be the same length; got {len(labels)}, {len(aucs)}"
        raise ValueError(msg)
    if any(not 0.5 <= a <= 1.0 for a in aucs):
        msg = f"AUCs below chance or above 1 cannot be drawn on this scale: {list(aucs)}"
        raise ValueError(msg)

    # barh puts index 0 at the bottom; reverse so the first item passed reads
    # first, top-down, the order the accompanying prose walks through.
    y = list(range(len(labels)))[::-1]

    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(labels) + 1.8))
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)

    ax.barh(y, [a - 0.5 for a in aucs], left=0.5, height=0.34, color=SERIES_1, zorder=2)
    ax.axvline(0.5, color=BASELINE, linewidth=1.5, zorder=3)
    ax.annotate(
        "0.5 = indistinguishable",
        (0.5, max(y) + 0.55),
        textcoords="offset points",
        xytext=(6, 0),
        color=INK_MUTED,
        fontsize=9,
        va="center",
    )

    for yi, value in zip(y, aucs, strict=True):
        ax.annotate(
            f"{value:.3f}",
            (value, yi),
            textcoords="offset points",
            xytext=(7, 0),
            va="center",
            color=INK_SECONDARY,
            fontsize=9.5,
        )

    ax.set_yticks(y, list(labels))
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.6, max(y) + 0.9)
    ax.set_xlabel("AUC", color=INK_SECONDARY, fontsize=9.5)
    return _finish(fig, ax, title, subtitle, path)


def main() -> None:
    """Regenerate every judge figure referenced by PROGRESS.md.

    Values are the reported results themselves, kept here so the figures and
    the log cannot drift apart: changing a number means changing it in one
    place that produces the picture the log shows.
    """
    rubric = (
        "clarity",
        "role_consistency",
        "politeness_appropriateness",
        "conflict_management",
        "corpus_plausibility",
        "contextual_fit",
    )
    plot_paired_dimensions(
        rubric,
        real_scores=(4.70, 4.58, 4.45, 4.50, 4.25, 4.12),
        generated_scores=(4.72, 4.62, 4.62, 4.67, 4.53, 4.62),
        path=DOCS_FIGURES_DIR / "judge_paired_fidelity.png",
        subtitle="40 matched pairs, both sides formatted identically. Gap runs generated-high throughout.",
    )

    plot_factor_interaction(
        ("llama-generated", "qwen-generated"),
        {"llama judging": (3.725, 4.328), "qwen judging": (2.675, 3.697)},
        path=DOCS_FIGURES_DIR / "judge_swap_interaction.png",
        title="Judge swap: does a model rate its own family higher?",
        subtitle="Non-parallel lines are self-preference. 120 replies, each scored by both models.",
        x_label="which model wrote the reply",
    )

    plot_discrimination_auc(
        (
            "unmatched format\n(subject-line artifact)",
            "format-matched",
            "format-matched,\nentities scrubbed",
            "body text only",
        ),
        (0.887, 0.719, 0.819, 0.841),
        path=DOCS_FIGURES_DIR / "judge_discrimination_auc.png",
        subtitle="Same 40 pairs the judge could not separate. No LLM involved.",
    )


if __name__ == "__main__":
    main()
