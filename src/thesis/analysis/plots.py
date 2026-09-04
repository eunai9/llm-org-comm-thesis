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

**One exception to that.** :func:`plot_value_spread` needs the raw
per-reply values rather than a handful of summary numbers, so its figure is
produced by the analysis that holds the data instead of by :func:`main`.
Hardcoding a distribution here would mean inventing values that were never
measured, which is worse than the figure living slightly out of the way.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")  # No display in WSL; must be set before pyplot is imported.

import matplotlib.pyplot as plt
import numpy as np
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

# The subset of matplotlib's legend positions these charts use; typed as a
# Literal so a typo is a type error rather than a silently ignored string.
LegendLoc = Literal["upper left", "upper right", "lower left", "lower right"]


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
    legend_loc: LegendLoc = "upper left",
) -> Path:
    """Interaction plot: one line per series across the levels of one factor.

    Non-parallel lines are the interaction. Capped at two series, which is
    what the judge-swap design produces and what keeps the parallelism
    readable.

    ``legend_loc`` exists because which corner is empty depends on the data:
    the default suits series that rise left-to-right, but a series that
    starts high collides with it, so the caller picks.
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
    legend = ax.legend(frameon=False, fontsize=9, loc=legend_loc)
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


def plot_value_spread(
    groups: dict[str, Sequence[float]],
    path: Path,
    *,
    title: str,
    subtitle: str = "",
    x_label: str = "",
    bins: int = 24,
    annotate_distinct: bool = True,
) -> Path:
    """Two groups' value distributions, drawn as side-by-side histograms.

    Built for a measurement question rather than an outcome one: whether a
    variable takes enough distinct values to resolve the effect being
    looked for. A per-sentence rate computed over one-sentence text piles
    onto a handful of spikes, and that is visible here in a way no summary
    statistic conveys -- a mean and standard deviation look perfectly
    healthy for a variable that only ever takes three values.

    ``annotate_distinct`` turns off the distinct-value count for the other use
    of this chart, comparing two continuous distributions: there, every value
    is distinct by construction, so the count says nothing and the mean is what
    the reader wants marked instead.
    """
    if len(groups) != 2:
        msg = f"expected exactly 2 groups to compare, got {len(groups)}"
        raise ValueError(msg)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)
    all_values = [v for values in groups.values() for v in values]
    edges = np.linspace(min(all_values), max(all_values), bins + 1)

    for ax, color, (name, values) in zip(axes, (SERIES_1, SERIES_2), groups.items(), strict=True):
        _style_axes(ax)
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
        ax.set_axisbelow(True)
        ax.hist(list(values), bins=edges, color=color, zorder=2)
        if annotate_distinct:
            n_distinct = len(set(values))
            label = f"{name} — {n_distinct} distinct value{'s' if n_distinct != 1 else ''}"
        else:
            mean = float(np.mean(list(values)))
            label = f"{name} — mean {mean:.2f}"
            ax.axvline(mean, color=INK_SECONDARY, linewidth=1.2, linestyle="--", zorder=3)
        ax.annotate(
            label,
            (0.99, 0.86),
            xycoords="axes fraction",
            ha="right",
            color=color,
            fontsize=9.5,
            fontweight="600",
        )
        ax.set_ylim(top=ax.get_ylim()[1] * 1.28)
        ax.set_ylabel("replies", color=INK_SECONDARY, fontsize=9)

    axes[1].set_xlabel(x_label, color=INK_SECONDARY, fontsize=9.5)
    fig.subplots_adjust(hspace=0.18)
    return _finish(fig, axes[0], title, subtitle, path)


def plot_category_counts(
    labels: Sequence[str],
    counts: Sequence[int],
    path: Path,
    *,
    title: str,
    subtitle: str = "",
    x_label: str = "",
    highlight: Sequence[str] = (),
) -> Path:
    """Counts across mutually exclusive categories, as horizontal bars.

    Horizontal because category names are words, not numbers, and rotating
    them under a vertical axis costs the reader a head-tilt for nothing.
    Sorted by the caller, not here: the order is usually part of the argument
    being made (largest first, or the "everything fine" category first), and
    re-sorting would quietly overrule it.

    ``highlight`` names the categories drawn in the second series colour --
    for a chart whose point is that one category dominates, colour carries
    that rather than a caption asking the reader to find it.
    """
    if len(labels) != len(counts):
        msg = f"labels and counts must be the same length; got {len(labels)}, {len(counts)}"
        raise ValueError(msg)

    y = list(range(len(labels)))[::-1]
    colors = [SERIES_2 if label in set(highlight) else SERIES_1 for label in labels]

    fig, ax = plt.subplots(figsize=(7.4, 0.46 * len(labels) + 1.8))
    _style_axes(ax)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    ax.barh(y, list(counts), height=0.62, color=colors, zorder=2)

    for yi, value in zip(y, counts, strict=True):
        ax.annotate(
            str(value),
            (value, yi),
            textcoords="offset points",
            xytext=(6, 0),
            va="center",
            color=INK_SECONDARY,
            fontsize=9.5,
        )

    ax.set_yticks(y, list(labels))
    ax.set_xlim(0, max(counts) * 1.12)
    ax.set_xlabel(x_label, color=INK_SECONDARY, fontsize=9.5)
    return _finish(fig, ax, title, subtitle, path)


def plot_embedding_map(
    coords: np.ndarray,
    labels: Sequence[str],
    path: Path,
    *,
    title: str,
    subtitle: str = "",
    axis_label: str = "t-SNE dimension",
) -> Path:
    """Two labelled groups of messages in a 2-D projection of embedding space.

    **The axes carry no units and are deliberately left untick-ed.** A t-SNE
    coordinate is not a quantity: distances between well-separated clusters are
    not meaningful and the scale is arbitrary, so drawing ticks would invite
    exactly the over-reading the projection cannot support. What the picture
    *can* show is whether two groups occupy the same region -- the only claim
    made from it here, and one whose numeric counterpart is the classifier AUC
    reported alongside it.

    Points are semi-transparent because the interesting outcome is overlap:
    with opaque marks, whichever group is drawn second looks larger merely for
    being on top.
    """
    if len(coords) != len(labels):
        msg = f"coords and labels must be the same length; got {len(coords)}, {len(labels)}"
        raise ValueError(msg)
    names = list(dict.fromkeys(labels))
    if len(names) != 2:
        msg = f"expected exactly 2 groups to compare, got {len(names)}"
        raise ValueError(msg)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    _style_axes(ax)
    ax.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)

    labels_array = np.asarray(labels)
    for name, color in zip(names, (SERIES_1, SERIES_2), strict=True):
        mask = labels_array == name
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=26,
            color=color,
            alpha=0.62,
            linewidths=0,
            label=f"{name} (n={int(mask.sum())})",
            zorder=2,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{axis_label} 1", color=INK_SECONDARY, fontsize=9.5)
    ax.set_ylabel(f"{axis_label} 2", color=INK_SECONDARY, fontsize=9.5)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
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

    # The widened run (section 29): all 190 available pairs rather than the
    # first 40, powering the equivalence test enough to actually resolve
    # the borderline dimensions from the n=40 pilot.
    plot_paired_dimensions(
        rubric,
        real_scores=(4.46, 4.45, 4.15, 4.19, 4.26, 4.15),
        generated_scores=(4.69, 4.36, 4.24, 4.39, 4.33, 4.09),
        path=DOCS_FIGURES_DIR / "judge_paired_fidelity_n190.png",
        title="Judge scores: real vs generated reply to the same message (n=190)",
        subtitle="All 190 available pairs. Equivalence now holds on every dimension.",
    )

    plot_discrimination_auc(
        ("length alone\n(log word count)", "full text\n(TF-IDF)"),
        (0.946, 0.966),
        path=DOCS_FIGURES_DIR / "judge_discrimination_length_covariate.png",
        title="How much of the separability is just reply length?",
        subtitle="n=190 pairs. Length alone accounts for almost all of it.",
    )

    # Section 32: the same n=190 comparison re-run against the corrected
    # personas, whose style statistics are computed over the sampling
    # frame's own token band. Kept as its own figure rather than
    # overwriting the one above, since section 29's text describes that run.
    plot_paired_dimensions(
        rubric,
        real_scores=(4.57, 4.51, 4.17, 4.24, 4.25, 4.12),
        generated_scores=(4.66, 4.33, 4.33, 4.23, 4.29, 4.02),
        path=DOCS_FIGURES_DIR / "judge_paired_fidelity_corrected.png",
        title="Judge scores after the persona correction (n=190)",
        subtitle="Equivalence on all six dimensions, and now no detectable difference on any.",
    )

    # The accidental-but-controlled length manipulation: the instructed
    # target rose 46% between the two runs and output barely moved. An
    # interaction plot because the finding *is* the non-parallelism --
    # a flat response line against a steeply rising instruction line.
    plot_factor_interaction(
        ("original personas", "corrected personas"),
        {"instructed target": (53.5, 78.2), "actual output": (18.5, 19.9)},
        path=DOCS_FIGURES_DIR / "length_instruction_response.png",
        title="Does the model follow its instructed reply length?",
        subtitle="Raising the stated target by 46% moved actual output by ~7%.",
        x_label="persona style statistics used",
        y_label="words per reply",
    )

    # Section 33: the Q1 direction effect before and after the persona fix.
    # Plotted as two lines over the same x-axis because the finding is that
    # the *shape* changed -- a V with lateral lowest became a monotonic
    # gradient -- which a table of coefficients states but does not show.
    plot_factor_interaction(
        ("writing down", "writing to a peer", "writing up"),
        {"original personas": (0.475, 0.244, 0.379), "corrected personas": (0.323, 0.375, 0.431)},
        path=DOCS_FIGURES_DIR / "q1_direction_before_after.png",
        title="The Q1 direction effect did not survive the persona fix",
        subtitle="Same design, same model, corrected persona statistics. The pattern changes shape.",
        x_label="who the persona is writing to",
        y_label="mean imperative ratio",
        # The original-personas series starts high, in the default corner.
        legend_loc="lower right",
    )

    # Section 36: two independent measurement approaches -- a per-reply
    # linear ratio and a per-sentence logistic model -- converge on the
    # same small monotonic shape, even though neither reaches significance.
    # An interaction plot for the same reason as the pair above: agreement
    # in shape between two different methods is a claim about parallel
    # lines, which a coefficient table states but a reader has to take on
    # faith without seeing it.
    plot_factor_interaction(
        ("writing down", "writing to a peer", "writing up"),
        {
            "reply-level ratio (linear)": (0.323, 0.375, 0.431),
            "sentence-level probability (logistic)": (0.319, 0.347, 0.393),
        },
        path=DOCS_FIGURES_DIR / "q1_sentence_vs_reply_level.png",
        title="Two measurement grains, the same shape, neither significant",
        subtitle="Fixing the resolution problem does not rescue significance -- but it does not change the pattern either.",
        x_label="who the persona is writing to",
        y_label="imperative rate / probability",
        legend_loc="lower right",
    )

    # Section 38: Q2 re-run against the rebuilt corpus (section 37). Unlike
    # every earlier Q2 run, one dimension (role_consistency) now shows a
    # real, non-equivalent gap -- the point of this figure is to show that
    # it is the outlier, not to bury it among five dimensions that still
    # read as equivalent.
    plot_paired_dimensions(
        (
            "role_consistency",
            "contextual_fit",
            "corpus_plausibility",
            "clarity",
            "politeness_appropriateness",
            "conflict_management",
        ),
        real_scores=(4.47, 4.07, 4.20, 4.63, 4.18, 4.33),
        generated_scores=(4.24, 3.97, 4.34, 4.50, 4.13, 4.12),
        path=DOCS_FIGURES_DIR / "judge_paired_fidelity_rebuilt.png",
        title="Judge scores after the corpus rebuild (n=183)",
        subtitle="Role consistency is now a real gap (p=.015, not equivalent); the other five still read as equivalent.",
    )

    # Section 39: Q1 re-run against the rebuilt corpus (section 37), through
    # the new src/thesis/analysis/q1.py module. Plotted as the sentence-level
    # model's predicted probability (the same transform section 36 used), so
    # the shape change -- a smooth rise becoming a V, with "writing up" now
    # sitting further from lateral -- reads directly rather than needing a
    # coefficient table decoded first.
    plot_factor_interaction(
        ("writing down", "writing to a peer", "writing up"),
        {
            "pre-rebuild (section 36)": (0.319, 0.347, 0.393),
            "rebuilt corpus (section 39)": (0.317, 0.283, 0.369),
        },
        path=DOCS_FIGURES_DIR / "q1_rebuild_before_after.png",
        title="Q1 after the corpus rebuild: mostly the same null, one contrast moves",
        subtitle="Sentence-level predicted probability of an imperative sentence. 'Writing up' now differs from lateral at p=.046.",
        x_label="who the persona is writing to",
        y_label="predicted probability (sentence-level)",
    )


if __name__ == "__main__":
    main()
