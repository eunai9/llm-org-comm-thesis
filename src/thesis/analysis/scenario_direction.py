"""Q1: how does the direction effect differ across the six scenarios?

Every Q1 result so far pooled over the six task types (the scenarios). This
module splits qwen3.8-27b's main set by task type and, inside each one,
compares the replies written down, to a peer and up. It reports:

- the mean of each measure per scenario and direction, with an F-test that
  direction matters inside the scenario (model: measure ~ direction + stakes
  + incoming tone + persona, HC3 errors, Holm over all scenario tests), and
  an F-test that the direction effect differs across scenarios;
- the decision mix (accept, decline, defer, ...) and a chi-square test that
  it depends on direction;
- the share of "high" confidence;
- simple keyword markers of who does the work: the reply tells the other
  person to act, or says "I'll" / "I will". These are rough counts, not a
  validated measure.

The measures come from the cache that ``thesis.analysis.stakes_tone`` built.
Run with ``python -m thesis.analysis.scenario_direction``.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from thesis.analysis.plots import (
    BASELINE,
    GRIDLINE,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES_1,
    SERIES_2,
    SERIES_3,
    SURFACE,
    _style_axes,
)
from thesis.analysis.role_inference import build_absolute_items_from_grid
from thesis.analysis.stakes_tone import SCORES_PATH, WRITER, load_grid, tone_scores
from thesis.llm.cache import ResponseCache
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import CACHE_DIR, DOCS_FIGURES_DIR, MANIFESTS_DIR, ensure_dirs

log = get_logger(__name__)

MANIFEST_PATH = MANIFESTS_DIR / "scenario_direction_qwen.json"
FIGURE_PATH = DOCS_FIGURES_DIR / "scenario_direction_qwen.png"
MEASURES = ("tone", "politeness_logit", "order_share", "hedge_share", "words")
DIRECTIONS = ("down", "lateral", "up")
_LOGIT_EPS = 1e-6
_TELLS_OTHER = (
    r"^(?:send|pull|get|verify|check|confirm|make sure|let me know|please)"
    r"|\. (?:send|pull|get|verify|check|confirm|please)"
)
_SELF_DOES = r"\bi'?ll\b|\bi will\b"
_MARKERS = {
    "tells_other": _TELLS_OTHER,
    "self_does": _SELF_DOES,
    "question": r"\?",
    "thanks": r"thank|appreciate",
}


def load_frame() -> pd.DataFrame:
    """qwen's draw-1 replies with their text, measures and keyword markers."""
    grid = load_grid()
    items, _ = build_absolute_items_from_grid(grid)
    grid["text"] = grid["cell_id"].map({item.item_id: item.text for item in items})
    scores = pd.read_parquet(SCORES_PATH).drop(columns="tone", errors="ignore")
    scores["tone"] = scores["cell_id"].map(
        tone_scores(items, ResponseCache(CACHE_DIR), judge=False)
    )
    clipped = scores["politeness"].clip(_LOGIT_EPS, 1 - _LOGIT_EPS)
    scores["politeness_logit"] = np.log(clipped / (1 - clipped))
    frame = grid.merge(scores, on="cell_id")
    lower = frame["text"].str.lower().str.replace(chr(0x2019), "'", regex=False)
    for name, pattern in _MARKERS.items():
        frame[name] = lower.str.contains(pattern, regex=True).astype(float)
    frame["high_confidence"] = (frame["confidence"] == "high").astype(float)
    return frame


def direction_p(part: pd.DataFrame, measure: str) -> float | None:
    """p-value that direction matters for ``measure`` inside one scenario."""
    if part[measure].std() == 0:
        return None
    fit = smf.ols(
        f"{measure} ~ C(direction) + C(stakes) + C(tone_in) + C(persona_id)", data=part
    ).fit(cov_type="HC3")
    terms = [t for t in fit.params.index if t.startswith("C(direction)")]
    return float(fit.f_test([f"{t} = 0" for t in terms]).pvalue)


def interaction_p(frame: pd.DataFrame, measure: str) -> float:
    """p-value that the direction effect on ``measure`` differs across scenarios."""
    fit = smf.ols(
        f"{measure} ~ C(direction) * C(task_type) + C(stakes) + C(tone_in) + C(persona_id)",
        data=frame,
    ).fit(cov_type="HC3")
    terms = [t for t in fit.params.index if ":" in t]
    return float(fit.f_test([f"{t} = 0" for t in terms]).pvalue)


def measure_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean per scenario, direction and measure, with Holm-corrected direction tests."""
    rows = []
    for task, part in frame.groupby("task_type"):
        for measure in MEASURES:
            means = part.groupby("direction")[measure].mean()
            rows.append(
                {
                    "task_type": task,
                    "measure": measure,
                    **{d: float(means[d]) for d in DIRECTIONS},
                    "down_minus_up": float(means["down"] - means["up"]),
                    "p": direction_p(part, measure),
                }
            )
    table = pd.DataFrame(rows)
    tested = table["p"].notna()
    table.loc[tested, "p_holm"] = multipletests(table.loc[tested, "p"], method="holm")[1]
    return table


def decision_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Decision mix (percent) per scenario and direction, with a chi-square p-value."""
    rows = []
    for task, part in frame.groupby("task_type"):
        counts = pd.crosstab(part["direction"], part["decision"])
        shares = counts.div(counts.sum(axis=1), axis=0).mul(100).round(1)
        rows.append(
            {
                "task_type": task,
                "percent": shares.loc[list(DIRECTIONS)].to_dict(orient="index"),
                "chi2_p": float(chi2_contingency(counts)[1]),
            }
        )
    return rows


def plot_scenarios(frame: pd.DataFrame, order: list[str]) -> None:
    """One panel per measure: the three direction means in each scenario."""
    panels = [
        ("tone", "Authoritative tone (1-7)", 1.0),
        ("politeness_logit", "Politeness (log-odds)", 1.0),
        ("order_share", "Order share (%)", 100.0),
        ("tells_other", "Tells the other to act (%)", 100.0),
    ]
    colors = {"down": SERIES_1, "lateral": SERIES_3, "up": SERIES_2}
    names = {"down": "Writing down", "lateral": "To a peer", "up": "Writing up"}
    means = frame.groupby(["task_type", "direction"])[[p[0] for p in panels]].mean()
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (measure, title, scale) in zip(axes, panels, strict=True):
        _style_axes(ax)
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=1)
        ax.set_axisbelow(True)
        values = {d: [means.loc[(t, d), measure] * scale for t in order] for d in DIRECTIONS}
        ax.hlines(
            y,
            np.minimum(values["down"], values["up"]),
            np.maximum(values["down"], values["up"]),
            color=BASELINE,
            linewidth=1.5,
            zorder=1,
        )
        for direction in DIRECTIONS:
            ax.scatter(
                values[direction],
                y,
                s=55,
                color=colors[direction],
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=2,
                label=names[direction],
            )
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
    axes[0].set_yticks(y, [t.replace("_", " ") for t in order])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        frameon=False,
        fontsize=9.5,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.95),
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        "qwen3.8-27b: the direction gap in each scenario",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.9,
        "Mean of 80 replies per dot. Scenarios sorted by the down-up tone gap, largest on top.",
        color=INK_SECONDARY,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_logging()
    ensure_dirs()
    frame = load_frame()
    table = measure_table(frame)
    gaps = table[table["measure"] == "tone"].set_index("task_type")["down_minus_up"]
    order = gaps.sort_values(ascending=False).index.tolist()
    markers = [*_MARKERS, "high_confidence"]
    marker_means = frame.groupby(["task_type", "direction"])[markers].mean().mul(100).round(1)

    manifest: dict[str, Any] = {
        "writer": WRITER,
        "n": len(frame),
        "n_per_cell": int(frame.groupby(["task_type", "direction"]).size().min()),
        "measures": table.round(4).to_dict(orient="records"),
        "direction_x_scenario_p": {m: interaction_p(frame, m) for m in MEASURES},
        "decisions": decision_table(frame),
        "markers_percent": {task: marker_means.loc[task].to_dict(orient="index") for task in order},
        "marker_patterns": _MARKERS,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    plot_scenarios(frame, order)
    log.info("wrote %s and %s", MANIFEST_PATH, FIGURE_PATH)


if __name__ == "__main__":
    main()
