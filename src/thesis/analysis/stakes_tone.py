"""Q1: do stakes and the incoming message's tone change qwen's replies,
inside each direction?

The qwen3.8-27b main set crosses direction with stakes (routine, high) and
the tone of the incoming message (deferential, warm, neutral, assertive),
but every Q1 result so far pooled over those two factors. This module fits,
for each direction separately, one regression per measure:

    measure ~ stakes + incoming tone + task type + persona

with routine stakes and a neutral incoming message as the baselines and
HC3 standard errors. A Holm correction runs over every stakes and tone
coefficient. A second model on all directions together tests whether the
stakes or tone effects differ by direction (an F-test on the interaction
terms).

The measures are the gpt-oss-120b tone score (1-7), politeness from the
TyDiP classifier (log-odds), the share of order sentences, the share of
hedge sentences, and length in words. All 1,440 replies of draw 1 are used.

Run with ``python -m thesis.analysis.stakes_tone``. Add ``--judge`` to send
the replies the tone judge has not scored yet to gpt-oss-120b on Groq;
without it, only cached tone scores are used.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from thesis.analysis.judge_swap import client_for_model
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
from thesis.analysis.politeness import cached_tone_scores, politeness_scores
from thesis.analysis.role_inference import GENERATING_GRIDS, build_absolute_items_from_grid
from thesis.analysis.tone_judge import DEFAULT_JUDGE_MODEL, score_tone
from thesis.data.features import (
    _HEDGE_PHRASES,
    _HEDGE_TOKENS,
    _count_phrase_hits,
    _has_obligation_modal,
    _load_nlp,
    is_imperative,
)
from thesis.judge.prompt import JudgeItem
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import (
    CACHE_DIR,
    COST_LEDGER,
    DOCS_FIGURES_DIR,
    INTERIM_DIR,
    MANIFESTS_DIR,
    ensure_dirs,
)

log = get_logger(__name__)

WRITER = "qwen3.8-27b"
SCORES_PATH = INTERIM_DIR / "q1_qwen_stakes_tone_scores.parquet"
MANIFEST_PATH = MANIFESTS_DIR / "stakes_tone_qwen.json"
FIGURE_PATH = DOCS_FIGURES_DIR / "stakes_tone_qwen.png"
MEASURES = ("tone", "politeness_logit", "order_share", "hedge_share", "words")
DIRECTIONS = ("down", "lateral", "up")
TERMS = ("high", "assertive", "deferential", "warm")
_FACTORS = "C(stakes, Treatment('routine')) + C(tone_in, Treatment('neutral'))"
_CONTROLS = "C(task_type) + C(persona_id)"
_LOGIT_EPS = 1e-6


def load_grid() -> pd.DataFrame:
    """qwen's draw-1 replies, with the incoming tone taken from the scenario id."""
    grid = pd.read_parquet(GENERATING_GRIDS[WRITER])
    grid = grid[grid["replicate"] == 1].copy()
    grid["tone_in"] = grid["scenario_id"].str.split("__").str[-1]
    return grid


def local_scores(items: list[JudgeItem]) -> pd.DataFrame:
    """Order share, hedge share, length and politeness of each reply."""
    nlp = _load_nlp()
    rows = []
    for item in items:
        sents = list(nlp(item.text).sents)
        n = max(len(sents), 1)
        orders = sum(bool(is_imperative(s) or _has_obligation_modal(s)) for s in sents)
        hedges = sum(
            _count_phrase_hits(s.text.lower(), _HEDGE_PHRASES)
            + sum(t.lower_ in _HEDGE_TOKENS for t in s)
            > 0
            for s in sents
        )
        rows.append((item.item_id, orders / n, hedges / n, len(item.text.split())))
    frame = pd.DataFrame(rows, columns=["cell_id", "order_share", "hedge_share", "words"])
    frame["politeness"] = politeness_scores([item.text for item in items])
    return frame


def tone_scores(items: list[JudgeItem], cache: ResponseCache, *, judge: bool) -> dict[str, int]:
    """gpt-oss-120b tone scores: all items with ``judge``, else cached ones only."""
    if not judge:
        return cached_tone_scores(items, cache)
    client = client_for_model(DEFAULT_JUDGE_MODEL)
    scores, summary = score_tone(
        items, client, model=DEFAULT_JUDGE_MODEL, cache=cache, ledger=CostLedger(COST_LEDGER)
    )
    log.info("tone judge: %s", summary)
    return scores


def fit_within(frame: pd.DataFrame, measure: str) -> list[dict[str, Any]]:
    """Stakes and incoming-tone coefficients of ``measure``, one model per direction."""
    rows = []
    for direction in DIRECTIONS:
        part = frame[frame["direction"] == direction]
        fit = smf.ols(f"{measure} ~ {_FACTORS} + {_CONTROLS}", data=part).fit(cov_type="HC3")
        for term in fit.params.index:
            if "stakes" in term or "tone_in" in term:
                rows.append(
                    {
                        "measure": measure,
                        "direction": direction,
                        "term": term.split("[T.")[1].rstrip("]"),
                        "coef": float(fit.params[term]),
                        "se": float(fit.bse[term]),
                        "p": float(fit.pvalues[term]),
                        "n": int(fit.nobs),
                    }
                )
    return rows


def interaction_tests(frame: pd.DataFrame, measure: str) -> dict[str, float]:
    """p-values of the F-tests that the stakes or tone effects differ by direction."""
    fit = smf.ols(
        f"{measure} ~ C(direction, Treatment('lateral')) * ({_FACTORS}) + {_CONTROLS}",
        data=frame,
    ).fit(cov_type="HC3")
    inter = [t for t in fit.params.index if ":" in t]
    return {
        f"{factor}_x_direction_p": float(
            fit.f_test([f"{t} = 0" for t in inter if factor in t]).pvalue
        )
        for factor in ("stakes", "tone_in")
    }


def plot_effects(coef: pd.DataFrame) -> None:
    """One panel per measure: each factor level's effect with a 95% interval,
    one dot per direction."""
    panels = [
        ("tone", "Authoritative tone (1-7)"),
        ("politeness_logit", "Politeness (log-odds)"),
        ("order_share", "Order share"),
        ("words", "Length (words)"),
    ]
    colors = {"down": SERIES_1, "lateral": SERIES_3, "up": SERIES_2}
    names = {"down": "Writing down", "lateral": "To a peer", "up": "Writing up"}
    labels = ["High stakes", "Assertive msg", "Deferential msg", "Warm msg"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (measure, title) in zip(axes, panels, strict=True):
        _style_axes(ax)
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=1)
        ax.set_axisbelow(True)
        ax.axvline(0, color=BASELINE, linewidth=1.5)
        for j, direction in enumerate(DIRECTIONS):
            part = coef[(coef["measure"] == measure) & (coef["direction"] == direction)]
            part = part.set_index("term").loc[list(TERMS)]
            y = np.arange(len(TERMS))[::-1] + (1 - j) * 0.22
            ax.errorbar(
                part["coef"],
                y,
                xerr=1.96 * part["se"],
                fmt="o",
                color=colors[direction],
                markersize=6,
                elinewidth=2,
                label=names[direction],
            )
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
    axes[0].set_yticks(np.arange(len(TERMS))[::-1], labels)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        ncol=3,
        frameon=False,
        fontsize=9.5,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.95),
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        "qwen3.8-27b: effect of stakes and incoming tone, inside each direction",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.9,
        "Difference from routine stakes / a neutral incoming message. Dots: estimate; bars: 95% CI.",
        color=INK_SECONDARY,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge", action="store_true", help="Score uncached replies with the tone judge."
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()
    grid = load_grid()
    items, _ = build_absolute_items_from_grid(grid)

    if SCORES_PATH.exists():
        scores = pd.read_parquet(SCORES_PATH)
    else:
        scores = local_scores(items)
        scores.to_parquet(SCORES_PATH)
    scores = scores.drop(columns="tone", errors="ignore")
    scores["tone"] = scores["cell_id"].map(
        tone_scores(items, ResponseCache(CACHE_DIR), judge=args.judge)
    )
    clipped = scores["politeness"].clip(_LOGIT_EPS, 1 - _LOGIT_EPS)
    scores["politeness_logit"] = np.log(clipped / (1 - clipped))
    frame = grid.merge(scores, on="cell_id")

    rows = [row for m in MEASURES for row in fit_within(frame.dropna(subset=[m]), m)]
    coef = pd.DataFrame(rows)
    coef["p_holm"] = multipletests(coef["p"], method="holm")[1]

    manifest: dict[str, Any] = {
        "writer": WRITER,
        "judge": DEFAULT_JUDGE_MODEL,
        "n": len(frame),
        "n_tone": int(frame["tone"].notna().sum()),
        "baselines": {"stakes": "routine", "tone_in": "neutral"},
        "means": {
            m: {
                factor: frame.groupby(["direction", factor])[m]
                .mean()
                .unstack()
                .round(3)
                .to_dict(orient="index")
                for factor in ("stakes", "tone_in")
            }
            for m in MEASURES
        },
        "within_direction": coef.round(4).to_dict(orient="records"),
        "direction_interaction": {
            m: interaction_tests(frame.dropna(subset=[m]), m) for m in MEASURES
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    plot_effects(coef)
    log.info("wrote %s and %s", MANIFEST_PATH, FIGURE_PATH)


if __name__ == "__main__":
    main()
