"""How much does the model disagree with itself?

Section 49 ran two prompts over the same 183 emails. Only 41% of the
individual decisions matched. Two things differed between those runs: the
prompt, and the model's own randomness. So that 41% cannot be read as a
prompt effect.

This module removes the prompt from the comparison. The same prompt is run
twice over the same 183 emails, with only the draw index different. Whatever
disagreement is left is the model alone. That is the noise floor, and every
paired comparison in this project is read against it. A prompt effect the
size of the noise floor is not a prompt effect.

Nothing here calls a model. Both runs are already on disk, written by
``thesis.analysis.pairs`` with ``--draw 1`` and ``--draw 2``. Run it with
``python -m thesis.analysis.draw_stability``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from thesis.analysis.hierarchy import cell_id_without_replicate
from thesis.analysis.mirroring import (
    HEADLINE_SIGNAL,
    compare_runs,
    load_nlp,
    pair_key,
    score_texts,
)
from thesis.analysis.plots import plot_category_counts
from thesis.data.features import _load_nlp, extract_features
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import DOCS_FIGURES_DIR, INTERIM_DIR, MANIFESTS_DIR, ensure_dirs

log = get_logger(__name__)

DRAW1_PATH: Path = INTERIM_DIR / "real_vs_generated_pairs_act.parquet"
DRAW2_PATH: Path = INTERIM_DIR / "real_vs_generated_pairs_act_draw2.parquet"

# Section 49: the share of decisions that survived a change of prompt. The
# number this module produces is the same quantity with the prompt held
# fixed, so the two only mean something side by side.
AGREEMENT_ACROSS_PROMPTS = 0.41


@dataclass(frozen=True, slots=True)
class DecisionStability:
    """How often the same email gets the same decision twice.

    ``share_expected`` is what two unrelated draws with these same totals
    would match by luck. ``kappa`` is Cohen's kappa: how far the observed
    agreement gets from that luck level towards perfect. 0 is no better than
    luck. 1 is perfect agreement.
    """

    n: int
    n_agree: int
    share_agree: float
    share_expected: float
    kappa: float
    counts: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class DrawNoise:
    """The noise floor for everything else the two runs are compared on.

    Each measure is reported for both draws, with the paired test on the
    per-reply difference and the correlation between the draws. A high mean
    with a low correlation means the measure is stable in aggregate and not
    stable for one reply.
    """

    n_paired: int
    borrowed_words_draw1: float
    borrowed_words_draw2: float
    borrowed_words_change: float
    borrowed_words_p_value: float
    borrowed_words_correlation: float
    flagged_draw1: float
    flagged_draw2: float
    newly_flagged: int
    no_longer_flagged: int
    flagged_p_value: float
    words_draw1: float
    words_draw2: float
    words_change: float
    words_p_value: float
    words_correlation: float
    imperative_ratio_draw1: float
    imperative_ratio_draw2: float
    imperative_ratio_change: float
    imperative_ratio_p_value: float
    imperative_ratio_correlation: float


def merge_draws(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Join the two runs reply by reply, on persona and message.

    Same pairing :func:`thesis.analysis.mirroring.compare_runs` uses, so the
    decision numbers and the mirroring numbers describe the same rows.
    """
    merged = first.assign(pair_key=pair_key(first["cell_id"])).merge(
        second.assign(pair_key=pair_key(second["cell_id"])),
        on="pair_key",
        suffixes=("_1", "_2"),
    )
    if merged.empty:
        msg = "no persona and message appear in both draws; the two files are not the same design"
        raise ValueError(msg)
    return merged


def _decision_stability(draw1: pd.Series, draw2: pd.Series) -> DecisionStability:
    """The statistic :func:`decision_stability` and :func:`grid_draw_reliability`
    both need: agreement, the cross-table, and Cohen's kappa, given the two
    draws already paired one row per item. Shared here so the pairs-table
    shape and the grid shape compute it the same way."""
    labels = sorted(set(draw1) | set(draw2))
    table = pd.crosstab(draw1, draw2).reindex(index=labels, columns=labels, fill_value=0)
    n = len(draw1)
    expected = sum(int(table.loc[label].sum()) * int(table[label].sum()) for label in labels) / (
        n * n
    )
    return DecisionStability(
        n=n,
        n_agree=int((draw1 == draw2).sum()),
        share_agree=round(float((draw1 == draw2).mean()), 3),
        share_expected=round(float(expected), 3),
        kappa=round(float(cohen_kappa_score(draw1, draw2, labels=labels)), 3),
        counts={
            str(row): {str(column): int(value) for column, value in table.loc[row].items()}
            for row in table.index
        },
    )


def decision_stability(first: pd.DataFrame, second: pd.DataFrame) -> DecisionStability:
    """Agreement, the cross-table, and Cohen's kappa for the decision field."""
    merged = merge_draws(first, second)
    return _decision_stability(merged["decision_1"], merged["decision_2"])


def imperative_ratios(cell_ids: Sequence[str], replies: Sequence[str]) -> pd.Series:
    """Orders per reply, measured with the call ``q1.py`` uses on generated replies."""
    nlp = _load_nlp()
    docs = nlp.pipe([str(reply) for reply in replies])
    return pd.Series(
        [
            extract_features(str(cell_id), doc).imperative_ratio
            for cell_id, doc in zip(cell_ids, docs, strict=True)
        ]
    )


def _paired_p_value(first: pd.Series, second: pd.Series) -> float:
    """Signed-rank test, or 1.0 when every pair is identical."""
    if not (second - first).any():
        return 1.0
    return round(float(stats.wilcoxon(second, first).pvalue), 4)


def draw_noise(first: pd.DataFrame, second: pd.DataFrame) -> DrawNoise:
    """The noise floor: what two draws of one prompt do to every measure."""
    nlp = load_nlp()
    comparison = compare_runs(first, second, nlp=nlp)
    merged = merge_draws(first, second)

    borrowed = {
        arm: score_texts(merged[f"stimulus_text_{arm}"], merged[f"generated_reply_{arm}"], nlp=nlp)[
            HEADLINE_SIGNAL
        ]
        for arm in ("1", "2")
    }
    orders = {
        arm: imperative_ratios(merged["pair_key"], merged[f"generated_reply_{arm}"])
        for arm in ("1", "2")
    }
    # compare_runs reports length without a test, because there it is context
    # for the mirroring numbers. Here length is one of the measures whose
    # noise floor is being asked for, so it gets the same paired test as the
    # others.
    lengths = {
        arm: merged[f"generated_reply_{arm}"].fillna("").str.split().str.len().astype(float)
        for arm in ("1", "2")
    }

    return DrawNoise(
        n_paired=comparison.n_paired,
        borrowed_words_draw1=comparison.borrowed_words.before,
        borrowed_words_draw2=comparison.borrowed_words.after,
        borrowed_words_change=comparison.borrowed_words.change,
        borrowed_words_p_value=comparison.borrowed_words.p_value,
        borrowed_words_correlation=round(float(borrowed["1"].corr(borrowed["2"])), 3),
        flagged_draw1=comparison.flagged_share.before,
        flagged_draw2=comparison.flagged_share.after,
        newly_flagged=comparison.newly_flagged,
        no_longer_flagged=comparison.no_longer_flagged,
        flagged_p_value=comparison.flagged_share.p_value,
        words_draw1=comparison.reply_words.before,
        words_draw2=comparison.reply_words.after,
        words_change=comparison.reply_words.change,
        words_p_value=_paired_p_value(lengths["1"], lengths["2"]),
        words_correlation=round(float(lengths["1"].corr(lengths["2"])), 3),
        imperative_ratio_draw1=round(float(orders["1"].mean()), 3),
        imperative_ratio_draw2=round(float(orders["2"].mean()), 3),
        imperative_ratio_change=round(float((orders["2"] - orders["1"]).mean()), 3),
        imperative_ratio_p_value=_paired_p_value(orders["1"], orders["2"]),
        imperative_ratio_correlation=round(float(orders["1"].corr(orders["2"])), 3),
    )


def spearman_brown(r: float, k: int) -> float:
    """What a ``k``-draw average's reliability would be, given one draw's
    correlation ``r`` with another draw of itself.

    ``r_k = k*r / (1 + (k-1)*r)``. Section 50 measured ``r=0.15`` for orders
    per reply between two draws of one prompt: this says averaging two draws
    would reach about 0.26, and three about 0.35. It turns "is a third draw
    worth the extra generation time" into a number instead of a guess.
    """
    return k * r / (1 + (k - 1) * r)


def _icc_2_1(x: pd.Series, y: pd.Series) -> float:
    """ICC(2,1): two-way random effects, single measurement, absolute
    agreement, for exactly two raters (here, two draws of the same cell).

    0 means the cell tells you nothing about the score; a random guess would
    do as well. 1 means the two draws agree exactly. Unlike a correlation,
    this is sensitive to a draw that is systematically higher or lower than
    the other, not only to whether they rank cells the same way.
    """
    values = np.column_stack([x.to_numpy(dtype=float), y.to_numpy(dtype=float)])
    n, k = values.shape
    if n < 2:
        return float("nan")
    grand_mean = values.mean()
    subject_means = values.mean(axis=1)
    rater_means = values.mean(axis=0)

    ss_subjects = k * float(np.sum((subject_means - grand_mean) ** 2))
    ss_raters = n * float(np.sum((rater_means - grand_mean) ** 2))
    ss_total = float(np.sum((values - grand_mean) ** 2))
    ss_error = ss_total - ss_subjects - ss_raters

    ms_subjects = ss_subjects / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denominator = ms_subjects + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    if denominator == 0:
        return float("nan")
    return float((ms_subjects - ms_error) / denominator)


@dataclass(frozen=True, slots=True)
class MeasureReliability:
    """Draw 1 against draw 2 for one continuous measure, across every cell.

    ``pearson`` and ``spearman`` say how well draw 1 predicts draw 2 for one
    cell. ``icc`` asks the same question on the measure's own scale, not
    just its ranking. ``spearman_brown_k2``/``k3`` use ``pearson`` to say
    what a 2-draw or 3-draw average of this measure would reach.
    """

    measure: str
    n_cells: int
    pearson: float
    spearman: float
    icc: float
    spearman_brown_k2: float
    spearman_brown_k3: float


def _measure_reliability(measure: str, draw1: pd.Series, draw2: pd.Series) -> MeasureReliability:
    pearson = float(draw1.corr(draw2))
    return MeasureReliability(
        measure=measure,
        n_cells=len(draw1),
        pearson=round(pearson, 3),
        spearman=round(float(draw1.corr(draw2, method="spearman")), 3),
        icc=round(_icc_2_1(draw1, draw2), 3),
        spearman_brown_k2=round(spearman_brown(pearson, 2), 3),
        spearman_brown_k3=round(spearman_brown(pearson, 3), 3),
    )


@dataclass(frozen=True, slots=True)
class GridReliability:
    """Draw 1 against draw 2 across every cell of a multi-draw Q1 grid.

    The same question section 50 asked of 183 pairs, generalized to every
    cell a grid holds. See :func:`grid_draw_reliability`.
    """

    n_cells: int
    imperative_ratio: MeasureReliability
    hedge_rate: MeasureReliability
    decision: DecisionStability


def grid_draw_reliability(reply_features: pd.DataFrame) -> GridReliability:
    """Draw 1 against draw 2, across every cell of a multi-draw Q1 grid.

    ``reply_features`` is one row per generated reply (as
    :func:`thesis.analysis.q1.extract_q1_reply_features` returns), carrying
    ``cell_id``, ``replicate`` (exactly the values 1 and 2), ``decision``,
    ``imperative_ratio`` and ``hedge_rate``. Rows are paired by cell
    identity (:func:`thesis.analysis.hierarchy.cell_id_without_replicate`),
    the same way :func:`merge_draws` pairs the pairs-table shape by
    ``pair_key`` -- the statistic itself (:func:`_measure_reliability`,
    :func:`_decision_stability`) is the same code either way.
    """
    replicates = sorted(reply_features["replicate"].unique())
    if replicates != [1, 2]:
        msg = f"grid_draw_reliability needs exactly draws 1 and 2; got {replicates}"
        raise ValueError(msg)

    base = reply_features.assign(cell_base_id=cell_id_without_replicate(reply_features["cell_id"]))
    draw1 = base[base["replicate"] == 1]
    draw2 = base[base["replicate"] == 2]
    merged = draw1.merge(draw2, on="cell_base_id", suffixes=("_1", "_2"))
    if merged.empty:
        msg = "no cell appears in both draws"
        raise ValueError(msg)

    return GridReliability(
        n_cells=len(merged),
        imperative_ratio=_measure_reliability(
            "imperative_ratio", merged["imperative_ratio_1"], merged["imperative_ratio_2"]
        ),
        hedge_rate=_measure_reliability(
            "hedge_rate", merged["hedge_rate_1"], merged["hedge_rate_2"]
        ),
        decision=_decision_stability(merged["decision_1"], merged["decision_2"]),
    )


def plot(
    stability: DecisionStability,
    *,
    across_prompts: float = AGREEMENT_ACROSS_PROMPTS,
    path: Path = DOCS_FIGURES_DIR / "test_retest_decision_agreement.png",
) -> Path:
    """One bar per comparison: same prompt at a different draw, and section 49's prompt change.

    Drawn here rather than in ``plots.py`` because the value comes from the
    data this module holds. The section 49 bar is a pinned constant, the same
    way every reported figure in ``plots.py`` pins its numbers.
    """
    luck = "what luck alone would give"
    return plot_category_counts(
        [
            f"same prompt, different draw (n={stability.n})",
            luck,
            f"different prompt, section 49 (n={stability.n})",
        ],
        [
            round(stability.share_agree * 100),
            round(stability.share_expected * 100),
            round(across_prompts * 100),
        ],
        path,
        title="How often does the model give the same decision twice?",
        subtitle=(
            "Share of emails keeping the same decision. The top bar changes nothing "
            "but the draw index."
        ),
        x_label="% of emails with the same decision",
        highlight=[luck],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draw1", default=str(DRAW1_PATH))
    parser.add_argument("--draw2", default=str(DRAW2_PATH))
    parser.add_argument("--manifest", default=str(MANIFESTS_DIR / "draw_stability.json"))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    first = pd.read_parquet(args.draw1)
    second = pd.read_parquet(args.draw2)
    stability = decision_stability(first, second)
    noise = draw_noise(first, second)
    summary = {
        "decision_stability": asdict(stability),
        "noise_floor": asdict(noise),
        "decision_agreement_across_prompts_section_49": AGREEMENT_ACROSS_PROMPTS,
    }
    log.info("wrote %s", plot(stability))
    Path(args.manifest).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log.info("draw stability: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
