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
    """How often the same email gets the same decision across every draw.

    ``share_expected`` is what unrelated draws with these same totals would
    match by luck. ``kappa`` is how far the observed agreement gets from
    that luck level towards perfect: 0 is no better than luck, 1 is perfect
    agreement. At exactly two draws this is Cohen's kappa and
    ``share_agree``/``share_expected`` are pairwise. At three or more it is
    Fleiss' kappa (the standard generalization past two raters) and both
    shares mean "every draw agrees", not just one pair -- see
    :func:`decision_stability_n`. ``counts`` is the draw-1-against-draw-2
    cross-table; it is only defined for exactly two draws, and empty
    otherwise, since a cross-table has no natural shape past two raters.
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
    """The statistic :func:`decision_stability` needs, and
    :func:`decision_stability_n` delegates to at exactly two draws:
    agreement, the cross-table, and Cohen's kappa, given the two draws
    already paired one row per item. Shared here so the pairs-table shape
    and the grid shape compute it the same way."""
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


def _fleiss_kappa(draws: Sequence[pd.Series]) -> DecisionStability:
    """Fleiss' kappa: the standard generalization of Cohen's kappa past two
    raters, for ``draws`` items rated by the same fixed number of raters
    each (here, every item has one decision per draw).

    ``share_agree``/``n_agree`` mean every draw agreeing on an item, not
    just one pair -- with :math:`k` draws there is no single pair left, the
    same reason :func:`_measure_reliability` reports a mean over pairs for
    its correlations instead of one pair's. ``share_expected`` is the chance
    that :math:`k` raters, each independently drawing from the pooled
    category rates, would all land on the same category by luck.

    Notation follows Fleiss (1971): :math:`n_{ij}` is how many of the
    :math:`k` raters put item :math:`i` in category :math:`j`; :math:`p_j`
    is category :math:`j`'s share of all :math:`N \\times k` ratings;
    :math:`P_i` is item :math:`i`'s own agreement rate; :math:`\\bar P` and
    :math:`P_e` are the mean observed and mean chance agreement.
    """
    table = pd.concat(draws, axis=1)
    n_items, k = table.shape
    labels = sorted(set(table.to_numpy().ravel()))
    counts = np.array(
        [[int((table.iloc[i] == label).sum()) for label in labels] for i in range(n_items)]
    )

    p_j = counts.sum(axis=0) / (n_items * k)
    p_i = (np.sum(counts**2, axis=1) - k) / (k * (k - 1))
    p_bar = float(p_i.mean())
    p_e = float(np.sum(p_j**2))
    kappa = 0.0 if p_e == 1.0 else (p_bar - p_e) / (1.0 - p_e)

    all_agree = np.all(table.eq(table.iloc[:, 0], axis=0), axis=1)
    return DecisionStability(
        n=n_items,
        n_agree=int(all_agree.sum()),
        share_agree=round(float(all_agree.mean()), 3),
        share_expected=round(float(np.sum(p_j**k)), 3),
        kappa=round(float(kappa), 3),
        counts={},
    )


def decision_stability_n(draws: Sequence[pd.Series]) -> DecisionStability:
    """:func:`_decision_stability` for exactly two draws, unchanged (so
    every already-published two-draw kappa stays exactly what it was), and
    :func:`_fleiss_kappa` for three or more.

    Public because :mod:`thesis.analysis.role_inference` reuses this for
    judge self-consistency (agreement across repeated judge calls on the
    same item), not only for draw-to-draw agreement. The statistic itself
    doesn't care what produced the repeated observations."""
    if len(draws) == 2:
        return _decision_stability(draws[0], draws[1])
    return _fleiss_kappa(draws)


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


def _icc_2_1(values: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, single measurement, absolute
    agreement, for any number of raters (here, draws of the same cell).

    0 means the cell tells you nothing about the score; a random guess would
    do as well. 1 means every draw agrees exactly. Unlike a correlation,
    this is sensitive to a draw that is systematically higher or lower than
    the others, not only to whether they rank cells the same way. The two
    raters (draws) that gave this its name are the minimum, not a limit --
    ``values`` is ``n_cells`` rows by ``k`` draws, ``k >= 2``, and every term
    below is already written in terms of ``k``, so nothing changes here
    when it is more than 2.
    """
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
    """One continuous measure's agreement across every draw of every cell.

    ``pearson`` and ``spearman`` say how well one draw predicts another, for
    one cell. With exactly two draws that is the one pair's correlation;
    with more, it is the mean correlation over every pair, since there is no
    longer a single pair to report. ``icc`` asks the same question on the
    measure's own scale, not just its ranking, computed directly from every
    draw at once (:func:`_icc_2_1`), not by averaging pairs.
    ``spearman_brown_k2``/``k3`` use the mean pairwise ``pearson`` to say
    what a 2-draw or 3-draw average of this measure would reach.
    """

    measure: str
    n_cells: int
    pearson: float
    spearman: float
    icc: float
    spearman_brown_k2: float
    spearman_brown_k3: float


def _mean_pairwise_correlation(draws: Sequence[pd.Series], *, method: str = "pearson") -> float:
    """The mean correlation over every pair of draws. One pair (two draws)
    has only itself to report; more draws have no single pair, so this
    stands in for "the" pairwise correlation the same way it always has."""
    pairs = [
        draws[i].corr(draws[j], method=method)
        for i in range(len(draws))
        for j in range(i + 1, len(draws))
    ]
    return float(np.mean(pairs))


def _measure_reliability(measure: str, draws: Sequence[pd.Series]) -> MeasureReliability:
    pearson = _mean_pairwise_correlation(draws, method="pearson")
    values = np.column_stack([d.to_numpy(dtype=float) for d in draws])
    return MeasureReliability(
        measure=measure,
        n_cells=len(draws[0]),
        pearson=round(pearson, 3),
        spearman=round(_mean_pairwise_correlation(draws, method="spearman"), 3),
        icc=round(_icc_2_1(values), 3),
        spearman_brown_k2=round(spearman_brown(pearson, 2), 3),
        spearman_brown_k3=round(spearman_brown(pearson, 3), 3),
    )


@dataclass(frozen=True, slots=True)
class GridReliability:
    """Every draw against every other, across every cell of a multi-draw
    Q1 grid.

    The same question section 50 asked of 183 pairs, generalized to every
    cell a grid holds, and (since the main Q1 run's third draw, Sep 23) to
    any number of draws, not only two. See :func:`grid_draw_reliability`.
    """

    n_cells: int
    k_draws: int
    imperative_ratio: MeasureReliability
    hedge_rate: MeasureReliability
    decision: DecisionStability


def grid_draw_reliability(reply_features: pd.DataFrame) -> GridReliability:
    """Every draw against every other, across every cell of a multi-draw
    Q1 grid.

    ``reply_features`` is one row per generated reply (as
    :func:`thesis.analysis.q1.extract_q1_reply_features` returns), carrying
    ``cell_id``, ``replicate``, ``decision``, ``imperative_ratio`` and
    ``hedge_rate``. Rows are grouped by cell identity
    (:func:`thesis.analysis.hierarchy.cell_id_without_replicate`), the same
    way :func:`merge_draws` pairs the pairs-table shape by ``pair_key``, and
    only cells present in every draw are kept -- a cell missing one draw
    (a validation failure, say) cannot be compared across all of them.

    Two draws is still the common case and reads exactly as it always has
    (:func:`_measure_reliability`, :func:`decision_stability_n` both
    special-case it). Three or more draws is new, found when the main Q1
    run's third draw hit the old two-draws-only version of this function
    (PROGRESS_llms.md, Sep 23).
    """
    replicates = sorted(reply_features["replicate"].unique())
    if len(replicates) < 2:
        msg = f"grid_draw_reliability needs at least two draws; got {replicates}"
        raise ValueError(msg)

    base = reply_features.assign(cell_base_id=cell_id_without_replicate(reply_features["cell_id"]))
    wide = base.pivot(
        index="cell_base_id",
        columns="replicate",
        values=["imperative_ratio", "hedge_rate", "decision"],
    )
    wide = wide.dropna()  # keep only cells present in every draw
    if wide.empty:
        msg = "no cell appears in every draw"
        raise ValueError(msg)

    def _draws(measure: str) -> list[pd.Series]:
        return [wide[measure][r].reset_index(drop=True) for r in replicates]

    return GridReliability(
        n_cells=len(wide),
        k_draws=len(replicates),
        imperative_ratio=_measure_reliability("imperative_ratio", _draws("imperative_ratio")),
        hedge_rate=_measure_reliability("hedge_rate", _draws("hedge_rate")),
        decision=decision_stability_n(_draws("decision")),
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
