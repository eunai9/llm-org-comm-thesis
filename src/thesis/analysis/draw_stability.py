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

import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

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


def decision_stability(first: pd.DataFrame, second: pd.DataFrame) -> DecisionStability:
    """Agreement, the cross-table, and Cohen's kappa for the decision field."""
    merged = merge_draws(first, second)
    draw1, draw2 = merged["decision_1"], merged["decision_2"]
    labels = sorted(set(draw1) | set(draw2))
    table = pd.crosstab(draw1, draw2).reindex(index=labels, columns=labels, fill_value=0)
    n = len(merged)
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
