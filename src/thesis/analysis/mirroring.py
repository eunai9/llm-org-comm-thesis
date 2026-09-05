"""Does a reply hand the sender's own request straight back?

Reading 100 generated replies by hand (section 35) found this as the single
most common failure: asked to approve two vacation days, the persona answers
"Can you confirm that these dates are acceptable?"; asked to send a list to
Richard, it answers "Can you send the list to Richard?". A quarter of the
sample did it. The LLM judge cannot see it at all -- it is shown the reply and
not the message the reply answers -- so the finding existed only as a number
one reader produced by reading, computable for no other reply in the project.

This module makes it measurable. Nothing here calls a model: it is spaCy plus
set arithmetic, so it runs over every reply the project has ever generated, and
over the real human replies as the reference point that says what an ordinary
value looks like.

**Three signals, kept separate rather than blended into one index.** Each is a
different claim about what mirroring *is*, and which of them actually tracks
the hand codes is a question for the data, not for the author of the module:

- ``borrowed_words`` -- the share of the reply's own content vocabulary that
  already appeared in the incoming message. A reply that gives the request back
  is built out of the sender's nouns and verbs.
- ``longest_repeat`` -- the longest run of consecutive words repeated verbatim
  from the incoming message, as a share of the reply's length. Catches
  near-quotation, which ``borrowed_words`` scores no higher than ordinary
  topical overlap.
- ``returned_request`` -- ``borrowed_words``, but only when the incoming message
  asks for something *and* the reply also asks for something. This is the
  closest to the concept: mirroring is answering a request with the same
  request, and neither half alone is a failure.

**The validation is against the hand codes, and it is honest about what that
can show.** :func:`validate` reports how well each signal separates the 25
replies coded as mirroring from the other 75, as an AUC. A threshold for a
yes/no rule can only be chosen by looking at those same 100 items, so any
precision or recall quoted at that threshold is optimistic and stays labelled
as such until a second coder's sheet exists.

Run with ``python -m thesis.analysis.mirroring``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import spacy
from scipy import stats
from sklearn.metrics import roc_auc_score
from spacy.language import Language
from spacy.tokens import Doc, Span

from thesis.analysis.embedding_map import truncate_words
from thesis.analysis.pairs import PAIRS_PATH
from thesis.analysis.plots import (
    plot_category_counts,
    plot_discrimination_auc,
    plot_value_spread,
)
from thesis.analysis.review_summary import DIRECTION_LABELS
from thesis.data.features import is_imperative
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import DOCS_FIGURES_DIR, MANIFESTS_DIR, TABLES_DIR, ensure_dirs

log = get_logger(__name__)

CODED_PATH: Path = TABLES_DIR / "manual_review_coded_first_pass.csv"

# The signal the write-up quotes. It is the plainest of the three and the one
# that tracks the hand codes best; the alternatives are kept and reported
# rather than dropped, so that choice stays checkable.
HEADLINE_SIGNAL = "borrowed_words"

# Where "most of this reply is the sender's own words" starts. Chosen by
# looking at the 100 coded replies, which makes any precision or recall quoted
# at it optimistic -- it is used for a *rate to compare across groups*, never
# as evidence that the level itself is right. A second coder's sheet is what
# would fix that.
HIGH_OVERLAP = 0.80

# "Can you send it", "Could we discuss", "Please review" -- a request that is
# not structurally an imperative. Kept deliberately narrow: a bare question
# mark is not a request ("Did the deal close?" asks for information, which is a
# perfectly good reply to make), so the interrogative test below also requires
# the second person.
_POLITE_REQUEST = re.compile(r"\b(?:can|could|would|will)\s+(?:you|we)\b|\bplease\b", re.I)
_SECOND_PERSON = re.compile(r"\byou\b|\byour\b", re.I)

# Signals, in the order the write-up walks through them.
SIGNALS: tuple[str, ...] = ("borrowed_words", "longest_repeat", "returned_request")


@dataclass(frozen=True, slots=True)
class MirroringFeatures:
    """One reply, measured against the message it answers."""

    borrowed_words: float
    longest_repeat: float
    returned_request: float
    reply_is_request: bool
    stimulus_is_request: bool
    reply_words: int


def load_nlp() -> Language:
    """spaCy with the lemmatizer on.

    :mod:`thesis.data.features` disables it for throughput over 254k messages.
    Here there are a few hundred texts and matching "sending" to "send" is the
    whole point of comparing vocabularies, so it is worth the cost.
    """
    return spacy.load("en_core_web_sm", disable=["ner"])


def _content_lemmas(doc: Doc) -> list[str]:
    """Lower-cased lemmas of the content-bearing tokens, in order."""
    return [
        token.lemma_.lower()
        for token in doc
        if not (token.is_stop or token.is_punct or token.is_space)
    ]


def _is_request(sent: Span) -> bool:
    text = sent.text.strip()
    if is_imperative(sent):
        return True
    if _POLITE_REQUEST.search(text) is not None:
        return True
    return text.endswith("?") and _SECOND_PERSON.search(text) is not None


def contains_request(doc: Doc) -> bool:
    """Whether any sentence asks the other party to do something."""
    return any(_is_request(sent) for sent in doc.sents)


def longest_shared_run(reply: Sequence[str], stimulus: Sequence[str]) -> int:
    """Length of the longest run of words appearing in both, in the same order.

    Standard longest-common-substring dynamic programme over token lists. The
    texts are short -- a reply is ~20 words and a stimulus at most a few
    hundred -- so the quadratic table is a few thousand cells.
    """
    if not reply or not stimulus:
        return 0
    previous = [0] * (len(stimulus) + 1)
    best = 0
    for i in range(1, len(reply) + 1):
        current = [0] * (len(stimulus) + 1)
        for j in range(1, len(stimulus) + 1):
            if reply[i - 1] == stimulus[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def features_for(stimulus: Doc, reply: Doc) -> MirroringFeatures:
    """Measure one reply against the message it answers."""
    reply_lemmas = _content_lemmas(reply)
    stimulus_lemmas = _content_lemmas(stimulus)

    if reply_lemmas:
        shared = set(reply_lemmas) & set(stimulus_lemmas)
        borrowed = len(shared) / len(set(reply_lemmas))
        repeat = longest_shared_run(reply_lemmas, stimulus_lemmas) / len(reply_lemmas)
    else:
        borrowed = 0.0
        repeat = 0.0

    reply_asks = contains_request(reply)
    stimulus_asks = contains_request(stimulus)
    # Gated, not blended: when either side is not a request there is no request
    # to hand back, and scoring such a pair on borrowed vocabulary alone would
    # count ordinary topical overlap as a failure.
    returned = borrowed if (reply_asks and stimulus_asks) else 0.0

    return MirroringFeatures(
        borrowed_words=round(borrowed, 4),
        longest_repeat=round(repeat, 4),
        returned_request=round(returned, 4),
        reply_is_request=reply_asks,
        stimulus_is_request=stimulus_asks,
        reply_words=len(reply.text.split()),
    )


def score_texts(
    stimuli: Sequence[str],
    replies: Sequence[str],
    *,
    nlp: Language | None = None,
) -> pd.DataFrame:
    """Measure every (incoming message, reply) pair. One row per pair, in order."""
    if len(stimuli) != len(replies):
        msg = f"stimuli and replies must be the same length; got {len(stimuli)}, {len(replies)}"
        raise ValueError(msg)
    nlp = nlp or load_nlp()
    stimulus_docs = list(nlp.pipe(list(stimuli)))
    reply_docs = list(nlp.pipe(list(replies)))
    rows = [
        asdict(features_for(stimulus, reply))
        for stimulus, reply in zip(stimulus_docs, reply_docs, strict=True)
    ]
    return pd.DataFrame.from_records(rows)


def validate(scored: pd.DataFrame, is_mirrored: Sequence[bool]) -> dict[str, float]:
    """How well each signal separates the hand-coded mirrored replies from the rest."""
    labels = [int(value) for value in is_mirrored]
    return {signal: round(float(roc_auc_score(labels, scored[signal])), 3) for signal in SIGNALS}


@dataclass(frozen=True, slots=True)
class MeasureChange:
    """One quantity, before and after, with the paired test on the difference."""

    before: float
    after: float
    change: float
    p_value: float


@dataclass(frozen=True, slots=True)
class RunComparison:
    """What a prompt change did to the mirroring measure, on the same stimuli."""

    n_paired: int
    borrowed_words: MeasureChange
    flagged_share: MeasureChange
    newly_flagged: int
    no_longer_flagged: int
    reply_words: MeasureChange


def compare_runs(
    before: pd.DataFrame, after: pd.DataFrame, *, nlp: Language | None = None
) -> RunComparison:
    """Did a prompt change move the measure, on the same stimuli?

    Paired by ``cell_id``: the same persona answering the same real message
    under two prompts, so the comparison holds everything except the prompt
    fixed and the test can be a signed-rank on the per-reply difference rather
    than a two-sample test that throws that pairing away.

    Reported alongside it is McNemar's test on the flagged/not-flagged pairs,
    which is the right test for "did the *rate* move" when the same items are
    counted twice -- a chi-square would treat the two runs as independent
    samples and overstate the evidence.
    """
    nlp = nlp or load_nlp()
    merged = before.merge(after, on="cell_id", suffixes=("_before", "_after"))
    if merged.empty:
        msg = "no cell_id appears in both runs; the two files are not the same design"
        raise ValueError(msg)

    scores = {
        arm: score_texts(merged[f"stimulus_text_{arm}"], merged[f"generated_reply_{arm}"], nlp=nlp)[
            HEADLINE_SIGNAL
        ]
        for arm in ("before", "after")
    }
    difference = scores["after"] - scores["before"]
    # Every reply identical would make the signed-rank test undefined; that is
    # a real outcome (the prompt changed nothing), not an error to raise on.
    p_value = (
        float(stats.wilcoxon(scores["after"], scores["before"]).pvalue) if difference.any() else 1.0
    )

    flagged = {arm: scores[arm] >= HIGH_OVERLAP for arm in ("before", "after")}
    moved_up = int((~flagged["before"] & flagged["after"]).sum())
    moved_down = int((flagged["before"] & ~flagged["after"]).sum())
    mcnemar = (
        float(stats.binomtest(moved_down, moved_down + moved_up, 0.5).pvalue)
        if moved_down + moved_up
        else 1.0
    )

    words = {
        arm: float(merged[f"generated_reply_{arm}"].str.split().str.len().mean())
        for arm in ("before", "after")
    }

    return RunComparison(
        n_paired=len(merged),
        borrowed_words=MeasureChange(
            before=round(float(scores["before"].mean()), 3),
            after=round(float(scores["after"].mean()), 3),
            change=round(float(difference.mean()), 3),
            p_value=round(p_value, 4),
        ),
        flagged_share=MeasureChange(
            before=round(float(flagged["before"].mean()), 3),
            after=round(float(flagged["after"].mean()), 3),
            change=round(float(flagged["after"].mean() - flagged["before"].mean()), 3),
            p_value=round(mcnemar, 4),
        ),
        newly_flagged=moved_up,
        no_longer_flagged=moved_down,
        # No test on length: it is reported because a prompt that cures
        # mirroring by making replies longer is a different result from one
        # that cures it at the same length, not because length is an outcome
        # this comparison is designed to test.
        reply_words=MeasureChange(
            before=round(words["before"], 1),
            after=round(words["after"], 1),
            change=round(words["after"] - words["before"], 1),
            p_value=float("nan"),
        ),
    )


def run(
    pairs: pd.DataFrame, coded: pd.DataFrame, *, nlp: Language | None = None
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score every pair, validate against the hand codes, and summarize both.

    The real human reply to the same message is scored too, twice: as written,
    and cut to the generated reply's own length. Without the second one the
    comparison is unfair in a way that flatters the simulator's critics --
    ``borrowed_words`` is a share of a reply's distinct vocabulary, and a long
    reply has more room for words the sender never used, so a real reply would
    score lower for being long even if it echoed just as much.
    """
    nlp = nlp or load_nlp()
    stimuli = list(pairs["stimulus_text"])
    generated = list(pairs["generated_reply"])
    real = list(pairs["real_reply_body_recleaned"])
    real_cut = [
        truncate_words(text, len(reply.split()))
        for text, reply in zip(real, generated, strict=True)
    ]

    scored = pairs[["cell_id", "thread_id", "persona_id", "direction"]].reset_index(drop=True)
    for name, replies in (
        ("generated", generated),
        ("real", real),
        ("real_length_matched", real_cut),
    ):
        measured = score_texts(stimuli, replies, nlp=nlp).add_prefix(f"{name}_")
        scored = pd.concat([scored, measured], axis=1)

    coded_scores = score_texts(coded["stimulus_text"], coded["generated_reply"], nlp=nlp)
    is_mirrored = (coded["failure_mode"] == "mirrors_request").tolist()
    aucs = validate(coded_scores, is_mirrored)

    flagged = scored["generated_borrowed_words"] >= HIGH_OVERLAP
    summary: dict[str, object] = {
        "n_pairs": len(scored),
        "n_hand_coded": len(coded),
        "auc_vs_hand_codes": aucs,
        "headline_signal": HEADLINE_SIGNAL,
        "threshold": HIGH_OVERLAP,
        "threshold_chosen_on_the_coded_sample": True,
        "mean_borrowed_words": {
            "generated": round(float(scored["generated_borrowed_words"].mean()), 3),
            "real": round(float(scored["real_borrowed_words"].mean()), 3),
            "real_length_matched": round(
                float(scored["real_length_matched_borrowed_words"].mean()), 3
            ),
        },
        "share_over_threshold": {
            "generated": round(float(flagged.mean()), 3),
            "real_length_matched": round(
                float((scored["real_length_matched_borrowed_words"] >= HIGH_OVERLAP).mean()), 3
            ),
        },
        "flagged_rate_by_direction": {
            str(direction): {
                "rate": round(float(group.mean()), 3),
                "n": int(group.size),
            }
            for direction, group in flagged.groupby(scored["direction"])
        },
    }
    return scored, summary


def plot(
    scored: pd.DataFrame, summary: dict[str, object], *, figure_prefix: str = "mirroring_"
) -> list[Path]:
    """Three figures: what the measure detects, what it finds, and where.

    ``figure_prefix`` keeps a re-run on a different generation from overwriting
    the figures an already-written section points at.
    """
    aucs: dict[str, float] = summary["auc_vs_hand_codes"]  # type: ignore[assignment]
    paths = [
        plot_discrimination_auc(
            (
                "words borrowed\nfrom the sender",
                "longest repeated\nphrase",
                "both sides ask\nfor something",
            ),
            (aucs["borrowed_words"], aucs["longest_repeat"], aucs["returned_request"]),
            path=DOCS_FIGURES_DIR / f"{figure_prefix}signal_auc.png",
            title="Which signal finds the replies a reader called mirroring?",
            subtitle=(
                "Checked against 100 hand-coded replies. 0.5 = the signal tells them apart "
                "no better than chance."
            ),
            x_label="how well the signal separates them",
            chance_note="0.5 = no better than chance",
        ),
        plot_value_spread(
            {
                "AI replies": scored["generated_borrowed_words"].tolist(),
                "real replies, cut to the same length": scored[
                    "real_length_matched_borrowed_words"
                ].tolist(),
            },
            path=DOCS_FIGURES_DIR / f"{figure_prefix}generated_vs_real.png",
            title="How much of a reply is built from the sender's own words?",
            subtitle=(
                "Share of the reply's distinct content words that already appeared in the "
                f"message it answers. n={len(scored)} each."
            ),
            x_label="share of the reply's words taken from the sender",
            annotate_distinct=False,
        ),
    ]

    by_direction: dict[str, dict[str, float]] = summary["flagged_rate_by_direction"]  # type: ignore[assignment]
    order = sorted(by_direction, key=lambda key: by_direction[key]["rate"])
    paths.append(
        plot_category_counts(
            [
                f"{DIRECTION_LABELS.get(key, key)} (n={int(by_direction[key]['n'])})"
                for key in order
            ],
            [round(by_direction[key]["rate"] * 100) for key in order],
            DOCS_FIGURES_DIR / f"{figure_prefix}rate_by_direction.png",
            title="Replies built mostly from the sender's own words, by who is written to",
            subtitle=(
                f"Share scoring at or above {HIGH_OVERLAP:.2f}. That cut-off was chosen by "
                "looking at the 100 coded replies, so read the level with care and the "
                "differences with more."
            ),
            x_label="% of replies",
        )
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=str(PAIRS_PATH))
    parser.add_argument("--coded", default=str(CODED_PATH))
    parser.add_argument("--out", default=str(TABLES_DIR / "mirroring_scores.csv"))
    parser.add_argument(
        "--figure-prefix",
        default="mirroring_",
        help="Filename prefix for the figures, so a re-run does not overwrite reported ones.",
    )
    parser.add_argument(
        "--manifest",
        default=str(MANIFESTS_DIR / "mirroring.json"),
        help="Where to write the summary. Give a re-run its own file, as with the figures.",
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        metavar="PAIRS",
        help="A second pairs file to compare against, paired by cell_id.",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    pairs = pd.read_parquet(args.pairs)
    scored, summary = run(pairs, pd.read_csv(args.coded))
    if args.compare_to:
        summary["compared_with_previous_prompt"] = asdict(
            compare_runs(pd.read_parquet(args.compare_to), pairs)
        )
    scored.to_csv(args.out, index=False)
    for path in plot(scored, summary, figure_prefix=args.figure_prefix):
        log.info("wrote %s", path)
    Path(args.manifest).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log.info("mirroring summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
