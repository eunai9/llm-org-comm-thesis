"""Where generated replies land relative to real ones, in embedding space.

This is a validation check, not a research question: before any result about
hierarchy or fidelity means anything, the generated text has to be shown to
occupy roughly the region of message-space the real corpus occupies. The judge
already says the two are equivalent on a rubric, and a TF-IDF classifier
already says they are almost perfectly separable -- so a representation-level
look is what adjudicates between those two answers.

Three views, each answering a narrower version of "are these the same kind of
text?":

- **The map.** t-SNE over sentence embeddings of the real reply and the
  generated reply to the same stimulus. Read only for overlap; t-SNE distances
  and cluster sizes carry no interpretation.
- **The same map with length controlled.** The known dominant difference is
  length (generated replies are ~20 words, real ones ~85), and length is
  visible to an embedding model. Each real reply is therefore truncated to its
  own pair's generated word count and everything is recomputed. Whatever
  separation survives is not about length.
- **Topical tracking.** Cosine similarity between a generated reply and the
  real reply it was matched to, against the same generated reply and a real
  reply from a *different* thread. If the two distributions coincide, the
  simulator is writing generic email that ignores its stimulus; the gap
  between them is how much the stimulus actually drives the output.

**Folds are grouped by thread.** A real and a generated reply to the same
stimulus share their topic, so splitting a pair across train and test would let
the classifier recognise the thread rather than the authorship -- the same leak
:mod:`thesis.analysis.fidelity` fixed for the TF-IDF classifier.

Run with ``python -m thesis.analysis.embedding_map``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from thesis.analysis.pairs import PAIRS_PATH
from thesis.analysis.plots import plot_discrimination_auc, plot_embedding_map, plot_value_spread
from thesis.llm.embeddings import DEFAULT_EMBED_MODEL, embed_texts, l2_normalize
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import DOCS_FIGURES_DIR, MANIFESTS_DIR, ensure_dirs

log = get_logger(__name__)

# Fixed so the map is the same picture in a supervision meeting as it was when
# it was written up. t-SNE has no stable solution without it.
SEED = 20260830
N_FOLDS = 5

REAL = "real reply"
GENERATED = "generated reply"


@dataclass(frozen=True, slots=True)
class MapResult:
    """Everything the write-up quotes from one view of the embedding space."""

    view: str
    n_pairs: int
    separability_auc: float
    mean_real_words: float
    mean_generated_words: float


def truncate_words(text: str, n_words: int) -> str:
    """First ``n_words`` whitespace-separated tokens of ``text``."""
    return " ".join(text.split()[: max(n_words, 1)])


def separability_auc(
    vectors: np.ndarray,
    is_generated: Sequence[bool],
    groups: Sequence[str],
    *,
    n_folds: int = N_FOLDS,
) -> float:
    """Cross-validated AUC of a linear classifier separating real from generated.

    Grouped by thread so that a pair never straddles the split, and stratified
    so both classes appear in every fold.
    """
    labels = np.asarray(is_generated, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scores = cross_val_score(
        LogisticRegression(max_iter=2000),
        l2_normalize(vectors),
        labels,
        groups=np.asarray(groups),
        cv=splitter,
        scoring="roc_auc",
    )
    return float(np.mean(scores))


def project(vectors: np.ndarray, *, perplexity: float = 30.0) -> np.ndarray:
    """2-D t-SNE projection over cosine distance, seeded for reproducibility."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    )
    return np.asarray(tsne.fit_transform(vectors))


def matched_vs_mismatched_cosine(
    real_vectors: np.ndarray, generated_vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of each generated reply to its own real reply, and to another's.

    The mismatched comparison is a derangement of the real replies (a fixed
    random shift), so every generated reply is compared against exactly one
    real reply from a different thread -- the same number of comparisons as the
    matched set, which keeps the two distributions directly comparable.
    """
    real = l2_normalize(real_vectors)
    generated = l2_normalize(generated_vectors)
    matched = np.sum(real * generated, axis=1)
    shifted = np.roll(real, shift=len(real) // 2, axis=0)
    mismatched = np.sum(shifted * generated, axis=1)
    return matched, mismatched


def _view(
    frame: pd.DataFrame,
    real_texts: Sequence[str],
    generated_texts: Sequence[str],
    *,
    view: str,
    embed_model: str,
    figure_path: Path,
    title: str,
    subtitle: str,
) -> tuple[MapResult, np.ndarray, np.ndarray]:
    texts = list(real_texts) + list(generated_texts)
    vectors = embed_texts(texts, model=embed_model)
    n = len(real_texts)
    real_vectors, generated_vectors = vectors[:n], vectors[n:]

    labels = [REAL] * n + [GENERATED] * n
    groups = list(frame["thread_id"]) * 2
    auc = separability_auc(vectors, [False] * n + [True] * n, groups)

    coords = project(vectors)
    plot_embedding_map(coords, labels, figure_path, title=title, subtitle=subtitle)

    result = MapResult(
        view=view,
        n_pairs=n,
        separability_auc=round(auc, 3),
        mean_real_words=round(float(np.mean([len(t.split()) for t in real_texts])), 1),
        mean_generated_words=round(float(np.mean([len(t.split()) for t in generated_texts])), 1),
    )
    return result, real_vectors, generated_vectors


def run(
    frame: pd.DataFrame,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    figure_prefix: str = "embedding_",
) -> dict[str, object]:
    """Produce the maps, the separability bars, and the topical-tracking check.

    Successively fairer versions of the same comparison, so that whatever
    separability remains at the end cannot be attributed to a formatting
    confound rather than to how the two actually write:

    1. the real reply exactly as the corpus stores it;
    2. the same reply with any quoted ancestor removed by the current cleaner;
    3. that, truncated to the generated reply's own word count.

    **Step 2 disappears once the corpus itself is clean**, and that is the
    normal case after the Aug 31 rebuild: re-cleaning already-clean text is the
    identity, so drawing it as a separate bar would invite the reader to
    compare two numbers that are the same measurement twice. It is reported as
    skipped instead.

    ``figure_prefix`` names the figures. Re-running this against new data must
    not overwrite the figures a written-up section already points at -- the
    same reason the judge figures carry a per-run suffix.
    """
    stored_texts = list(frame["real_reply_body"])
    real_texts = list(frame["real_reply_body_recleaned"])
    generated_texts = list(frame["generated_reply"])
    quotes_already_stripped = stored_texts == real_texts
    n_pairs = len(frame)

    stored, real_vectors, generated_vectors = _view(
        frame,
        stored_texts,
        generated_texts,
        view="as stored",
        embed_model=embed_model,
        figure_path=DOCS_FIGURES_DIR / f"{figure_prefix}map_stored.png",
        title="Real and generated replies to the same messages, embedded",
        subtitle=(
            f"Real replies as the corpus stores them. t-SNE, n={n_pairs} pairs; "
            "read for overlap only."
        ),
    )

    views = [stored]
    bars: list[tuple[str, float]] = [("1. real reply as stored", stored.separability_auc)]

    if not quotes_already_stripped:
        cleaned, real_vectors, generated_vectors = _view(
            frame,
            real_texts,
            generated_texts,
            view="quotes removed",
            embed_model=embed_model,
            figure_path=DOCS_FIGURES_DIR / f"{figure_prefix}map_quotes_removed.png",
            title="The same comparison once quoted ancestors are removed",
            subtitle="Real replies re-cleaned with the current quote stripper.",
        )
        views.append(cleaned)
        bars.append(("2. old quoted email cut off", cleaned.separability_auc))

    truncated = [
        truncate_words(real, len(generated.split()))
        for real, generated in zip(real_texts, generated_texts, strict=True)
    ]
    matched, _, _ = _view(
        frame,
        truncated,
        generated_texts,
        view="length-matched",
        embed_model=embed_model,
        figure_path=DOCS_FIGURES_DIR / f"{figure_prefix}map_length_matched.png",
        title="The same comparison with reply length held equal",
        subtitle=(
            "Each real reply cut to its own pair's generated word count. "
            "Separation surviving this is not about length."
        ),
    )
    views.append(matched)
    bars.append(
        (f"{len(bars) + 1}. also cut to the same\nlength as the AI reply", matched.separability_auc)
    )

    plot_discrimination_auc(
        [label for label, _ in bars],
        [value for _, value in bars],
        path=DOCS_FIGURES_DIR / f"{figure_prefix}separability_auc.png",
        title="Can a program tell a real reply from an AI one?",
        subtitle=(
            "How often it guesses right. Each row takes away one clue that is not about "
            "writing style."
        ),
        x_label="how often the program guessed right",
        chance_note="0.5 = pure guessing",
    )

    matched_cos, mismatched_cos = matched_vs_mismatched_cosine(real_vectors, generated_vectors)
    plot_value_spread(
        {
            "generated vs. its own real reply": matched_cos.tolist(),
            "generated vs. another thread's real reply": mismatched_cos.tolist(),
        },
        path=DOCS_FIGURES_DIR / f"{figure_prefix}topical_tracking.png",
        title="Does a generated reply track the message it was answering?",
        subtitle=f"Cosine similarity in embedding space, n={n_pairs} each.",
        x_label="cosine similarity",
        annotate_distinct=False,
    )

    return {
        "embedding_model": embed_model,
        "seed": SEED,
        "quotes_already_stripped": quotes_already_stripped,
        "views": [asdict(view) for view in views],
        "topical_tracking": {
            "mean_cosine_matched": round(float(np.mean(matched_cos)), 3),
            "mean_cosine_mismatched": round(float(np.mean(mismatched_cos)), 3),
            "share_matched_higher": round(float(np.mean(matched_cos > mismatched_cos)), 3),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=str(PAIRS_PATH))
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument(
        "--figure-prefix",
        default="embedding_",
        help="Filename prefix for the figures, so a re-run does not overwrite reported ones.",
    )
    parser.add_argument("--out", default=str(MANIFESTS_DIR / "embedding_map.json"))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    frame = pd.read_parquet(args.pairs)
    summary = run(frame, embed_model=args.embed_model, figure_prefix=args.figure_prefix)

    out = Path(args.out)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log.info("embedding map summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
