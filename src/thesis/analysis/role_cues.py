"""Layer 2 of the new Q1 plan: which cues make hierarchy visible to the judge.

Layer 1 (``role_inference.py``) asked whether a blind judge can tell, from
one email, whether it was written up, to a peer or down. This module asks
which measurable cues drive that judge's guess. For each source (each
model's Q1 grid, and real email for reference) it reads the judge's guess
for every item from the response cache, measures cues on the exact text the
judge saw, and fits a logistic regression of "the judge says down" (and
"the judge says up") on the standardized cues.

No model is called: every guess comes from the cache that Layer 1 filled,
and a cache miss raises instead of reaching Groq.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import statsmodels.api as sm
from spacy.tokens import Doc

from thesis.analysis.plots import plot_coefficient_heatmap
from thesis.analysis.q1_measures import (
    GILBERT_PATH,
    gilbert_score,
    gilbert_tokens,
    load_gilbert_phrases,
    politeness_markers,
    pronoun_rates,
)
from thesis.analysis.q1_real import load_bodies
from thesis.analysis.role_inference import (
    GENERATING_GRIDS,
    REAL_SAMPLE_SEED,
    build_absolute_items_from_grid,
    build_absolute_items_from_real_email,
    run_role_inference_absolute,
    sample_real_directions,
)
from thesis.data.features import _load_nlp, extract_features
from thesis.judge.prompt import JudgeItem
from thesis.llm.base import CompletionRequest, CompletionResponse, Provider
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import CACHE_DIR, DOCS_FIGURES_DIR, INTERIM_DIR, MANIFESTS_DIR, ensure_dirs

log = get_logger(__name__)

JUDGE_MODEL = "openai/gpt-oss-120b@low"
REAL_SAMPLE_SIZE = 1000
CUES: Final[tuple[str, ...]] = (
    "imperative_ratio",
    "hedge_rate",
    "you_rate",
    "gratitude",
    "gilbert_score",
    "log_words",
)
CUE_LABELS: Final[dict[str, str]] = {
    "imperative_ratio": "orders (share of sentences)",
    "hedge_rate": "hedges",
    "you_rate": '"you" rate',
    "gratitude": "gratitude",
    "gilbert_score": "hierarchy phrases",
    "log_words": "length (log words)",
}
OUTCOMES: Final[tuple[str, ...]] = ("down", "up")
MANIFEST_PATH = MANIFESTS_DIR / "role_cues.json"


class CacheMissError(RuntimeError):
    """A judge call this module needs is not in the cache."""


class CacheOnlyClient:
    """Stands in for the Groq client so cached guesses are read, never bought."""

    provider: Provider = "groq"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        msg = "judge call not in cache; run thesis.analysis.role_inference first"
        raise CacheMissError(msg)


def judge_guesses(
    items: list[JudgeItem],
    true_directions: dict[str, str],
    *,
    model: str,
    cache: ResponseCache,
) -> pd.DataFrame:
    """The judge's guess for every item, read from the cache."""
    with tempfile.TemporaryDirectory() as tmp:
        results, _ = run_role_inference_absolute(
            items,
            CacheOnlyClient(),
            model=model,
            cache=cache,
            ledger=CostLedger(Path(tmp) / "ledger.csv"),
            run_id="role_cues",
            true_directions=true_directions,
        )
    return pd.DataFrame(
        [
            {
                "item_id": r.item_id,
                "true_direction": r.true_direction,
                "guess": r.inferred_direction,
            }
            for r in results
        ]
    )


def reply_cues(text: str, doc: Doc, phrases: Mapping[tuple[str, ...], float]) -> dict[str, float]:
    """Every cue for one text, measured the same way as the Q1 measures."""
    features = extract_features("item", doc)
    pronouns = pronoun_rates(text)
    tokens = gilbert_tokens(text)
    return {
        "imperative_ratio": features.imperative_ratio,
        "hedge_rate": features.hedge_rate,
        "you_rate": pronouns.you_rate,
        "gratitude": float(politeness_markers(doc).gratitude),
        "gilbert_score": 100 * gilbert_score(tokens, phrases) / max(len(tokens), 1),
        "log_words": float(np.log(max(pronouns.n_words, 1))),
    }


def fit_cue_model(frame: pd.DataFrame, *, cues: Sequence[str], outcome: str) -> pd.DataFrame:
    """Logistic regression of "the judge says ``outcome``" on standardized cues.

    Each coefficient is the change in log-odds per one standard deviation of
    the cue, with robust (HC1) standard errors. A cue with no variation in
    ``frame`` is left out.
    """
    used = [c for c in cues if frame[c].std() > 0]
    x = (frame[used] - frame[used].mean()) / frame[used].std()
    y = (frame["guess"] == outcome).astype(float)
    fit = sm.Logit(y, sm.add_constant(x)).fit(disp=False, cov_type="HC1")
    ci = fit.conf_int()
    return pd.DataFrame(
        {
            "cue": used,
            "coefficient": [float(fit.params[c]) for c in used],
            "se": [float(fit.bse[c]) for c in used],
            "p": [float(fit.pvalues[c]) for c in used],
            "ci_low": [float(ci.loc[c, 0]) for c in used],
            "ci_high": [float(ci.loc[c, 1]) for c in used],
        }
    )


def _cue_frame(
    items: list[JudgeItem], guesses: pd.DataFrame, phrases: Mapping[tuple[str, ...], float]
) -> pd.DataFrame:
    texts = {item.item_id: item.text for item in items}
    ids = guesses["item_id"].tolist()
    docs = _load_nlp().pipe([texts[i] for i in ids])
    cues = pd.DataFrame(
        [reply_cues(texts[i], doc, phrases) for i, doc in zip(ids, docs, strict=True)]
    )
    return pd.concat([guesses.reset_index(drop=True), cues], axis=1)


def _sources() -> dict[str, tuple[list[JudgeItem], dict[str, str]]]:
    sources = {
        label: build_absolute_items_from_grid(pd.read_parquet(path))
        for label, path in GENERATING_GRIDS.items()
    }
    real_emails = pd.read_parquet(INTERIM_DIR / "q1_real_emails.parquet")
    real_directions = sample_real_directions(
        real_emails[["message_uid", "direction"]].drop_duplicates(subset="message_uid"),
        n=REAL_SAMPLE_SIZE,
        seed=REAL_SAMPLE_SEED,
    )
    real_bodies = load_bodies(real_directions["message_uid"])
    return {
        "real email": build_absolute_items_from_real_email(real_bodies, real_directions)
    } | sources


def main() -> None:
    configure_logging()
    ensure_dirs()
    cache = ResponseCache(CACHE_DIR)
    phrases = load_gilbert_phrases(GILBERT_PATH)

    manifest: dict[str, Any] = {"judge": JUDGE_MODEL, "cues": list(CUES), "sources": {}}
    for label, (items, truth) in _sources().items():
        guesses = judge_guesses(items, truth, model=JUDGE_MODEL, cache=cache)
        frame = _cue_frame(items, guesses, phrases)
        entry: dict[str, Any] = {
            "n": len(frame),
            "guess_shares": frame["guess"].value_counts(normalize=True).round(4).to_dict(),
            "cue_means": frame[list(CUES)].mean().round(4).to_dict(),
        }
        for outcome in OUTCOMES:
            table = fit_cue_model(frame, cues=CUES, outcome=outcome)
            entry[f"judge_says_{outcome}"] = table.round(4).to_dict("records")
        manifest["sources"][label] = entry
        log.info("%s: %d items", label, len(frame))

    labels = list(manifest["sources"])
    for outcome in OUTCOMES:
        values = np.full((len(CUES), len(labels)), np.nan)
        pvalues = np.full_like(values, np.nan)
        for j, label in enumerate(labels):
            for row in manifest["sources"][label][f"judge_says_{outcome}"]:
                i = CUES.index(row["cue"])
                values[i, j], pvalues[i, j] = row["coefficient"], row["p"]
        manifest[f"figure_{outcome}"] = str(
            plot_coefficient_heatmap(
                [CUE_LABELS[c] for c in CUES],
                labels,
                values,
                pvalues,
                DOCS_FIGURES_DIR / f"role_cues_{outcome}.png",
                title=f'Which cues make the judge say "{outcome}"?',
                subtitle="Change in log-odds per 1 SD of the cue. * p<.05, ** p<.01, *** p<.001.",
            )
        )
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
