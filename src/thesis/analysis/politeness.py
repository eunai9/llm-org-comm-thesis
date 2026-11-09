"""Q1: politeness of a whole email from a model trained on human ratings.

The model is ``Genius1237/xlm-roberta-large-tydip`` (Srinivasan and Choi,
2022). It is XLM-RoBERTa large, fine-tuned on the English part of TyDiP,
which takes the most and least polite requests of the Stanford Politeness
Corpus (Danescu-Niculescu-Mizil et al., 2013). Those labels come from human
ratings. The score is the model's probability that the email is polite.

The emails are the tone judge's sample (:func:`thesis.analysis.tone_judge.
build_sources`), so the two scores can be compared email by email. The
direction contrasts are fitted as for the tone score. The gpt-oss-120b tone
scores are read from the response cache only; no API call is made.

Run with ``python -m thesis.analysis.politeness``.
"""

from __future__ import annotations

import argparse
import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from thesis.analysis.hierarchy import fit_direction_fixed_effects, fit_direction_mixed_model
from thesis.analysis.tone_judge import (
    DEFAULT_JUDGE_MODEL,
    InvalidToneResponseError,
    _contrasts,
    build_sources,
    build_tone_request,
    validate_tone_response,
)
from thesis.judge.prompt import JudgeItem
from thesis.llm.cache import ResponseCache, cache_key
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import CACHE_DIR, INTERIM_DIR, MANIFESTS_DIR, ensure_dirs

log = get_logger(__name__)

POLITENESS_MODEL = "Genius1237/xlm-roberta-large-tydip"
MANIFEST_PATH = MANIFESTS_DIR / "politeness_250.json"
SCORES_PATH = INTERIM_DIR / "politeness_250.parquet"
# Most scores sit near 1, so the log-odds are fitted too: they spread out
# differences that the probability squeezes at the top.
OUTCOMES = ("politeness", "politeness_logit")
_LOGIT_EPS = 1e-6
_TONE_PROVIDER = "groq"


@cache
def _classifier() -> Any:
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=POLITENESS_MODEL,
        device="cpu",
        truncation=True,
        max_length=512,
        top_k=None,
    )


def politeness_scores(texts: list[str], *, batch_size: int = 8) -> list[float]:
    """Probability that each text is polite. Texts past 512 tokens are cut."""
    outputs = _classifier()(texts, batch_size=batch_size)
    return [next(s["score"] for s in scores if s["label"] == "polite") for scores in outputs]


def cached_tone_scores(items: list[JudgeItem], cache_store: ResponseCache) -> dict[str, int]:
    """The gpt-oss-120b tone score of each item that is in the cache."""
    scores = {}
    for item in items:
        request = build_tone_request(item, DEFAULT_JUDGE_MODEL)
        response = cache_store.get(cache_key(request, _TONE_PROVIDER))
        if response is None or response.parsed is None:
            continue
        try:
            scores[item.item_id] = validate_tone_response(response.parsed)
        except InvalidToneResponseError:
            continue
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-direction", type=int, default=250)
    parser.add_argument("--out", default=str(MANIFEST_PATH))
    parser.add_argument(
        "--scores",
        default=str(SCORES_PATH),
        help="Per-email scores. Reused if the file exists, so only the model fits rerun.",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()
    cache_store = ResponseCache(CACHE_DIR)
    sources, real_emails, grids = build_sources(args.per_direction)

    scores_path = Path(args.scores)
    if scores_path.exists():
        scores = pd.read_parquet(scores_path)
    else:
        parts = []
        for label, items in sources.items():
            part = pd.DataFrame(
                {
                    "source": label,
                    "item_id": [item.item_id for item in items],
                    "politeness": politeness_scores([item.text for item in items]),
                }
            )
            part["tone"] = part["item_id"].map(cached_tone_scores(items, cache_store))
            log.info("%s: scored %d", label, len(part))
            parts.append(part)
        scores = pd.concat(parts, ignore_index=True)
        scores.to_parquet(scores_path)
    clipped = scores["politeness"].clip(_LOGIT_EPS, 1 - _LOGIT_EPS)
    scores["politeness_logit"] = np.log(clipped / (1 - clipped))

    manifest: dict[str, Any] = {
        "model": POLITENESS_MODEL,
        "per_direction": args.per_direction,
        "scores_file": scores_path.name,
    }
    results: dict[str, Any] = {}
    for label in sources:
        frame = scores[scores["source"] == label]
        entry: dict[str, Any] = {"n": len(frame), "n_with_tone": int(frame["tone"].notna().sum())}
        if label == "real email":
            frame = frame.merge(real_emails, left_on="item_id", right_on="message_uid")
            for outcome in OUTCOMES:
                entry[outcome] = {
                    "mixed": _contrasts(
                        fit_direction_mixed_model(
                            frame,
                            outcome,
                            cluster_col="sender_id",
                            covariates=("sender_rank_level",),
                        )
                    ),
                    "fixed_effects": _contrasts(
                        fit_direction_fixed_effects(frame, outcome, cluster_col="sender_id")
                    ),
                }
            entry["cue_spearman"] = {
                cue: round(float(stats.spearmanr(frame["politeness"], frame[cue]).statistic), 4)
                for cue in ("imperative_ratio", "hedge_rate", "question_ratio", "deference_rate")
            }
        else:
            frame = frame.merge(grids[label], left_on="item_id", right_on="cell_id")
            for outcome in OUTCOMES:
                entry[outcome] = {
                    "fixed_effects": _contrasts(fit_direction_fixed_effects(frame, outcome))
                }
        with_tone = frame.dropna(subset=["tone"])
        entry["tone_spearman"] = round(
            float(stats.spearmanr(with_tone["politeness"], with_tone["tone"]).statistic), 4
        )
        entry["mean_by_direction"] = {
            str(k): round(float(v), 4)
            for k, v in frame.groupby("direction")["politeness"].mean().items()
        }
        results[label] = entry
        log.info("%s: %s", label, entry)
    manifest["results"] = results
    Path(args.out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
