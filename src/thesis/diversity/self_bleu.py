"""Q2: how diverse is the wording of generated replies? Self-BLEU.

Self-BLEU scores each text by BLEU against all the other texts of its set as
references, then averages. High means the texts reuse the same n-grams, so a
higher score means less diverse wording.

Two comparisons:

- **Against real email, same messages.** Each pairs file holds a real reply
  and a generated reply to the same incoming message, so both sets cover the
  same topics. Self-BLEU is computed within each side and compared pair by
  pair (Wilcoxon signed-rank). It is also computed with each real reply cut to
  its generated partner's word count, as in :mod:`thesis.analysis.embedding_map`,
  because longer texts share more n-grams by chance.
- **Within one scenario, across personas.** Each Q1 grid has 144 scenarios
  with 10 replies each. Each reply is scored against the other replies to the
  same scenario. This asks whether a different writer persona changes the
  wording. Real email has no such set, so this compares models only.

Run with ``python -m thesis.diversity.self_bleu``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import sacrebleu
from scipy import stats

from thesis.analysis.embedding_map import truncate_words
from thesis.analysis.role_inference import GENERATING_GRIDS
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import INTERIM_DIR, MANIFESTS_DIR, ensure_dirs

log = get_logger(__name__)

PAIRS_FILES: Final[dict[str, str]] = {
    "llama3.2:3b": "real_vs_generated_pairs_act.parquet",
    "gpt-oss-20b": "pairs_gpt_oss.parquet",
    "gpt-oss-120b": "pairs_gpt_oss_120b.parquet",
    "qwen3.8-27b": "pairs_qwen.parquet",
}
MANIFEST_PATH = MANIFESTS_DIR / "self_bleu.json"
SEED = 20261103
N_BOOTSTRAP = 2000


def self_bleu_scores(texts: Sequence[str]) -> np.ndarray:
    """BLEU of each text against all the others, on a 0-1 scale."""
    return np.array(
        [
            sacrebleu.sentence_bleu(text, [*texts[:i], *texts[i + 1 :]]).score / 100
            for i, text in enumerate(texts)
        ]
    )


def bootstrap_interval(values: np.ndarray, *, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling texts."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(N_BOOTSTRAP, len(values))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def _summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": round(float(values.mean()), 4),
        "ci95": bootstrap_interval(values),
        "n": len(values),
    }


def compare_with_real(frame: pd.DataFrame) -> dict[str, Any]:
    """Self-BLEU of real and generated replies to the same messages."""
    generated = list(frame["generated_reply"])
    real = list(frame["real_reply_body_recleaned"])
    views = {
        "as stored": real,
        "length-matched": [
            truncate_words(r, len(g.split())) for r, g in zip(real, generated, strict=True)
        ],
    }
    generated_scores = self_bleu_scores(generated)
    out: dict[str, Any] = {"generated": _summary(generated_scores)}
    for view, real_texts in views.items():
        real_scores = self_bleu_scores(real_texts)
        test = stats.wilcoxon(generated_scores, real_scores)
        out[view] = {
            "real": _summary(real_scores),
            "difference_generated_minus_real": round(
                float((generated_scores - real_scores).mean()), 4
            ),
            "wilcoxon_p": round(float(test.pvalue), 6),
        }
    return out


def within_scenario(grid: pd.DataFrame) -> dict[str, Any]:
    """Self-BLEU of each reply against the other replies to the same scenario."""
    scores = np.concatenate(
        [self_bleu_scores(list(group["body"])) for _, group in grid.groupby("scenario_id")]
    )
    return {**_summary(scores), "n_scenarios": int(grid["scenario_id"].nunique())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(MANIFEST_PATH))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    manifest: dict[str, Any] = {"seed": SEED, "n_bootstrap": N_BOOTSTRAP, "versus_real": {}}
    for label, name in PAIRS_FILES.items():
        manifest["versus_real"][label] = {
            "pairs_file": name,
            **compare_with_real(pd.read_parquet(INTERIM_DIR / name)),
        }
        log.info("%s versus real: %s", label, manifest["versus_real"][label])

    manifest["within_scenario"] = {}
    for label, path in GENERATING_GRIDS.items():
        grid = pd.read_parquet(path)
        grid = grid[grid["replicate"] == 1]
        manifest["within_scenario"][label] = {"grid_file": path.name, **within_scenario(grid)}
        log.info("%s within scenario: %s", label, manifest["within_scenario"][label])

    Path(args.out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
