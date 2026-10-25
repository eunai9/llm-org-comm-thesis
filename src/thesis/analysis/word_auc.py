"""Can a word-choice classifier tell a generated reply from a real one?

Three AUCs per pairs file, the table in ``PROGRESS_llms.md`` section 8:

- **Length only**: logistic regression on each reply's word count.
- **Full text**: TF-IDF plus logistic regression
  (:func:`thesis.analysis.fidelity.model_free_discrimination_auc`).
- **Same length**: the same, with each real reply cut to its generated
  partner's word count, so length cannot explain what is left.

**Signature and header lines are removed first, from both sides.** Real
Enron replies carry routing addresses, company names, the Houston street
address and phone lines. A classifier finds those instantly, and they are
not writing. The rule here replaces a one-off script whose code was not
kept; on the four models logged before, it reproduces the logged text AUCs
within 0.002 and the length-only AUCs within 0.009.

Run with ``python -m thesis.analysis.word_auc --pairs PATH --out PATH``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from thesis.analysis.fidelity import model_free_discrimination_auc
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import ensure_dirs

log = get_logger(__name__)

SEED = 20260807
N_FOLDS = 5

_HEADER = re.compile(r"^\s*(to|cc|bcc|subject|sent by|from)\s*:", re.I)
_ROUTING = re.compile(r"/HOU/|@ECT|@ENRON|/ENRON|/Corp/|/NA/|/EU/|/EES", re.I)
_COMPANY = re.compile(
    r"^\s*Enron[\w .,&'-]*?(Corp(oration)?|Inc|Company|America|Services|Ltd|L\.?P)?\.?\s*$", re.I
)
_ADDRESS = re.compile(r"1400 Smith|Houston,?\s+(Texas|TX)", re.I)
_PHONE = re.compile(
    r"^\s*(phone|fax|tel|cell|mobile|pager)\b|\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}", re.I
)


def keep_line(line: str) -> bool:
    """False for a header, routing, company-name, address, phone or address-list line."""
    tokens = line.split()
    if tokens and sum("@" in token for token in tokens) / len(tokens) > 0.5:
        return False
    if line.strip() and _COMPANY.match(line):
        return False
    return not any(pattern.search(line) for pattern in (_HEADER, _ROUTING, _ADDRESS, _PHONE))


def clean_lines(text: str) -> tuple[str, int]:
    """The text without removed lines, and how many lines were removed."""
    lines = str(text).splitlines()
    kept = [line for line in lines if keep_line(line)]
    return "\n".join(kept), len(lines) - len(kept)


def _word_count(text: str) -> int:
    return len(text.split())


def length_only_auc(real: list[str], generated: list[str]) -> float:
    """Cross-validated AUC of a classifier that sees only word counts."""
    counts = np.array([_word_count(text) for text in real + generated], dtype=float)
    labels = [0] * len(real) + [1] * len(generated)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_val_score(
        LogisticRegression(), counts.reshape(-1, 1), labels, cv=cv, scoring="roc_auc"
    )
    return float(np.mean(scores))


def word_auc_summary(pairs: pd.DataFrame) -> dict[str, Any]:
    """The three AUCs for one pairs file, plus what the line removal changed."""
    valid = pairs[pairs["generated_reply"].notna()]
    valid = valid[valid["generated_reply"].str.strip() != ""]
    real_column = (
        "real_reply_body_recleaned" if "real_reply_body_recleaned" in valid else "real_reply_body"
    )
    real_cleaned = [clean_lines(text) for text in valid[real_column]]
    generated_cleaned = [clean_lines(text) for text in valid["generated_reply"]]
    real = [text for text, _ in real_cleaned]
    generated = [text for text, _ in generated_cleaned]
    real_cut = [
        " ".join(text.split()[: _word_count(partner)])
        for text, partner in zip(real, generated, strict=True)
    ]
    return {
        "n_pairs": len(valid),
        "auc_length_only": round(length_only_auc(real, generated), 3),
        "auc_full_text": round(model_free_discrimination_auc(real, generated), 3),
        "auc_same_length": round(model_free_discrimination_auc(real_cut, generated), 3),
        "real_replies_changed": sum(removed > 0 for _, removed in real_cleaned),
        "real_lines_removed": sum(removed for _, removed in real_cleaned),
        "generated_replies_changed": sum(removed > 0 for _, removed in generated_cleaned),
        "mean_real_words": round(float(np.mean([_word_count(t) for t in real])), 1),
        "mean_generated_words": round(float(np.mean([_word_count(t) for t in generated])), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    summary = word_auc_summary(pd.read_parquet(args.pairs))
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log.info("word AUC summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
