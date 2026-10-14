"""Candidate Q1 hierarchy measures for the parallel persona-variant experiment.

Each measure is tested on real Enron email first; one that shows no
hierarchy difference there is dropped before it is used on simulator output.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Final

import pandas as pd
from statsmodels.stats.multitest import multipletests

from thesis.analysis.hierarchy import (
    FixedEffectsResult,
    MixedModelResult,
    fit_direction_fixed_effects,
    fit_direction_mixed_model,
)
from thesis.analysis.q1_real import EMAILS_PATH, load_bodies
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import MANIFESTS_DIR

log = get_logger(__name__)

REAL_MANIFEST_PATH = MANIFESTS_DIR / "q1_measures_real.json"
PRONOUN_OUTCOMES: Final[tuple[str, ...]] = ("i_rate", "we_rate", "you_rate")
SENDER_COL = "sender_id"
RANK_COL = "sender_rank_level"
REFERENCE = "lateral"
CONTRASTS: Final[tuple[str, ...]] = ("down", "up")

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# First-person singular, first-person plural and second-person forms, the three
# pronoun groups Kacewicz et al. (2014) relate to social standing.
_I_FORMS: Final[frozenset[str]] = frozenset({"i", "me", "my", "mine", "myself"})
_WE_FORMS: Final[frozenset[str]] = frozenset({"we", "us", "our", "ours", "ourselves"})
_YOU_FORMS: Final[frozenset[str]] = frozenset({"you", "your", "yours", "yourself", "yourselves"})


@dataclass(frozen=True, slots=True)
class PronounRates:
    """Each pronoun group as a share of all words in one text."""

    n_words: int
    i_rate: float
    we_rate: float
    you_rate: float


def pronoun_rates(text: str) -> PronounRates:
    """Share of words that are I-, we- and you-forms.

    A contraction ("I'll") is one word, counted by its pronoun part. An
    all-caps "US" is the country, not the pronoun.
    """
    words = _WORD.findall(text)
    if not words:
        return PronounRates(0, 0.0, 0.0, 0.0)
    counts = {"i": 0, "we": 0, "you": 0}
    for word in words:
        if word == "US":
            continue
        head = word.split("'")[0].lower()
        if head in _I_FORMS:
            counts["i"] += 1
        elif head in _WE_FORMS:
            counts["we"] += 1
        elif head in _YOU_FORMS:
            counts["you"] += 1
    n = len(words)
    return PronounRates(n, counts["i"] / n, counts["we"] / n, counts["you"] / n)


def direction_contrasts(emails: pd.DataFrame, outcomes: Sequence[str]) -> pd.DataFrame:
    """Down and up contrasts against lateral, for each outcome, from two fits.

    "mixed" is the real-email benchmark fit: random intercept per sender,
    sender rank as a control. "fixed_effects" uses sender dummies with
    sender-clustered errors, so it compares directions within one sender.
    """
    rows = []
    for outcome in outcomes:
        fits: dict[str, MixedModelResult | FixedEffectsResult] = {
            "mixed": fit_direction_mixed_model(
                emails, outcome, cluster_col=SENDER_COL, reference=REFERENCE, covariates=(RANK_COL,)
            ),
            "fixed_effects": fit_direction_fixed_effects(
                emails, outcome, cluster_col=SENDER_COL, reference=REFERENCE
            ),
        }
        for fit_name, result in fits.items():
            for level in CONTRASTS:
                coefficient, p_value = result.contrast(level)
                rows.append(
                    {
                        "outcome": outcome,
                        "fit": fit_name,
                        "level": level,
                        "coefficient": coefficient,
                        "p_value": p_value,
                        "n_emails": result.n_observations,
                        "n_senders": result.n_groups,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    """Test the pronoun measures on the strict real-email benchmark sample."""
    configure_logging()
    emails = pd.read_parquet(EMAILS_PATH)
    emails = emails[emails["sample"] == "strict"].reset_index(drop=True)
    bodies = load_bodies(emails["message_uid"])
    rates = pd.DataFrame(
        [
            {"message_uid": uid, **asdict(pronoun_rates(body))}
            for uid, body in zip(bodies["message_uid"], bodies["body_clean"], strict=True)
        ]
    )
    frame = emails.merge(rates, on="message_uid", how="inner", validate="one_to_one")
    log.info("measured %d strict-sample emails", len(frame))

    table = direction_contrasts(frame, PRONOUN_OUTCOMES)
    for _, part in table.groupby("fit"):
        table.loc[part.index, "p_holm"] = multipletests(part["p_value"], method="holm")[1]
    means = frame.groupby("direction")[list(PRONOUN_OUTCOMES)].mean()

    print("Mean rate by direction (share of words):")
    print(means.round(4).to_string())
    print()
    print(table.round(4).to_string(index=False))

    REAL_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "sample": "strict",
                "n_emails": len(frame),
                "means_by_direction": means.round(5).to_dict(),
                "contrasts": table.round(5).to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s", REAL_MANIFEST_PATH)


if __name__ == "__main__":
    main()
