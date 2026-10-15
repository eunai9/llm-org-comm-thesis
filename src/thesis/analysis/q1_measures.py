"""Candidate Q1 hierarchy measures for the parallel persona-variant experiment.

Each measure is tested on real Enron email first; one that shows no
hierarchy difference there is dropped before it is used on simulator output.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
import spacy
from spacy.tokens import Doc, Span, Token
from statsmodels.stats.multitest import multipletests

from thesis.analysis.hierarchy import (
    FixedEffectsResult,
    MixedModelResult,
    fit_direction_fixed_effects,
    fit_direction_mixed_model,
)
from thesis.analysis.q1_real import EMAILS_PATH, THREADS_PATH, load_bodies
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import EXTERNAL_DIR, MANIFESTS_DIR

log = get_logger(__name__)

REAL_MANIFEST_PATH = MANIFESTS_DIR / "q1_measures_real.json"
PRONOUN_OUTCOMES: Final[tuple[str, ...]] = ("i_rate", "we_rate", "you_rate")
POLITENESS_OUTCOMES: Final[tuple[str, ...]] = ("gratitude", "apology", "please", "greeting")
GILBERT_PATH = EXTERNAL_DIR / "gilbert2012_hier_phrases.txt"
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


# --- Measure 2: Gilbert (2012) hierarchy phrases ----------------------------

_GILBERT_TOKEN = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


def gilbert_tokens(text: str) -> list[str]:
    """Tokens in the form Gilbert's phrase list uses: lowercase, no apostrophes,
    punctuation as separate tokens."""
    return _GILBERT_TOKEN.findall(text.lower().replace("'", "").replace("\u2019", ""))


def load_gilbert_phrases(path: Path) -> dict[tuple[str, ...], float]:
    """Nonzero phrase weights from Gilbert's list. Positive means upward.

    Sender fixed effects are not phrases and are skipped, as are lines whose
    weight does not parse as a number.
    """
    phrases: dict[tuple[str, ...], float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        phrase, _, weight = line.rpartition("\t")
        if not phrase or phrase.startswith("sender_"):
            continue
        try:
            beta = float(weight)
        except ValueError:
            continue
        if beta != 0.0:
            phrases[tuple(phrase.split())] = beta
    return phrases


def gilbert_score(tokens: Sequence[str], phrases: Mapping[tuple[str, ...], float]) -> float:
    """Sum of the weights of every phrase occurrence in ``tokens``."""
    lengths = {len(phrase) for phrase in phrases}
    score = 0.0
    for n in lengths:
        for i in range(len(tokens) - n + 1):
            score += phrases.get(tuple(tokens[i : i + n]), 0.0)
    return score


# --- Measure 3: politeness strategies ----------------------------------------
# Rules follow ConvoKit's politeness_api strategies (Danescu-Niculescu-Mizil et
# al. 2013), applied per sentence on a spaCy parse.

_GRATITUDE: Final[frozenset[str]] = frozenset({"thank", "thanks"})
_APOLOGY: Final[frozenset[str]] = frozenset({"sorry", "woops", "oops"})
_GREETING: Final[frozenset[str]] = frozenset({"hi", "hello", "hey"})


@dataclass(frozen=True, slots=True)
class PolitenessMarkers:
    """Whether any sentence of one text uses each strategy."""

    gratitude: bool
    apology: bool
    please: bool
    greeting: bool


def _is_gratitude(sent: Span) -> bool:
    return any(
        t.lower_ in _GRATITUDE or (t.lower_ == "i" and t.head.lower_ == "appreciate") for t in sent
    )


def _is_apology(sent: Span) -> bool:
    return any(
        t.lower_ in _APOLOGY
        or (t.lower_ == "i" and t.dep_ == "nsubj" and t.head.lower_ == "apologize")
        or (t.lower_ == "me" and t.dep_ == "dobj" and t.head.lower_ in {"forgive", "excuse"})
        for t in sent
    )


def _first_word(sent: Span) -> str:
    words = [t for t in sent if not t.is_space and not t.is_punct]
    return words[0].lower_ if words else ""


def politeness_markers(doc: Doc) -> PolitenessMarkers:
    """Gratitude, apology, please and greeting, each true if any sentence has it.

    "Please" counts anywhere in a sentence, merging ConvoKit's "Please" and
    "Please start". A greeting counts only as a sentence's first word.
    """
    sents = list(doc.sents)
    return PolitenessMarkers(
        gratitude=any(_is_gratitude(s) for s in sents),
        apology=any(_is_apology(s) for s in sents),
        please=any(t.lower_ == "please" for t in doc),
        greeting=any(_first_word(s) in _GREETING for s in sents),
    )


# --- Measure 4: language style matching ---------------------------------------
# The nine LSM categories (Ireland & Pennebaker 2010). LIWC's word lists are
# licensed, so categories come from spaCy part-of-speech tags plus small
# closed word lists. Each word goes into at most one category, checked in the
# order below.

LSM_CATEGORIES: Final[tuple[str, ...]] = (
    "negations",
    "articles",
    "quantifiers",
    "personal_pronouns",
    "impersonal_pronouns",
    "auxiliary_verbs",
    "adverbs",
    "prepositions",
    "conjunctions",
)

_NEGATIONS: Final[frozenset[str]] = frozenset(
    {"no", "not", "n't", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere"}
)
_ARTICLES: Final[frozenset[str]] = frozenset({"a", "an", "the"})
_QUANTIFIERS: Final[frozenset[str]] = frozenset(
    {
        "all",
        "any",
        "some",
        "many",
        "much",
        "few",
        "several",
        "more",
        "most",
        "less",
        "least",
        "each",
        "every",
        "both",
        "lot",
        "lots",
        "plenty",
        "enough",
        "little",
    }
)
_PERSONAL_PRONOUNS: Final[frozenset[str]] = frozenset(
    {
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "we",
        "us",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
    }
)


def _lsm_category(token: Token) -> str | None:
    word = token.lower_
    if word in _NEGATIONS or token.dep_ == "neg":
        return "negations"
    if word in _ARTICLES:
        return "articles"
    if word in _QUANTIFIERS:
        return "quantifiers"
    if word in _PERSONAL_PRONOUNS:
        return "personal_pronouns"
    if token.pos_ == "PRON":
        return "impersonal_pronouns"
    if token.pos_ == "AUX":
        return "auxiliary_verbs"
    if token.pos_ == "ADV":
        return "adverbs"
    if token.pos_ == "ADP":
        return "prepositions"
    if token.pos_ in {"CCONJ", "SCONJ"}:
        return "conjunctions"
    return None


def function_word_shares(doc: Doc) -> dict[str, float]:
    """Share of words in each LSM category. Punctuation and spaces are not words."""
    words = [t for t in doc if not t.is_space and not t.is_punct]
    counts = dict.fromkeys(LSM_CATEGORIES, 0)
    for token in words:
        category = _lsm_category(token)
        if category is not None:
            counts[category] += 1
    n = len(words)
    return {c: (counts[c] / n if n else 0.0) for c in LSM_CATEGORIES}


def lsm_from_shares(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Mean over categories of 1 - |a - b| / (a + b + 0.0001), shares in percent."""
    scores = [1 - abs(a[c] - b[c]) / (a[c] + b[c] + 0.0001) for c in LSM_CATEGORIES]
    return sum(scores) / len(scores)


def style_matching(reply: Doc, incoming: Doc) -> float:
    """Language Style Matching between a reply and the email it answers."""
    return lsm_from_shares(
        _percent(function_word_shares(reply)), _percent(function_word_shares(incoming))
    )


def _percent(shares: Mapping[str, float]) -> dict[str, float]:
    return {c: 100 * v for c, v in shares.items()}


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


def parent_uids(threads: pd.DataFrame, uids: Sequence[str]) -> dict[str, str]:
    """For each message, the one just before it in the same thread.

    Messages that start a thread, or are not in ``threads``, are left out.
    """
    keyed = threads.set_index(["thread_id", "position_in_thread"])["message_uid"]
    wanted = threads[threads["message_uid"].isin(set(uids))]
    parents: dict[str, str] = {}
    for uid, thread, position in zip(
        wanted["message_uid"], wanted["thread_id"], wanted["position_in_thread"], strict=True
    ):
        parent = keyed.get((thread, position - 1))
        if parent is not None:
            parents[str(uid)] = str(parent)
    return parents


def _with_holm(table: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjusted p-values within each fit of one measure family."""
    table = table.copy()
    for _, part in table.groupby("fit"):
        table.loc[part.index, "p_holm"] = multipletests(part["p_value"], method="holm")[1]
    return table


def _family_result(frame: pd.DataFrame, outcomes: Sequence[str]) -> dict[str, Any]:
    table = _with_holm(direction_contrasts(frame, outcomes))
    means = frame.groupby("direction")[list(outcomes)].mean()
    print(means.round(4).to_string())
    print(table.round(4).to_string(index=False))
    print()
    return {
        "n_emails": len(frame),
        "n_senders": int(frame[SENDER_COL].nunique()),
        "means_by_direction": means.round(5).to_dict(),
        "contrasts": table.round(5).to_dict(orient="records"),
    }


def main() -> None:
    """Test every candidate measure on the strict real-email benchmark sample."""
    configure_logging()
    emails = pd.read_parquet(EMAILS_PATH)
    emails = emails[emails["sample"] == "strict"].reset_index(drop=True)
    bodies = load_bodies(emails["message_uid"]).set_index("message_uid")["body_clean"]
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    docs = dict(zip(bodies.index, nlp.pipe(bodies.tolist(), batch_size=128), strict=True))
    phrases = load_gilbert_phrases(GILBERT_PATH)
    log.info("parsed %d strict-sample emails; %d Gilbert phrases", len(docs), len(phrases))

    rows = []
    for uid, body in bodies.items():
        tokens = gilbert_tokens(body)
        rows.append(
            {
                "message_uid": uid,
                **asdict(pronoun_rates(body)),
                "gilbert_score": 100 * gilbert_score(tokens, phrases) / max(len(tokens), 1),
                **{k: float(v) for k, v in asdict(politeness_markers(docs[uid])).items()},
            }
        )
    frame = emails.merge(pd.DataFrame(rows), on="message_uid", how="inner", validate="one_to_one")

    threads = pd.read_parquet(
        THREADS_PATH, columns=["message_uid", "thread_id", "position_in_thread"]
    )
    replies = frame[frame["is_reply"]]
    parents = parent_uids(threads, replies["message_uid"].tolist())
    parent_bodies = load_bodies(parents.values()).set_index("message_uid")["body_clean"]
    parent_docs = dict(
        zip(parent_bodies.index, nlp.pipe(parent_bodies.tolist(), batch_size=128), strict=True)
    )
    lsm = pd.DataFrame(
        [
            {"message_uid": uid, "style_matching": style_matching(docs[uid], parent_docs[parent])}
            for uid, parent in parents.items()
            if parent in parent_docs
        ]
    )
    reply_frame = replies.merge(lsm, on="message_uid", how="inner", validate="one_to_one")
    log.info("style matching for %d of %d replies", len(reply_frame), len(replies))

    manifest: dict[str, Any] = {"sample": "strict"}
    for name, data, outcomes in (
        ("pronouns", frame, PRONOUN_OUTCOMES),
        ("hierarchy_phrases", frame, ("gilbert_score",)),
        ("politeness", frame, POLITENESS_OUTCOMES),
        ("style_matching", reply_frame, ("style_matching",)),
    ):
        print(f"== {name} ==")
        manifest[name] = _family_result(data, outcomes)

    REAL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %s", REAL_MANIFEST_PATH)


if __name__ == "__main__":
    main()
