"""Layer A of the power score: rule-based linguistic features, every message.

Deterministic and cheap enough to run on the whole corpus -- no LLM calls, no
network dependency. Five markers, each a per-sentence rate so message length
doesn't confound the signal:

- **Directives**: sentence-initial imperative mood ("Send me the report."),
  identified structurally via spaCy's dependency parse (a root verb with no
  nominal subject), plus obligation-modal sentences ("You must...",
  "Please..."). Directive language is the clearest linguistic signature of
  someone issuing instructions rather than making requests.
- **Hedges**: epistemic softeners ("perhaps", "I think", "sort of") that
  lower the speaker's commitment to a claim -- the classic marker of
  deference in politeness research (Brown & Levinson 1987; Lakoff 1973).
- **Deference markers**: explicit politeness/subordination phrases ("would
  you mind", "if possible", "sorry to bother you").
- **Commitments**: first-person commissives ("I will send...", "I'll
  handle...") -- promising future action, associated with accountability
  rather than authority.
- **Questions**: interrogative sentences, as a share of all sentences.

**Scope cut, named rather than discovered:** the plan called for Convokit's
published politeness-strategies feature set here. Its install stalled on
this environment (a heavy, sometimes-conflicting dependency tree: nltk
corpus downloads, scikit-learn, a pinned spaCy version) and was abandoned
rather than debugged at length, since a hand-curated lexicon serves the same
purpose for a bounded set of markers. This is a real trade: Convokit's
feature set is peer-reviewed and directly citable (Danescu-Niculescu-Mizil et
al. 2013); this lexicon is not, and its category boundaries are a judgment
call that should be stated as a limitation, not implied to be automatic. The
lexicons below draw on the same politeness-theory literature Convokit itself
implements (Brown & Levinson 1987), so the *constructs* are grounded even
though the *implementation* is not the published one.

Run with ``python -m thesis.data.features``.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import INTERIM_DIR, MESSAGES_PARQUET_GLOB, ensure_dirs

log = get_logger(__name__)

_SPACY_BATCH_SIZE = 256
_SPACY_N_PROCESS = 4

# A tiny number of messages in this corpus are not correspondence at all --
# pasted documents, log dumps, or attachments that ended up in the body text
# -- and run past a million characters (the largest is 1.7M). spaCy's parser
# needs roughly 1GB of temporary memory per 100,000 characters, so a message
# that size alone can exceed the machine's entire RAM budget; this is what
# caused several apparent "hangs" during development (a worker thrashing on
# memory allocation looks identical to a deadlock from the outside, until it
# either OOMs or spaCy's own nlp.max_length safety check raises). Filtering
# these out in SQL, before spaCy ever sees them, is cheap and drops only 46
# of 237,673 messages (0.02%) -- corpus noise, not lost business
# correspondence, all of which sits under a few thousand characters.
MAX_BODY_CHARS = 100_000

# --------------------------------------------------------------------------
# Lexicons. Multi-word entries are matched as substrings on the lower-cased
# sentence text; single tokens are matched against the lemmatized token set,
# which catches inflection ("hedges", "hedging") for free.
# --------------------------------------------------------------------------

_HEDGE_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "perhaps",
        "maybe",
        "possibly",
        "probably",
        "somewhat",
        "presumably",
        "arguably",
        "apparently",
        "seemingly",
        "roughly",
    }
)
_HEDGE_PHRASES: Final[tuple[str, ...]] = (
    "i think",
    "i guess",
    "i believe",
    "i suppose",
    "i assume",
    "it seems",
    "it seems like",
    "sort of",
    "kind of",
    "i'm not sure",
    "im not sure",
    "not certain",
    "correct me if i'm wrong",
    "as far as i know",
    "i could be wrong",
)

_DEFERENCE_PHRASES: Final[tuple[str, ...]] = (
    "would you mind",
    "if you don't mind",
    "if you dont mind",
    "if possible",
    "if it's not too much trouble",
    "when you get a chance",
    "when you have a chance",
    "no rush",
    "no worries if not",
    "sorry to bother",
    "sorry for the trouble",
    "at your convenience",
    "would it be possible",
    "i was wondering if",
    "i hope this is ok",
    "i hope that's ok",
    "please let me know if",
    "thanks in advance",
    "thank you in advance",
)

_COMMITMENT_PHRASES: Final[tuple[str, ...]] = (
    "i will",
    "i'll",
    "i promise",
    "i plan to",
    "i intend to",
    "i'm going to",
    "im going to",
    "we will",
    "we'll",
    "i shall",
    "i commit to",
)
# NOTE: an apostrophe-stripped "ill " variant was considered and dropped --
# substring matching "ill " also fires inside "will ", "still ", "chill ",
# and any other word ending "-ill" before a space, which is a real false
# positive rate, not a hypothetical one (caught by a smoke test: a sentence
# containing "still on for Friday" scored a nonzero commitment rate for no
# defensible reason). No evidence apostrophes are actually stripped in this
# corpus was gathered before adding it, so it is removed rather than patched
# with a word-boundary regex for a scenario that was never confirmed.

_OBLIGATION_MODALS: Final[frozenset[str]] = frozenset(
    {"must", "should", "shall", "need", "needs", "required"}
)
_OBLIGATION_PHRASES: Final[tuple[str, ...]] = ("have to", "has to", "need to", "needs to")

_deadline_pattern = re.compile(
    r"\b(deadline|due\s+by|no\s+later\s+than|by\s+(?:end\s+of\s+day|eod|cob|"
    r"monday|tuesday|wednesday|friday|thursday|saturday|sunday|tomorrow|"
    r"\d{1,2}[/-]\d{1,2})|\basap\b|\burgent\b)",
    re.IGNORECASE,
)


def _load_nlp() -> Language:
    """Load spaCy with only the components these features need.

    The tagger and parser are required for imperative detection (root verb,
    no nominal subject) and sentence boundaries. NER and the lexeme-based
    lemmatizer are disabled -- neither is used, and disabling them roughly
    doubles throughput, which matters at 254k messages.
    """
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def is_imperative(sent: Span) -> bool:
    """A sentence-initial root verb with no nominal subject: "Send the report."

    This is the standard structural definition of an English imperative --
    distinguishing it from a declarative requires the dependency parse, not
    just the presence of a verb, since "Send the report." and "He sends the
    report." share the same first token type.
    """
    root = sent.root
    if root.pos_ not in ("VERB", "AUX"):
        return False
    if root.tag_ != "VB":  # base form; rules out "Sending", "Sent", "Sends"
        return False
    return not any(child.dep_ in ("nsubj", "nsubjpass", "expl") for child in root.children)


def _has_obligation_modal(sent: Span) -> bool:
    lowered = sent.text.lower()
    if any(phrase in lowered for phrase in _OBLIGATION_PHRASES):
        return True
    return any(token.lower_ in _OBLIGATION_MODALS for token in sent)


def _count_phrase_hits(text_lower: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for phrase in phrases if phrase in text_lower)


@dataclass(frozen=True, slots=True)
class MessageFeatures:
    """Layer A features for one message. Rates are per sentence."""

    message_uid: str
    n_sentences: int
    n_tokens: int
    mean_sentence_len: float
    question_ratio: float
    imperative_ratio: float
    hedge_rate: float
    deference_rate: float
    commitment_rate: float
    has_deadline: bool


def extract_features(message_uid: str, doc: Doc) -> MessageFeatures:
    """Compute Layer A features from a parsed document."""
    sentences = list(doc.sents)
    n_sentences = len(sentences)
    n_tokens = sum(1 for token in doc if not token.is_space)

    if n_sentences == 0:
        return MessageFeatures(
            message_uid=message_uid,
            n_sentences=0,
            n_tokens=n_tokens,
            mean_sentence_len=0.0,
            question_ratio=0.0,
            imperative_ratio=0.0,
            hedge_rate=0.0,
            deference_rate=0.0,
            commitment_rate=0.0,
            has_deadline=bool(_deadline_pattern.search(doc.text)),
        )

    n_questions = 0
    n_imperatives = 0
    n_hedges = 0
    n_deference = 0
    n_commitments = 0
    sentence_lengths: list[int] = []

    for sent in sentences:
        text_lower = sent.text.lower()
        tokens_in_sent = [t for t in sent if not t.is_space and not t.is_punct]
        sentence_lengths.append(len(tokens_in_sent))

        if sent.text.rstrip().endswith("?"):
            n_questions += 1
        if is_imperative(sent) or _has_obligation_modal(sent):
            n_imperatives += 1

        hedge_hits = _count_phrase_hits(text_lower, _HEDGE_PHRASES)
        hedge_hits += sum(1 for token in sent if token.lower_ in _HEDGE_TOKENS)
        n_hedges += min(hedge_hits, 1)  # cap at one per sentence: rate, not raw count

        n_deference += min(_count_phrase_hits(text_lower, _DEFERENCE_PHRASES), 1)
        n_commitments += min(_count_phrase_hits(text_lower, _COMMITMENT_PHRASES), 1)

    return MessageFeatures(
        message_uid=message_uid,
        n_sentences=n_sentences,
        n_tokens=n_tokens,
        mean_sentence_len=sum(sentence_lengths) / n_sentences,
        question_ratio=n_questions / n_sentences,
        imperative_ratio=n_imperatives / n_sentences,
        hedge_rate=n_hedges / n_sentences,
        deference_rate=n_deference / n_sentences,
        commitment_rate=n_commitments / n_sentences,
        has_deadline=bool(_deadline_pattern.search(doc.text)),
    )


@dataclass(frozen=True, slots=True)
class SentenceFeatures:
    """One sentence's markers, unaggregated.

    ``extract_features`` divides by ``n_sentences`` to get a per-message
    rate, which is the right unit for a normal multi-sentence email but
    breaks down for very short replies: a rate computed over one or two
    sentences can only take a handful of values (0, 0.5, 1 for two
    sentences), coarse enough to swamp a real effect the size this project
    is looking for. This function exists for exactly that case -- when the
    question is about the sentence itself, not a ratio derived from it, use
    these rows directly (one binary observation per sentence) in a logistic
    model instead of a linear one over the collapsed rate. See
    :func:`thesis.analysis.hierarchy.fit_sentence_level_model`.
    """

    message_uid: str
    sentence_index: int
    is_imperative: bool
    is_question: bool
    has_hedge: bool
    has_deference: bool
    has_commitment: bool


def extract_sentence_features(message_uid: str, doc: Doc) -> list[SentenceFeatures]:
    """One row per sentence, using the exact same classification rules
    ``extract_features`` aggregates -- so a sentence-level analysis and the
    corpus-wide per-message rates it started from can never quietly
    disagree about what counts as imperative, hedged, or deferential.
    """
    rows = []
    for i, sent in enumerate(doc.sents):
        text_lower = sent.text.lower()
        hedge_hits = _count_phrase_hits(text_lower, _HEDGE_PHRASES)
        hedge_hits += sum(1 for token in sent if token.lower_ in _HEDGE_TOKENS)
        rows.append(
            SentenceFeatures(
                message_uid=message_uid,
                sentence_index=i,
                is_imperative=is_imperative(sent) or _has_obligation_modal(sent),
                is_question=sent.text.rstrip().endswith("?"),
                has_hedge=hedge_hits > 0,
                has_deference=_count_phrase_hits(text_lower, _DEFERENCE_PHRASES) > 0,
                has_commitment=_count_phrase_hits(text_lower, _COMMITMENT_PHRASES) > 0,
            )
        )
    return rows


FEATURES_SCHEMA = pa.schema(
    [
        pa.field("message_uid", pa.string(), nullable=False),
        pa.field("n_sentences", pa.int32(), nullable=False),
        pa.field("n_tokens", pa.int32(), nullable=False),
        pa.field("mean_sentence_len", pa.float32(), nullable=False),
        pa.field("question_ratio", pa.float32(), nullable=False),
        pa.field("imperative_ratio", pa.float32(), nullable=False),
        pa.field("hedge_rate", pa.float32(), nullable=False),
        pa.field("deference_rate", pa.float32(), nullable=False),
        pa.field("commitment_rate", pa.float32(), nullable=False),
        pa.field("has_deadline", pa.bool_(), nullable=False),
    ]
)


def _feature_row(features: MessageFeatures) -> dict[str, object]:
    return {
        "message_uid": features.message_uid,
        "n_sentences": features.n_sentences,
        "n_tokens": features.n_tokens,
        "mean_sentence_len": features.mean_sentence_len,
        "question_ratio": features.question_ratio,
        "imperative_ratio": features.imperative_ratio,
        "hedge_rate": features.hedge_rate,
        "deference_rate": features.deference_rate,
        "commitment_rate": features.commitment_rate,
        "has_deadline": features.has_deadline,
    }


def count_oversized_messages(messages_glob: str, max_chars: int = MAX_BODY_CHARS) -> int:
    """How many non-empty messages exceed the length cap -- for the report."""
    con = duckdb.connect()
    row = con.execute(
        "SELECT count(*) FROM read_parquet(?) "
        "WHERE NOT is_empty_after_clean AND length(body_clean) > ?",
        [messages_glob, max_chars],
    ).fetchone()
    con.close()
    assert row is not None
    return int(row[0])


def iter_message_texts(
    messages_glob: str, max_chars: int = MAX_BODY_CHARS
) -> Iterator[tuple[str, str]]:
    """Yield (message_uid, body_clean) for every non-empty message short
    enough for spaCy to parse safely.

    Empty-after-cleaning messages (forwards with no authored text, ~6.6% of
    the corpus) have nothing for these features to measure. Messages over
    ``max_chars`` are excluded for the memory reason documented on
    :data:`MAX_BODY_CHARS` -- both filters happen in SQL, so the oversized
    text is never even fetched, let alone handed to spaCy.

    Streams from DuckDB rather than calling ``.fetchall()`` -- at ~238k
    messages, materializing every row as a Python object up front measurably
    adds to peak memory in an environment already tight on RAM (WSL's default
    cap is well under what a 4-worker spaCy pipeline plus DuckDB plus this
    list would ask for at once; see run_extraction's chunking for the other
    half of this fix).
    """
    con = duckdb.connect()
    cursor = con.execute(
        "SELECT message_uid, body_clean FROM read_parquet(?) "
        "WHERE NOT is_empty_after_clean AND length(body_clean) <= ?",
        [messages_glob, max_chars],
    )
    while batch := cursor.fetchmany(2_000):
        yield from batch
    con.close()


def _iter_row_chunks(
    pairs: Iterator[tuple[str, str]], chunk_size: int
) -> Iterator[list[tuple[str, str]]]:
    chunk: list[tuple[str, str]] = []
    for pair in pairs:
        chunk.append(pair)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def run_extraction(
    messages_glob: str,
    out_path: object,
    *,
    batch_size: int = _SPACY_BATCH_SIZE,
    n_process: int = _SPACY_N_PROCESS,
    chunk_size: int = 10_000,
    limit: int | None = None,
) -> int:
    """Extract features for every message and write one Parquet file.

    Processes ``chunk_size`` messages at a time and writes each chunk as its
    own Parquet row-group, rather than holding parsed spaCy output for the
    entire corpus in memory before writing anything. Peak memory is bounded
    by one chunk, not by corpus size -- the earlier all-at-once version is
    what coincided with two WSL crashes during a 4-process run over the full
    ~238k messages.

    Returns the number of messages processed.
    """
    nlp = _load_nlp()
    pairs = iter_message_texts(messages_glob)

    writer: pq.ParquetWriter | None = None
    n_written = 0
    try:
        for chunk in _iter_row_chunks(pairs, chunk_size):
            if limit is not None and n_written >= limit:
                break
            if limit is not None:
                chunk = chunk[: limit - n_written]

            uids = [uid for uid, _ in chunk]
            texts = [text for _, text in chunk]
            docs = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
            rows = [
                _feature_row(extract_features(uid, doc))
                for uid, doc in zip(uids, docs, strict=True)
            ]

            table = pa.Table.from_pylist(rows, schema=FEATURES_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(out_path, FEATURES_SCHEMA, compression="zstd")
            writer.write_table(table)

            n_written += len(rows)
            log.info("processed %d messages", n_written)
    finally:
        if writer is not None:
            writer.close()

    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default=MESSAGES_PARQUET_GLOB)
    parser.add_argument("--out", default=str(INTERIM_DIR / "features.parquet"))
    parser.add_argument("--limit", type=int, default=None, help="For smoke tests.")
    parser.add_argument("--n-process", type=int, default=_SPACY_N_PROCESS)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Messages per write chunk. Smaller values give more frequent "
        "progress log lines, at the cost of more (smaller) Parquet row groups.",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    n_oversized = count_oversized_messages(args.messages)
    if n_oversized:
        log.info(
            "excluding %d message(s) over %d chars (see MAX_BODY_CHARS)",
            n_oversized,
            MAX_BODY_CHARS,
        )
    n = run_extraction(
        args.messages,
        args.out,
        n_process=args.n_process,
        chunk_size=args.chunk_size,
        limit=args.limit,
    )
    log.info("wrote %d rows to %s", n, args.out)


if __name__ == "__main__":
    main()
