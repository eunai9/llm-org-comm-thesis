"""Layer A linguistic feature tests.

spaCy model load is real (not mocked) and expensive per-process, so it is
loaded once at module scope and shared across every test in this file.
"""

from __future__ import annotations

import pytest

from thesis.data.features import _load_nlp, extract_features, extract_sentence_features

nlp = _load_nlp()


def features(text: str, uid: str = "t"):  # type: ignore[no-untyped-def]
    return extract_features(uid, nlp(text))


def sentence_features(text: str, uid: str = "t"):  # type: ignore[no-untyped-def]
    return extract_sentence_features(uid, nlp(text))


def test_empty_text_yields_zeroed_features() -> None:
    f = features("")
    assert f.n_sentences == 0
    assert f.question_ratio == 0.0
    assert f.imperative_ratio == 0.0
    assert not f.has_deadline


def test_imperative_sentence_is_detected() -> None:
    f = features("Send me the report by Friday.")
    assert f.imperative_ratio == 1.0


def test_declarative_sentence_is_not_imperative() -> None:
    f = features("He sends the report every Friday.")
    assert f.imperative_ratio == 0.0


def test_obligation_modal_counts_as_directive() -> None:
    f = features("You must complete this today.")
    assert f.imperative_ratio == 1.0


def test_have_to_phrase_counts_as_directive() -> None:
    f = features("You have to finish this by noon.")
    assert f.imperative_ratio == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "I think maybe we could look at this.",
        "Perhaps we should reconsider.",
        "I'm not sure this is right.",
    ],
)
def test_hedge_phrases_are_detected(text: str) -> None:
    assert features(text).hedge_rate > 0.0


def test_plain_statement_has_no_hedge() -> None:
    f = features("The meeting is scheduled for Friday.")
    assert f.hedge_rate == 0.0


@pytest.mark.parametrize(
    "text",
    [
        "Would you mind sending this over?",
        "Sorry to bother you with this.",
        "Thanks in advance for your help.",
    ],
)
def test_deference_phrases_are_detected(text: str) -> None:
    assert features(text).deference_rate > 0.0


@pytest.mark.parametrize(
    "text",
    [
        "I will send the report tomorrow.",
        "I'll take care of it.",
        "We will handle this internally.",
    ],
)
def test_commitment_phrases_are_detected(text: str) -> None:
    assert features(text).commitment_rate > 0.0


def test_word_ending_in_ill_is_not_a_false_commitment() -> None:
    """Regression: 'still'/'will'/'bill' etc. must not trigger commitment.

    A prior version matched the substring "ill " to catch an apostrophe-
    stripped "I'll", which also fires inside any word ending "-ill" before a
    space. Caught by manual smoke testing before it reached real data.
    """
    for text in ("Are we still on for Friday?", "Did you read his will?", "Pay the bill."):
        assert features(text).commitment_rate == 0.0, text


def test_question_mark_is_detected() -> None:
    f = features("Can you send this over? Sure thing.")
    assert f.question_ratio == 0.5


def test_no_questions_gives_zero_ratio() -> None:
    f = features("The meeting went well. We discussed the numbers.")
    assert f.question_ratio == 0.0


@pytest.mark.parametrize(
    "text",
    [
        "This is due by EOD Friday.",
        "The deadline is tomorrow.",
        "Please send this by 3/15.",
        "This is urgent, please respond asap.",
    ],
)
def test_deadline_markers_are_detected(text: str) -> None:
    assert features(text).has_deadline


def test_no_deadline_marker_present() -> None:
    f = features("Let's catch up sometime soon.")
    assert not f.has_deadline


def test_mean_sentence_length_is_computed() -> None:
    f = features("Yes. This is a somewhat longer sentence with several words in it.")
    assert f.n_sentences == 2
    assert f.mean_sentence_len > 0.0


def test_rates_are_bounded_between_zero_and_one() -> None:
    """Rates are capped at one hit per sentence, so no rate can exceed 1.0."""
    f = features(
        "I think maybe perhaps I guess I'm not sure. "
        "Please if possible would you mind, sorry to bother, thanks in advance."
    )
    assert 0.0 <= f.hedge_rate <= 1.0
    assert 0.0 <= f.deference_rate <= 1.0


def test_multiple_sentences_average_correctly() -> None:
    """One imperative among four sentences gives a ratio of 0.25, not 1.0."""
    f = features("Send the report. The weather is nice. I agree. Thanks for your help.")
    assert f.n_sentences == 4
    assert f.imperative_ratio == pytest.approx(0.25)


def test_oversized_message_is_excluded_not_crashed(tmp_path):  # type: ignore[no-untyped-def]
    """A message longer than MAX_BODY_CHARS must be skipped in SQL, not handed
    to spaCy -- this is the fix for a real corpus outlier (a 1.7M-character
    message) that crashed extraction with spaCy's own nlp.max_length guard.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from thesis.data.features import (
        MAX_BODY_CHARS,
        count_oversized_messages,
        iter_message_texts,
    )

    schema = pa.schema(
        [
            pa.field("message_uid", pa.string()),
            pa.field("body_clean", pa.string()),
            pa.field("is_empty_after_clean", pa.bool_()),
        ]
    )
    rows = [
        {"message_uid": "short", "body_clean": "Send the report.", "is_empty_after_clean": False},
        {
            "message_uid": "huge",
            "body_clean": "x" * (MAX_BODY_CHARS + 1),
            "is_empty_after_clean": False,
        },
        {"message_uid": "empty", "body_clean": "", "is_empty_after_clean": True},
    ]
    out_dir = tmp_path / "messages"
    out_dir.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_dir / "part-00000.parquet")
    glob = str(out_dir / "*.parquet")

    assert count_oversized_messages(glob) == 1

    uids = [uid for uid, _ in iter_message_texts(glob)]
    assert uids == ["short"]


# ---------------------------------------------------------- sentence-level


def test_sentence_features_returns_one_row_per_sentence() -> None:
    rows = sentence_features("Send the report. Let me know if you have questions.")
    assert len(rows) == 2
    assert [r.sentence_index for r in rows] == [0, 1]


def test_sentence_features_agrees_with_the_aggregate_ratio() -> None:
    """The whole point of extract_sentence_features is to never quietly
    disagree with extract_features about what counts as imperative -- both
    are exercised against the same text and cross-checked directly."""
    text = "Send the report. He sends the report every Friday. Could you check this?"
    agg = features(text)
    rows = sentence_features(text)
    assert len(rows) == agg.n_sentences
    assert sum(r.is_imperative for r in rows) / len(rows) == pytest.approx(agg.imperative_ratio)
    assert sum(r.is_question for r in rows) / len(rows) == pytest.approx(agg.question_ratio)


def test_sentence_features_empty_text_returns_no_rows() -> None:
    assert sentence_features("") == []


def test_sentence_features_marks_the_imperative_sentence_only() -> None:
    rows = sentence_features("Send the report. He sends the report every Friday.")
    assert rows[0].is_imperative
    assert not rows[1].is_imperative


def test_sentence_features_detects_hedge_deference_and_commitment() -> None:
    rows = sentence_features(
        "Perhaps we could revisit this. Would you mind checking? I will follow up."
    )
    assert rows[0].has_hedge
    assert rows[1].has_deference
    assert rows[2].has_commitment
