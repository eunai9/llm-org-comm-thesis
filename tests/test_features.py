"""Layer A linguistic feature tests.

spaCy model load is real (not mocked) and expensive per-process, so it is
loaded once at module scope and shared across every test in this file.
"""

from __future__ import annotations

import pytest

from thesis.data.features import _load_nlp, extract_features

nlp = _load_nlp()


def features(text: str, uid: str = "t"):  # type: ignore[no-untyped-def]
    return extract_features(uid, nlp(text))


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
