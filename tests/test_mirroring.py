"""Tests for the mirroring measure.

The spaCy model load is real and expensive per-process, so it is loaded once at
module scope, the same way test_features.py does it.

What is tested is what the measure claims: that a reply built out of the
sender's own words scores high and an ordinary reply on the same topic does
not, that requests are recognised without treating every question as one, and
that the validation against the hand codes is computed the right way round.
"""

from __future__ import annotations

import pandas as pd
import pytest

from thesis.analysis.mirroring import (
    SIGNALS,
    contains_request,
    features_for,
    load_nlp,
    longest_shared_run,
    score_texts,
    validate,
)

nlp = load_nlp()

STIMULUS = (
    "Genia, could you get the most current swap form from Tana and check it "
    "against the one we use for EEI? I would think they should be the same."
)


def measure(reply: str, stimulus: str = STIMULUS):  # type: ignore[no-untyped-def]
    return features_for(nlp(stimulus), nlp(reply))


# ------------------------------------------------------------ shared runs


def test_longest_shared_run_finds_the_repeated_stretch() -> None:
    assert longest_shared_run(["a", "b", "c", "d"], ["x", "b", "c", "y"]) == 2


def test_longest_shared_run_needs_the_words_adjacent() -> None:
    """Two words shared but not in sequence is not a repeated phrase."""
    assert longest_shared_run(["a", "b"], ["b", "z", "a"]) == 1


def test_longest_shared_run_on_empty_input() -> None:
    assert longest_shared_run([], ["a"]) == 0
    assert longest_shared_run(["a"], []) == 0


# --------------------------------------------------------------- requests


def test_imperative_counts_as_a_request() -> None:
    assert contains_request(nlp("Send me the signed copy."))


def test_polite_form_counts_as_a_request() -> None:
    assert contains_request(nlp("Could you send me the signed copy?"))
    assert contains_request(nlp("Please review the attached draft."))


def test_a_question_about_facts_is_not_a_request() -> None:
    """Asking for information is a perfectly good reply; only asks-of-you count."""
    assert not contains_request(nlp("Did the deal close on Friday?"))


def test_a_plain_statement_is_not_a_request() -> None:
    assert not contains_request(nlp("I sent the signed copy this morning."))


# --------------------------------------------------------------- measuring


def test_a_reply_that_hands_the_request_back_scores_high() -> None:
    handed_back = measure(
        "Can you get the most current swap form from Tana and check it against the EEI form?"
    )
    assert handed_back.borrowed_words > 0.8
    assert handed_back.returned_request > 0.8


def test_an_ordinary_reply_on_the_same_topic_scores_lower() -> None:
    """Topical overlap is not mirroring, or every on-topic reply would fail."""
    handed_back = measure(
        "Can you get the most current swap form from Tana and check it against the EEI form?"
    )
    ordinary = measure(
        "I checked with Tana this morning. The two forms differ in the indemnity clause, "
        "so I have asked outside counsel to look at it before we circulate anything."
    )
    assert ordinary.borrowed_words < handed_back.borrowed_words


def test_a_reply_that_answers_rather_than_asks_is_not_a_returned_request() -> None:
    """Both sides have to be asking, or an answer that reuses the words gets flagged."""
    answered = measure("Yes, the swap form and the EEI form are the same.")
    assert not answered.reply_is_request
    assert answered.returned_request == 0.0


def test_verbatim_repetition_shows_up_as_a_repeated_phrase() -> None:
    assert measure("get the most current swap form from Tana").longest_repeat > 0.8


def test_an_empty_reply_scores_zero_rather_than_raising() -> None:
    empty = measure("")
    assert empty.borrowed_words == 0.0
    assert empty.longest_repeat == 0.0


def test_score_texts_returns_one_row_per_pair() -> None:
    scored = score_texts([STIMULUS, STIMULUS], ["Will do.", "Can you check it?"], nlp=nlp)
    assert len(scored) == 2
    assert set(SIGNALS) <= set(scored.columns)


def test_score_texts_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        score_texts([STIMULUS], ["a", "b"], nlp=nlp)


# -------------------------------------------------------------- validation


def test_validate_scores_a_perfect_signal_at_one() -> None:
    scored = pd.DataFrame(
        {signal: [0.9, 0.9, 0.1, 0.1] for signal in SIGNALS},
    )
    assert validate(scored, [True, True, False, False])["borrowed_words"] == 1.0


def test_validate_scores_a_signal_pointing_the_wrong_way_below_chance() -> None:
    """A signal that ranks mirrored replies lowest must not be reported as good."""
    scored = pd.DataFrame({signal: [0.1, 0.1, 0.9, 0.9] for signal in SIGNALS})
    assert validate(scored, [True, True, False, False])["borrowed_words"] == 0.0
