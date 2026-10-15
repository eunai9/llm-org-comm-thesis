from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import spacy
from spacy.language import Language

from thesis.analysis.q1_measures import (
    LSM_CATEGORIES,
    PolitenessMarkers,
    function_word_shares,
    gilbert_score,
    gilbert_tokens,
    load_gilbert_phrases,
    lsm_from_shares,
    parent_uids,
    politeness_markers,
    pronoun_rates,
    style_matching,
)


def test_counts_each_pronoun_group_as_share_of_words() -> None:
    rates = pronoun_rates("I think we should send you my notes")
    assert rates.n_words == 8
    assert rates.i_rate == pytest.approx(2 / 8)
    assert rates.we_rate == pytest.approx(1 / 8)
    assert rates.you_rate == pytest.approx(1 / 8)


def test_contractions_count_as_one_word_and_keep_their_pronoun() -> None:
    rates = pronoun_rates("I'll call. We're late, you'd agree")
    assert rates.n_words == 6
    assert rates.i_rate == pytest.approx(1 / 6)
    assert rates.we_rate == pytest.approx(1 / 6)
    assert rates.you_rate == pytest.approx(1 / 6)


def test_uppercase_us_is_the_country_not_a_pronoun() -> None:
    rates = pronoun_rates("Send it to us before the US market opens")
    assert rates.we_rate == pytest.approx(1 / 9)


def test_empty_text_gives_zero_rates() -> None:
    rates = pronoun_rates("")
    assert (rates.n_words, rates.i_rate, rates.we_rate, rates.you_rate) == (0, 0.0, 0.0, 0.0)


def _synthetic_emails() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sender in range(30):
        for direction in ("down", "lateral", "up"):
            for _ in range(4):
                we = 0.05 + (0.04 if direction == "down" else 0.0) + rng.normal(0, 0.01)
                rows.append(
                    {
                        "sender_id": f"s{sender}",
                        "sender_rank_level": str(sender % 5 + 1),
                        "direction": direction,
                        "we_rate": we,
                    }
                )
    return pd.DataFrame(rows)


def test_direction_contrasts_recover_a_planted_writing_down_effect() -> None:
    from thesis.analysis.q1_measures import direction_contrasts

    table = direction_contrasts(_synthetic_emails(), ("we_rate",))
    down = table[(table["outcome"] == "we_rate") & (table["level"] == "down")].iloc[0]
    up = table[(table["outcome"] == "we_rate") & (table["level"] == "up")].iloc[0]
    assert down["coefficient"] == pytest.approx(0.04, abs=0.01)
    assert down["p_value"] < 0.001
    assert abs(up["coefficient"]) < 0.01
    assert set(table["fit"]) == {"mixed", "fixed_effects"}


@pytest.fixture(scope="module")
def nlp() -> Language:
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


# --- Measure 2: Gilbert (2012) hierarchy phrases -----------------------------


def test_gilbert_tokens_lowercase_drop_apostrophes_and_split_punctuation() -> None:
    assert gilbert_tokens("Let's meet, but THANKS!") == [
        "lets",
        "meet",
        ",",
        "but",
        "thanks",
        "!",
    ]


def test_load_gilbert_phrases_skips_senders_zero_and_unparseable(tmp_path: Path) -> None:
    path = tmp_path / "phrases.txt"
    path.write_text(
        "have you been\t-8.5\n"
        "sender_kitchen\t-6.0\n"
        "the ability to\t6.8\n"
        "zero phrase\t0\n"
        ", but lets\t-5.05e.05\n",
        encoding="utf-8",
    )
    assert load_gilbert_phrases(path) == {
        ("have", "you", "been"): -8.5,
        ("the", "ability", "to"): 6.8,
    }


def test_gilbert_score_sums_weights_of_every_matched_phrase() -> None:
    phrases = {("have", "you", "been"): -8.0, ("attach",): 6.0, ("you", "been"): 1.0}
    tokens = gilbert_tokens("Have you been told? I attach it. Attach both.")
    assert gilbert_score(tokens, phrases) == pytest.approx(-8.0 + 1.0 + 6.0 + 6.0)


# --- Measure 3: politeness strategies (ConvoKit rules) -----------------------


def test_politeness_markers_detect_each_strategy(nlp: Language) -> None:
    doc = nlp("Hi John. Please send the file. Thanks for the help. Sorry for the delay.")
    assert politeness_markers(doc) == PolitenessMarkers(
        gratitude=True, apology=True, please=True, greeting=True
    )


def test_politeness_markers_greeting_only_at_sentence_start(nlp: Language) -> None:
    doc = nlp("I said hi to him. The report is done.")
    assert politeness_markers(doc) == PolitenessMarkers(
        gratitude=False, apology=False, please=False, greeting=False
    )


def test_politeness_markers_i_appreciate_counts_as_gratitude(nlp: Language) -> None:
    assert politeness_markers(nlp("I appreciate your quick reply.")).gratitude


# --- Measure 4: language style matching --------------------------------------


def test_function_word_shares_put_each_word_in_one_category(nlp: Language) -> None:
    shares = function_word_shares(nlp("I did not see the report in it"))
    assert shares["personal_pronouns"] == pytest.approx(1 / 8)
    assert shares["negations"] == pytest.approx(1 / 8)
    assert shares["articles"] == pytest.approx(1 / 8)
    assert shares["prepositions"] == pytest.approx(1 / 8)
    assert shares["impersonal_pronouns"] == pytest.approx(1 / 8)
    assert shares["auxiliary_verbs"] == pytest.approx(1 / 8)
    assert set(shares) == set(LSM_CATEGORIES)


def test_style_matching_is_one_for_identical_text_and_lower_for_different(
    nlp: Language,
) -> None:
    text = "I think we should send the report to them, but not today."
    same = style_matching(nlp(text), nlp(text))
    different = style_matching(nlp(text), nlp("Revenue rose sharply."))
    assert same == pytest.approx(1.0, abs=1e-3)
    assert different < 0.5


def test_style_matching_formula_for_one_category() -> None:
    a = dict.fromkeys(LSM_CATEGORIES, 0.0)
    b = dict.fromkeys(LSM_CATEGORIES, 0.0)
    a["articles"], b["articles"] = 10.0, 30.0
    expected_articles = 1 - 20 / (40 + 0.0001)
    expected_rest = 1 - 0 / 0.0001
    assert lsm_from_shares(a, b) == pytest.approx(
        (expected_articles + expected_rest * (len(LSM_CATEGORIES) - 1)) / len(LSM_CATEGORIES)
    )


def test_parent_uids_finds_the_previous_message_in_the_same_thread() -> None:
    threads = pd.DataFrame(
        {
            "message_uid": ["a1", "a2", "a3", "b1", "b2"],
            "thread_id": ["a", "a", "a", "b", "b"],
            "position_in_thread": [0, 1, 2, 0, 1],
        }
    )
    assert parent_uids(threads, ["a3", "b2", "a1", "zz"]) == {"a3": "a2", "b2": "b1"}
