import numpy as np
import pandas as pd
import pytest

from thesis.analysis.q1_measures import pronoun_rates


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
