from __future__ import annotations

import pytest

from thesis.analysis.judge_consensus import answer_shares, consensus, pairwise_kappa


def test_consensus_uses_only_items_every_judge_answered() -> None:
    answers = {
        "a": {"1": "up", "2": "down", "3": "lateral"},
        "b": {"1": "up", "2": "down"},
        "c": {"1": "up", "2": "down", "3": "up"},
    }
    result = consensus(answers)
    assert result.n == 2
    assert result.share_agree == 1.0


def test_pairwise_kappa_is_one_for_identical_judges_and_names_each_pair() -> None:
    answers = {
        "a": {"1": "up", "2": "down", "3": "lateral"},
        "b": {"1": "up", "2": "down", "3": "lateral"},
        "c": {"1": "down", "2": "up", "3": "up"},
    }
    kappas = pairwise_kappa(answers)
    assert set(kappas) == {"a vs b", "a vs c", "b vs c"}
    assert kappas["a vs b"] == pytest.approx(1.0)
    assert kappas["a vs c"] < 0


def test_answer_shares_cover_every_direction() -> None:
    shares = answer_shares({"1": "down", "2": "down", "3": "up", "4": "down"})
    assert shares == {"up": 0.25, "lateral": 0.0, "down": 0.75}
