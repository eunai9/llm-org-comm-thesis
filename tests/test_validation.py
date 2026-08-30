"""Tests for the validation pass: embeddings, the embedding map, the review packet.

No model calls anywhere. The embedding client is exercised against a fake
transport, and everything downstream is given vectors directly -- what is
tested is the properties the validation argument rests on: that the vector
cache actually prevents a second call, that separability is measured with
pairs kept inside one fold, that the review sample is the same 100 items every
time it is drawn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from thesis.analysis.embedding_map import (
    matched_vs_mismatched_cosine,
    separability_auc,
    truncate_words,
)
from thesis.analysis.review_pack import build_packet, draw_sample, screen, summarize_flags
from thesis.analysis.review_summary import failure_counts, mirrored_rate_by_direction, summarize
from thesis.llm.embeddings import embed_texts, l2_normalize


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _Transport:
    """Counts calls so a cache hit is observable, not merely assumed."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, url: str, json: dict[str, Any], timeout: float) -> _FakeResponse:
        texts = list(json["input"])
        self.calls.append(texts)
        return _FakeResponse({"embeddings": [[float(len(t)), 1.0, 0.0] for t in texts]})


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> _Transport:
    fake = _Transport()
    monkeypatch.setattr("thesis.llm.embeddings.httpx.post", fake)
    return fake


# ------------------------------------------------------------------ embeddings


def test_embeddings_are_returned_in_the_order_requested(
    transport: _Transport, tmp_path: Path
) -> None:
    vectors = embed_texts(["a", "bbb", "cc"], cache_path=tmp_path / "cache.parquet")
    assert [v[0] for v in vectors] == [1.0, 3.0, 2.0]


def test_a_second_call_is_served_from_cache(transport: _Transport, tmp_path: Path) -> None:
    cache = tmp_path / "cache.parquet"
    embed_texts(["a", "b"], cache_path=cache)
    embed_texts(["a", "b"], cache_path=cache)
    assert len(transport.calls) == 1


def test_duplicate_texts_are_embedded_once(transport: _Transport, tmp_path: Path) -> None:
    """The corpus repeats boilerplate replies verbatim; paying twice is waste."""
    embed_texts(["same", "same", "other"], cache_path=tmp_path / "cache.parquet")
    assert transport.calls == [["same", "other"]]


def test_l2_normalize_makes_rows_unit_length() -> None:
    normalized = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_l2_normalize_survives_a_zero_row() -> None:
    assert np.all(np.isfinite(l2_normalize(np.zeros((1, 3)))))


# --------------------------------------------------------------- embedding map


def test_truncate_words_keeps_the_first_n() -> None:
    assert truncate_words("one two three four", 2) == "one two"


def test_truncate_words_never_returns_empty() -> None:
    """A zero-word generated reply would otherwise silently produce an empty pair."""
    assert truncate_words("one two", 0) == "one"


def test_separability_auc_is_perfect_on_linearly_separable_groups() -> None:
    rng = np.random.default_rng(0)
    real = rng.normal(0.0, 0.05, size=(20, 4))
    generated = rng.normal(5.0, 0.05, size=(20, 4))
    vectors = np.vstack([real, generated])
    groups = [f"t{i}" for i in range(20)] * 2
    auc = separability_auc(vectors, [False] * 20 + [True] * 20, groups, n_folds=4)
    assert auc == pytest.approx(1.0)


def test_separability_folds_keep_a_pair_together() -> None:
    """Grouping by thread is what stops the classifier recognising the topic."""
    from sklearn.model_selection import StratifiedGroupKFold

    from thesis.analysis.embedding_map import SEED

    groups = np.asarray([f"t{i}" for i in range(10)] * 2)
    labels = np.asarray([0] * 10 + [1] * 10)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train, test in splitter.split(np.zeros((20, 2)), labels, groups):
        assert not set(groups[train]) & set(groups[test])


def test_matched_cosine_beats_mismatched_when_pairs_are_aligned() -> None:
    real = np.eye(6)
    generated = np.eye(6) + 0.01
    matched, mismatched = matched_vs_mismatched_cosine(real, generated)
    assert matched.mean() > mismatched.mean()


# --------------------------------------------------------------- review packet


def _pairs_frame(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "thread_id": [f"t{i:03d}" for i in range(n)],
            "persona_id": ["r1_legal"] * n,
            "direction": ["up", "down", "lateral"] * (n // 3),
            "decision": ["accept"] * n,
            "confidence": ["high"] * n,
            "stimulus_text": [f"Please handle item {i}." for i in range(n)],
            "real_reply_body_recleaned": ["Will do, thanks."] * n,
            "generated_subject": ["Re: item"] * n,
            "generated_reply": ["I will handle it today."] * n,
            "reasoning_brief": ["routine"] * n,
        }
    )


def test_screen_flags_a_number_absent_from_the_stimulus() -> None:
    flags = screen("Can you confirm the volume?", "We agreed on 40 units.")
    assert flags["new_numbers"] == ["40"]


def test_screen_does_not_flag_a_number_the_stimulus_already_contained() -> None:
    flags = screen("We agreed on 40 units.", "Yes, 40 is right.")
    assert flags["new_numbers"] == []


def test_screen_detects_a_reply_that_stops_mid_sentence() -> None:
    assert screen("x", "Can we discuss the agreement")["ends_mid_sentence"] is True


def test_sample_is_identical_across_draws() -> None:
    """A review sample that moves between runs can never be pooled or re-checked."""
    frame = _pairs_frame()
    first = draw_sample(frame, n=12)
    second = draw_sample(frame, n=12)
    assert list(first["thread_id"]) == list(second["thread_id"])


def test_sample_keeps_the_direction_mix() -> None:
    frame = _pairs_frame(30)
    shares = draw_sample(frame, n=15)["direction"].value_counts(normalize=True)
    assert shares.max() - shares.min() < 0.2


def test_packet_leaves_every_coding_column_empty() -> None:
    """A pre-filled verdict is a suggestion, and a suggestion contaminates coding."""
    packet = build_packet(draw_sample(_pairs_frame(), n=9))
    assert (packet["failure_mode"] == "").all()
    assert (packet["notes"] == "").all()


def test_flag_summary_counts_what_it_says() -> None:
    packet = build_packet(draw_sample(_pairs_frame(), n=9))
    summary = summarize_flags(packet)
    assert summary["n_items"] == 9
    assert summary["share_no_signoff"] == 1.0


# -------------------------------------------------------------- review summary


def _coded_frame() -> pd.DataFrame:
    modes = ["ok"] * 4 + ["mirrors_request"] * 3 + ["generic"]
    return pd.DataFrame(
        {
            "failure_mode": modes,
            "direction": ["up", "up", "lateral", "lateral", "up", "down", "down", "lateral"],
            "generated_reply": [f"reply {i}" for i in range(8)],
            "plausible_as_a_reply": ["y"] * 5 + ["n"] * 3,
            "addresses_the_request": ["y"] * 5 + ["n"] * 3,
            "fabricates_detail": ["n"] * 7 + ["y"],
        }
    )


def test_failure_counts_put_ok_first() -> None:
    assert next(iter(failure_counts(_coded_frame()).index)) == "ok"


def test_mirrored_rate_is_a_share_of_that_direction() -> None:
    rates = mirrored_rate_by_direction(_coded_frame())
    assert rates.loc["down", "rate"] == pytest.approx(1.0)
    assert rates.loc["up", "rate"] == pytest.approx(1 / 3)


def test_summary_reports_distinct_replies_alongside_items() -> None:
    """190 pairs are not 190 independent texts; the summary has to say both."""
    summary = summarize(_coded_frame())
    assert summary["n_items"] == 8
    assert summary["n_distinct_generated_replies"] == 8
