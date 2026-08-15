"""Memory-stream tests.

Retrieval feeds a *cached* prompt prefix, so determinism is not a nicety here:
if the same store and query could return memories in a different order, the
prefix would differ byte-for-byte between runs and the cache would never be
read. That property gets its own tests alongside the scoring behaviour.
"""

from __future__ import annotations

import pytest

from thesis.sim.memory import (
    RECENCY_DECAY,
    REFLECTION_EVERY,
    TOP_K,
    MemoryItem,
    _normalize,
    as_reflection,
    build_reflection_prompt,
    recency_scores,
    reflection_batches,
    render_memory_block,
    retrieve,
    tfidf_relevance,
)


def _item(text: str, importance: float = 0.5, hours_ago: float = 1.0) -> MemoryItem:
    return MemoryItem(text=text, importance=importance, hours_ago=hours_ago)


# ------------------------------------------------------------- normalization


def test_normalize_maps_to_unit_range() -> None:
    assert _normalize([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_constant_signal_normalizes_to_ones_not_zeros() -> None:
    """A signal carrying no information must not veto every candidate."""
    assert _normalize([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]


def test_normalize_handles_empty() -> None:
    assert _normalize([]) == []


# ------------------------------------------------------------------- recency


def test_recency_decays_with_elapsed_time() -> None:
    scores = recency_scores([_item("a", hours_ago=0), _item("b", hours_ago=100)])
    assert scores[0] == 1.0
    assert scores[1] == pytest.approx(RECENCY_DECAY**100)
    assert scores[0] > scores[1]


# ----------------------------------------------------------------- relevance


def test_tfidf_relevance_prefers_lexical_overlap() -> None:
    docs = ["the settlement date moved forward", "lunch options near the office"]
    scores = tfidf_relevance("we need to discuss the settlement date", docs)
    assert scores[0] > scores[1]


def test_tfidf_relevance_survives_empty_vocabulary() -> None:
    """All-stopword input leaves no vocabulary; that is 'nothing relevant',
    not a crash."""
    assert tfidf_relevance("the and of", ["the and of"]) == [0.0]


def test_tfidf_relevance_handles_no_documents() -> None:
    assert tfidf_relevance("anything", []) == []


# ----------------------------------------------------------------- retrieval


def test_retrieve_returns_at_most_top_k() -> None:
    items = [_item(f"observation number {i}") for i in range(20)]
    assert len(retrieve(items, "observation", top_k=TOP_K)) == TOP_K


def test_retrieve_handles_empty_store() -> None:
    assert retrieve([], "anything") == []


def test_retrieve_is_deterministic() -> None:
    """A cached prefix requires byte-identical memory ordering across runs."""
    items = [_item(f"item {i}", importance=0.5, hours_ago=float(i)) for i in range(15)]
    first = [m.text for m in retrieve(items, "item")]
    second = [m.text for m in retrieve(items, "item")]
    assert first == second


def test_retrieve_breaks_ties_toward_earlier_items() -> None:
    """When every signal ties, input order decides -- not dict or sort chance.

    Items are given distinct text so they can be told apart, and a constant
    relevance scorer so that relevance cannot break the tie for us. Importance
    and recency are held equal, leaving the tie-break rule as the only thing
    that can determine the result.
    """
    items = [_item(f"item {i}", importance=0.5, hours_ago=1.0) for i in range(5)]

    def constant_relevance(query: str, documents: list[str]) -> list[float]:
        return [0.5] * len(documents)

    order = retrieve(items, "q", top_k=3, relevance=constant_relevance)  # type: ignore[arg-type]
    assert [m.text for m in order] == ["item 0", "item 1", "item 2"]


def test_importance_raises_retrieval_rank() -> None:
    low = _item("budget review meeting", importance=0.0, hours_ago=1.0)
    high = _item("budget review meeting", importance=1.0, hours_ago=1.0)
    retrieved = retrieve([low, high], "budget review meeting", top_k=1)
    assert retrieved[0].importance == 1.0


def test_recency_raises_retrieval_rank() -> None:
    old = _item("budget review meeting", importance=0.5, hours_ago=500.0)
    new = _item("budget review meeting", importance=0.5, hours_ago=0.0)
    retrieved = retrieve([old, new], "budget review meeting", top_k=1)
    assert retrieved[0].hours_ago == 0.0


def test_relevance_raises_retrieval_rank() -> None:
    off_topic = _item("cafeteria menu changed", importance=0.5, hours_ago=1.0)
    on_topic = _item("settlement date discussion", importance=0.5, hours_ago=1.0)
    retrieved = retrieve([off_topic, on_topic], "settlement date", top_k=1)
    assert retrieved[0].text == "settlement date discussion"


def test_relevance_scorer_is_pluggable() -> None:
    """The embedding upgrade must not require editing this module."""
    items = [_item("first"), _item("second")]

    def always_favour_second(query: str, documents: list[str]) -> list[float]:
        return [0.0, 1.0]

    retrieved = retrieve(items, "q", top_k=1, relevance=always_favour_second)  # type: ignore[arg-type]
    assert retrieved[0].text == "second"


# ---------------------------------------------------------------- reflection


def test_reflection_batches_group_by_interval() -> None:
    observations = [_item(f"o{i}") for i in range(10)]
    batches = reflection_batches(observations, every=REFLECTION_EVERY)
    assert len(batches) == 2
    assert all(len(b) == REFLECTION_EVERY for b in batches)


def test_trailing_partial_batch_is_dropped() -> None:
    """A reflection over 2 observations has far weaker support than over 5."""
    observations = [_item(f"o{i}") for i in range(12)]
    batches = reflection_batches(observations, every=5)
    assert len(batches) == 2
    assert sum(len(b) for b in batches) == 10


def test_too_few_observations_yields_no_reflections() -> None:
    assert reflection_batches([_item("only one")], every=5) == []


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        reflection_batches([_item("a")], every=0)


def test_reflection_prompt_contains_every_observation() -> None:
    batch = [_item(f"observation {i}") for i in range(5)]
    prompt = build_reflection_prompt(batch, 3)
    for item in batch:
        assert item.text in prompt
    assert "3" in prompt


def test_reflection_inherits_recency_of_its_source() -> None:
    """A reflection over old material must not present as a fresh memory."""
    source = _item("something old", hours_ago=400.0)
    reflection = as_reflection(source)
    assert reflection.is_reflection
    assert reflection.hours_ago == 400.0


# -------------------------------------------------------------- rendering


def test_render_separates_reflections_from_observations() -> None:
    items = [_item("I checked the numbers"), as_reflection(_item("I verify before agreeing"))]
    block = render_memory_block(items)
    assert "How you tend to operate:" in block
    assert "Recent context:" in block
    assert block.index("How you tend to operate:") < block.index("Recent context:")


def test_render_empty_memory_is_empty_string() -> None:
    assert render_memory_block([]) == ""
