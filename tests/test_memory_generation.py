"""Memory-generation tests.

No live model calls: a fake client returns controlled structured payloads, so
what's tested is the generation logic itself -- schema parsing, the recency
spread, importance mapping, reflection batching, and the freeze/load
round-trip -- not whether a particular local model behaves well.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import make_persona
from thesis.llm.base import CompletionRequest, CompletionResponse, Provider, Usage
from thesis.sim.memory import REFLECTION_EVERY, REFLECTIONS_PER_TRIGGER, MemoryItem
from thesis.sim.memory_generation import (
    IMPORTANCE_VALUE,
    OBSERVATION_WINDOW_HOURS,
    freeze_memory,
    generate_all,
    generate_observations,
    generate_persona_memory,
    generate_reflections,
    load_frozen_memory,
)


class _ScriptedClient:
    """Returns one canned parsed payload per call, in order."""

    provider: Provider = "anthropic"
    model = "test-model"

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = list(payloads)
        self.calls: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        payload = self._payloads.pop(0)
        return CompletionResponse(text="{}", usage=Usage(), model=self.model, parsed=payload)


def _obs_payload(n: int, importance: str = "medium") -> dict[str, Any]:
    return {
        "observations": [{"text": f"Observation {i}", "importance": importance} for i in range(n)]
    }


def _reflection_payload(n: int) -> dict[str, Any]:
    return {"reflections": [f"Reflection {i}" for i in range(n)]}


# ------------------------------------------------------------ observations


def test_generate_observations_spreads_across_the_window() -> None:
    client = _ScriptedClient([_obs_payload(5)])
    items = generate_observations(make_persona(), client, n=5)

    assert len(items) == 5
    # First-generated is oldest, last-generated is most recent.
    assert items[0].hours_ago == pytest.approx(OBSERVATION_WINDOW_HOURS)
    assert items[-1].hours_ago == pytest.approx(0.0)
    assert all(a.hours_ago > b.hours_ago for a, b in itertools.pairwise(items))


def test_generate_observations_maps_importance_labels() -> None:
    client = _ScriptedClient([_obs_payload(3, importance="high")])
    items = generate_observations(make_persona(), client, n=3)
    assert all(item.importance == IMPORTANCE_VALUE["high"] for item in items)


def test_generate_observations_skips_blank_text() -> None:
    client = _ScriptedClient(
        [
            {
                "observations": [
                    {"text": "  ", "importance": "low"},
                    {"text": "Real one", "importance": "low"},
                ]
            }
        ]
    )
    items = generate_observations(make_persona(), client, n=2)
    assert [item.text for item in items] == ["Real one"]


def test_generate_observations_truncates_to_n() -> None:
    """A model that ignores the requested count must not silently overshoot."""
    client = _ScriptedClient([_obs_payload(10)])
    items = generate_observations(make_persona(), client, n=3)
    assert len(items) == 3


def test_generate_observations_handles_missing_parsed_payload() -> None:
    """A response with no parseable JSON must not crash -- an empty store is
    the correct, honest result."""
    client = _ScriptedClient([{}])
    items = generate_observations(make_persona(), client, n=5)
    assert items == []


def test_none_of_the_generated_observations_are_marked_as_reflections() -> None:
    client = _ScriptedClient([_obs_payload(5)])
    items = generate_observations(make_persona(), client, n=5)
    assert not any(item.is_reflection for item in items)


# ------------------------------------------------------------- reflections


def _observations(n: int) -> list[MemoryItem]:
    return [MemoryItem(text=f"obs {i}", importance=0.5, hours_ago=float(n - i)) for i in range(n)]


def test_generate_reflections_calls_once_per_batch() -> None:
    observations = _observations(REFLECTION_EVERY * 2)
    client = _ScriptedClient([_reflection_payload(REFLECTIONS_PER_TRIGGER) for _ in range(2)])
    reflections = generate_reflections(observations, client)

    assert len(client.calls) == 2
    assert len(reflections) == REFLECTIONS_PER_TRIGGER * 2
    assert all(r.is_reflection for r in reflections)


def test_generate_reflections_inherits_the_batchs_most_recent_hours_ago() -> None:
    observations = _observations(REFLECTION_EVERY)
    newest = min(o.hours_ago for o in observations)
    client = _ScriptedClient([_reflection_payload(REFLECTIONS_PER_TRIGGER)])

    reflections = generate_reflections(observations, client)
    assert all(r.hours_ago == newest for r in reflections)


def test_generate_reflections_skips_a_trailing_partial_batch() -> None:
    """A batch of 2 has far weaker support than one of REFLECTION_EVERY;
    reflection_batches() already drops it -- this checks no call is made for it."""
    observations = _observations(REFLECTION_EVERY + 2)
    client = _ScriptedClient([_reflection_payload(REFLECTIONS_PER_TRIGGER)])
    generate_reflections(observations, client)
    assert len(client.calls) == 1


def test_generate_reflections_handles_no_full_batches() -> None:
    client = _ScriptedClient([])
    reflections = generate_reflections(_observations(2), client)
    assert reflections == []
    assert client.calls == []


# --------------------------------------------------------- full persona


def test_generate_persona_memory_combines_observations_and_reflections() -> None:
    n_obs = REFLECTION_EVERY * 2
    client = _ScriptedClient(
        [_obs_payload(n_obs)] + [_reflection_payload(REFLECTIONS_PER_TRIGGER) for _ in range(2)]
    )
    items = generate_persona_memory(make_persona(), client, n_observations=n_obs)

    observations = [i for i in items if not i.is_reflection]
    reflections = [i for i in items if i.is_reflection]
    assert len(observations) == n_obs
    assert len(reflections) == REFLECTIONS_PER_TRIGGER * 2


def test_generate_all_keys_by_persona_id() -> None:
    personas = [make_persona("p1"), make_persona("p2")]
    payloads = []
    for _ in personas:
        payloads.append(_obs_payload(REFLECTION_EVERY))
        payloads.append(_reflection_payload(REFLECTIONS_PER_TRIGGER))
    client = _ScriptedClient(payloads)

    stores = generate_all(personas, client)
    assert set(stores) == {"p1", "p2"}


# ------------------------------------------------------- freeze / load


def test_freeze_and_load_round_trips_exactly(tmp_path: Path) -> None:
    stores = {
        "p1": [MemoryItem(text="a", importance=0.5, hours_ago=10.0)],
        "p2": [MemoryItem(text="b", importance=0.9, hours_ago=1.0, kind="reflection")],
    }
    path = tmp_path / "snapshot.json"
    freeze_memory(stores, path)
    restored = load_frozen_memory(path)
    assert restored == stores


def test_load_frozen_memory_raises_a_clear_error_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no memory snapshot"):
        load_frozen_memory(tmp_path / "does_not_exist.json")
