"""Q1 module tests: design reconstruction and feature-extraction glue.

Every mixed-model / sentence-model statistical behavior is already covered by
tests/test_hierarchy.py -- these tests cover what is new in q1.py: which
scenarios and cells the reconstructed Q1 grid actually contains, how a
generated reply turns into the two feature grains those models need, and the
old-vs-new comparison-table logic. No test calls a real LLM: where a
generated grid is needed, a fake client (the same pattern
tests/test_run.py uses) stands in for Ollama.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from thesis.analysis.q1 import (
    HISTORICAL_REPLY_LEVEL,
    Q1_TASK_STAKES,
    ContrastComparison,
    _tone_from_scenario_id,
    build_q1_cells,
    build_q1_scenarios,
    compare_to_historical,
    extract_q1_reply_features,
    extract_q1_sentence_features,
    format_comparison_table,
    generate_q1_grid,
    parse_replies,
    run_q1_analysis,
)
from thesis.llm.base import Capabilities, CompletionRequest, CompletionResponse, Provider, Usage
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.sim.persona import Persona, PersonaStyle
from thesis.sim.scenario import DIRECTIONS, TONES


def _persona(persona_id: str, *, seniority_rank: int = 3) -> Persona:
    """A distinct-looking persona per id/rank -- ``render_persona_block``
    renders the style numbers (not the id) into the prompt, so two personas
    with identical style stats would render byte-identical prompts and
    collapse onto the same cache entry, which would break any test that
    expects one generated call per (persona, scenario) cell."""
    return Persona(
        persona_id=persona_id,
        seniority_rank=seniority_rank,
        rank_label=f"Rank {seniority_rank}",
        department="Trading",
        style=PersonaStyle(
            mean_tokens=40.0 + 10.0 * seniority_rank,
            mean_recipients=2.0,
            imperative_ratio=0.10 + 0.02 * seniority_rank,
            hedge_rate=0.03,
            deference_rate=0.005,
            question_ratio=0.09,
        ),
        n_people=10,
        n_messages=100,
        derivation="cell",
    )


# ------------------------------------------------------------------ design


def test_build_q1_scenarios_has_24_scenarios() -> None:
    assert len(build_q1_scenarios()) == 24


def test_build_q1_scenarios_pins_each_task_type_to_one_stakes_level() -> None:
    for scenario in build_q1_scenarios():
        assert scenario.stakes == Q1_TASK_STAKES[scenario.task_type]


def test_build_q1_scenarios_covers_every_direction_and_tone_per_task_type() -> None:
    scenarios = build_q1_scenarios()
    for task_type in Q1_TASK_STAKES:
        subset = [s for s in scenarios if s.task_type == task_type]
        assert {s.direction for s in subset} == set(DIRECTIONS)
        assert {s.tone for s in subset} == set(TONES)
        assert len(subset) == len(DIRECTIONS) * len(TONES)


def test_build_q1_cells_reproduces_the_240_reply_design() -> None:
    """The design this whole module exists to reconstruct: 10 personas x 24
    scenarios x 1 replicate = 240 cells -- matching every figure this
    project has quoted for the pilot's size."""
    personas = [_persona(f"p{i}") for i in range(10)]
    cells = build_q1_cells(personas, "llama3.2:3b", "sim_q1")
    assert len(cells) == 240
    assert len({c.persona.persona_id for c in cells}) == 10
    assert len({c.scenario.scenario_id for c in cells}) == 24
    assert {c.replicate for c in cells} == {1}


def test_build_q1_cells_respects_n_replicates() -> None:
    cells = build_q1_cells([_persona("p0")], "llama3.2:3b", "sim_q1", n_replicates=2)
    assert len(cells) == 24 * 2
    assert {c.replicate for c in cells} == {1, 2}


def test_build_q1_cells_are_cache_ordered() -> None:
    """Cells sharing a cache_group (model/persona/direction) must be
    consecutive -- if they are not, the cache-ordering step buys nothing,
    the exact bug ``sim.grid``'s own docstring warns against."""
    personas = [_persona(f"p{i}") for i in range(3)]
    cells = build_q1_cells(personas, "llama3.2:3b", "sim_q1")
    seen: set[str] = set()
    last_group = None
    for cell in cells:
        if cell.cache_group != last_group:
            assert cell.cache_group not in seen, "cache group is not contiguous"
            seen.add(cell.cache_group)
            last_group = cell.cache_group


def test_tone_from_scenario_id() -> None:
    assert _tone_from_scenario_id("approve_or_decline__up__high__assertive") == "assertive"
    assert _tone_from_scenario_id("report_problem__down__routine__warm") == "warm"


# --------------------------------------------------------------- comparison


class _FakeContrast:
    """A stand-in for MixedModelResult/SentenceModelResult exposing only
    the ``.contrast()`` method compare_to_historical actually needs."""

    def __init__(self, values: dict[str, tuple[float, float]]) -> None:
        self._values = values

    def contrast(self, level: str) -> tuple[float, float]:
        return self._values[level]


def test_compare_to_historical_pairs_old_and_new_values() -> None:
    new = _FakeContrast({"up": (0.20, 0.01), "down": (-0.10, 0.30)})
    comparisons = compare_to_historical(new, HISTORICAL_REPLY_LEVEL)
    assert len(comparisons) == 2
    up = next(c for c in comparisons if c.level == "up")
    assert up.old_coefficient == HISTORICAL_REPLY_LEVEL["up"][0]
    assert up.old_p_value == HISTORICAL_REPLY_LEVEL["up"][1]
    assert up.new_coefficient == 0.20
    assert up.new_p_value == 0.01


def test_compare_to_historical_is_sorted_by_level() -> None:
    new = _FakeContrast({"up": (0.1, 0.5), "down": (0.2, 0.4)})
    comparisons = compare_to_historical(new, HISTORICAL_REPLY_LEVEL)
    assert [c.level for c in comparisons] == sorted(c.level for c in comparisons)


def test_format_comparison_table_includes_label_and_every_level() -> None:
    comparisons = [
        ContrastComparison("down", -0.05, 0.44, -0.03, 0.60),
        ContrastComparison("up", 0.06, 0.40, 0.10, 0.20),
    ]
    table = format_comparison_table(comparisons, label="a test table")
    assert "a test table" in table
    assert "down" in table
    assert "up" in table
    assert "0.060" in table  # old up coefficient, formatted to 3dp


# ------------------------------------------------------------ feature glue


class FakeClient:
    """A client that writes a direction-dependent body and never touches
    the network -- the same role FakeClient plays in tests/test_run.py.

    The body text varies with the direction framing line already present in
    the rendered prompt (see ``sim.scenario._DIRECTION_FRAMING``), so the
    resulting grid has real, non-degenerate variation in imperative
    language across directions instead of one constant value everywhere.
    """

    provider: Provider = "ollama"

    def __init__(self) -> None:
        self.calls: list[CompletionRequest] = []

    def capabilities(self, model: str) -> Capabilities:
        return Capabilities(
            supports_sampling_params=True,
            min_cacheable_prompt_tokens=10**9,
            thinking_on_by_default=False,
        )

    def count_tokens(self, request: CompletionRequest) -> int:
        return 100

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        content = request.messages[0].content
        if "senior to you" in content:
            body = "Could you take a look at this when you have a moment?"
        elif "reports into" in content:
            body = "Send me the updated numbers by end of day."
        else:
            body = "Let's sync on this sometime this week."
        payload = {
            "subject": "Re: update",
            "body": body,
            "decision": "accept",
            "confidence": "medium",
            "reasoning_brief": "Routine, within my remit.",
        }
        return CompletionResponse(
            text="{}",
            usage=Usage(input_tokens=200, output_tokens=20),
            model=f"local/{request.model}",
            stop_reason="end_turn",
            parsed=payload,
        )

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        raise NotImplementedError

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        raise NotImplementedError


@pytest.fixture
def small_grid(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A tiny (3-persona) Q1 grid, generated once against a fake client into
    a tmp-scoped cache and ledger, so every test below can reuse the same
    generation without re-parsing the grid's replies with spaCy from scratch
    each time -- and without touching the project's real response cache or
    the tracked cost-ledger CSV."""
    personas = [_persona(f"p{i}", seniority_rank=i + 1) for i in range(3)]
    client = FakeClient()
    grid = generate_q1_grid(
        client,
        model="llama3.2:3b",
        personas=personas,
        stores={},
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
    )
    return grid


def test_generate_q1_grid_produces_one_row_per_cell(small_grid) -> None:  # type: ignore[no-untyped-def]
    assert small_grid.n_cells == 3 * 24  # 3 personas x 24 scenarios x 1 replicate
    assert len(small_grid.frame) == small_grid.n_cells
    assert small_grid.n_generated == small_grid.n_cells
    assert small_grid.n_from_cache == 0


def test_generate_q1_grid_second_call_is_served_from_cache(tmp_path: Path) -> None:
    personas = [_persona("p0")]
    cache = ResponseCache(tmp_path / "cache")
    ledger = CostLedger(tmp_path / "ledger.csv")

    first = generate_q1_grid(
        FakeClient(),
        model="llama3.2:3b",
        personas=personas,
        stores={},
        limit=4,
        cache=cache,
        ledger=ledger,
    )
    assert first.n_generated == 4

    second_client = FakeClient()
    second = generate_q1_grid(
        second_client,
        model="llama3.2:3b",
        personas=personas,
        stores={},
        limit=4,
        cache=cache,
        ledger=ledger,
    )
    assert second.n_from_cache == 4
    assert second.n_generated == 0
    assert second_client.calls == []


def test_extract_q1_reply_features_adds_tone_and_linguistic_columns(small_grid) -> None:  # type: ignore[no-untyped-def]
    features = extract_q1_reply_features(small_grid.frame)
    assert len(features) == len(small_grid.frame)
    for column in ("tone", "imperative_ratio", "hedge_rate", "question_ratio"):
        assert column in features.columns
    assert set(features["tone"]) == set(TONES)


def test_extract_q1_reply_features_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="no rows"):
        extract_q1_reply_features(pd.DataFrame())


def test_extract_q1_sentence_features_has_one_row_per_sentence(small_grid) -> None:  # type: ignore[no-untyped-def]
    reply_features = extract_q1_reply_features(small_grid.frame)
    sentence_features = extract_q1_sentence_features(small_grid.frame)

    assert sentence_features["cell_id"].nunique() <= len(small_grid.frame)
    counted = sentence_features.groupby("cell_id").size()
    expected = reply_features.set_index("cell_id")["n_sentences"]
    # Every reply in this fixture is exactly one sentence, but the join
    # itself -- not that specific count -- is what this test protects.
    for cell_id, n in counted.items():
        assert n == expected.loc[cell_id]


def test_extract_q1_sentence_features_carries_direction_and_tone(small_grid) -> None:  # type: ignore[no-untyped-def]
    sentence_features = extract_q1_sentence_features(small_grid.frame)
    assert set(sentence_features["direction"]) <= set(DIRECTIONS)
    assert set(sentence_features["tone"]) == set(TONES)
    assert "is_imperative" in sentence_features.columns


def test_extract_functions_reuse_precomputed_docs(small_grid) -> None:  # type: ignore[no-untyped-def]
    """Passing ``docs=`` must skip re-parsing -- confirmed indirectly by
    checking the two extraction paths agree when fed the same parse."""
    docs = parse_replies(small_grid.frame)
    with_docs = extract_q1_reply_features(small_grid.frame, docs=docs)
    without_docs = extract_q1_reply_features(small_grid.frame)
    pd.testing.assert_series_equal(with_docs["imperative_ratio"], without_docs["imperative_ratio"])


# --------------------------------------------------------------- end to end


def test_run_q1_analysis_produces_a_complete_result(small_grid) -> None:  # type: ignore[no-untyped-def]
    result = run_q1_analysis(small_grid)

    assert result.reply_model.n_groups == 3
    assert result.sentence_model.n_groups == 3
    assert len(result.reply_level_comparison) == 2
    assert len(result.sentence_level_comparison) == 2
    assert {c.level for c in result.reply_level_comparison} == {"up", "down"}
    assert result.decision_association.n_observations == len(small_grid.frame)
