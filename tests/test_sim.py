"""Simulator tests.

The properties tested here are the ones whose failure would be silent: a
prompt prefix that stops caching, a grid whose cells collide, a persona that
turns out to describe one identifiable person, and a schema that permits a
value the analysis cannot interpret.
"""

from __future__ import annotations

import pytest

from thesis.sim.grid import GridCell, expand, order_for_cache, summarize
from thesis.sim.persona import (
    DEPARTMENTS,
    MIN_PEOPLE_PER_PERSONA,
    RANKS,
    Persona,
    PersonaStyle,
    collapsed_ranks,
    coverage_report,
    render_persona_block,
)
from thesis.sim.prompt import (
    MAX_MODEL_CACHE_MINIMUM,
    assemble,
    build_stable_prefix,
    estimate_prefix_tokens,
)
from thesis.sim.scenario import DIRECTIONS, STAKES, TASK_TYPES, build_scenarios
from thesis.sim.schemas import (
    CONFIDENCE_LEVELS,
    DECISIONS,
    RESPONSE_SCHEMA,
    InvalidResponseError,
    validate_response,
)


def _style(**overrides: float) -> PersonaStyle:
    base = {
        "mean_tokens": 50.0,
        "mean_recipients": 2.0,
        "imperative_ratio": 0.15,
        "hedge_rate": 0.03,
        "deference_rate": 0.005,
        "question_ratio": 0.09,
    }
    base.update(overrides)
    return PersonaStyle(**base)


def _persona(persona_id: str = "r3_trading", rank: int = 3, **kwargs: object) -> Persona:
    defaults: dict[str, object] = {
        "persona_id": persona_id,
        "seniority_rank": rank,
        "rank_label": "Director",
        "department": "Trading",
        "style": _style(),
        "n_people": 25,
        "n_messages": 4236,
        "derivation": "cell",
    }
    defaults.update(kwargs)
    return Persona(**defaults)  # type: ignore[arg-type]


# ------------------------------------------------------------------ scenarios


def test_scenario_grid_is_fully_crossed() -> None:
    scenarios = build_scenarios()
    assert len(scenarios) == len(TASK_TYPES) * len(DIRECTIONS) * len(STAKES)
    assert len({s.scenario_id for s in scenarios}) == len(scenarios)


def test_every_direction_appears() -> None:
    """Direction is the Q1 manipulation; a missing level makes it inestimable."""
    directions = {s.direction for s in build_scenarios()}
    assert directions == set(DIRECTIONS)


# -------------------------------------------------------------------- persona


def test_persona_block_carries_no_identifying_text() -> None:
    """A persona must describe a role, never a person."""
    block = render_persona_block(_persona())
    assert "composite role archetype" in block
    assert "Director" in block and "Trading" in block
    # No name-like or address-like content should ever appear.
    assert "@" not in block


def test_persona_block_avoids_broken_article() -> None:
    """'a Employee' reached the model before this was fixed."""
    block = render_persona_block(_persona(rank=1, rank_label="Employee"))
    assert "a Employee" not in block


def test_collapsed_ranks_flags_identical_pooled_personas() -> None:
    """Two pooled personas at one rank differ only by label, not by style.

    The department factor is not varying there, and a department effect
    estimated at that rank would be measuring the label alone.
    """
    shared = _style()
    personas = [
        _persona("r5_trading", 5, department="Trading", style=shared, derivation="rank_pooled"),
        _persona("r5_legal", 5, department="Legal", style=shared, derivation="rank_pooled"),
    ]
    assert collapsed_ranks(personas) == [5]


def test_distinct_styles_are_not_flagged_as_collapsed() -> None:
    personas = [
        _persona("r5_trading", 5, style=_style(mean_tokens=40.0), derivation="rank_pooled"),
        _persona("r5_legal", 5, style=_style(mean_tokens=90.0), derivation="rank_pooled"),
    ]
    assert collapsed_ranks(personas) == []


def test_cell_derived_personas_are_not_flagged() -> None:
    shared = _style()
    personas = [
        _persona("r3_trading", 3, style=shared),
        _persona("r3_legal", 3, style=shared, department="Legal"),
    ]
    assert collapsed_ranks(personas) == []


def test_coverage_report_surfaces_pooling() -> None:
    personas = [
        _persona("r3_trading", 3),
        _persona("r5_legal", 5, derivation="rank_pooled"),
    ]
    report = coverage_report(personas)
    assert report["n_pooled_to_rank_level"] == 1
    assert report["pooled_personas"] == ["r5_legal"]
    assert report["min_people_per_persona"] == MIN_PEOPLE_PER_PERSONA


def test_rank_six_is_excluded_by_design() -> None:
    """Rank 6 is the smallest and most famous population -- highest risk."""
    assert 6 not in RANKS
    assert set(DEPARTMENTS) == {"Trading", "Legal"}


# --------------------------------------------------------------------- schema


def test_schema_forbids_extra_properties() -> None:
    """Required by the structured-output API on every object."""
    assert RESPONSE_SCHEMA["additionalProperties"] is False


def test_schema_declares_no_unsupported_numeric_bounds() -> None:
    """minimum/maximum are silently unsupported; relying on them would not
    constrain anything. This is why confidence is an ordinal enum."""
    for spec in RESPONSE_SCHEMA["properties"].values():
        assert "minimum" not in spec
        assert "maximum" not in spec
        assert "multipleOf" not in spec


def test_decision_is_a_required_dependent_variable() -> None:
    assert "decision" in RESPONSE_SCHEMA["required"]
    assert set(RESPONSE_SCHEMA["properties"]["decision"]["enum"]) == set(DECISIONS)


def test_validate_accepts_a_well_formed_response() -> None:
    payload = {
        "subject": "Re: volumes",
        "body": "Numbers attached.",
        "decision": "accept",
        "confidence": "high",
        "reasoning_brief": "Routine request within my remit.",
    }
    assert validate_response(payload) == payload


def test_validate_rejects_missing_fields() -> None:
    with pytest.raises(InvalidResponseError):
        validate_response({"subject": "x", "body": "y"})


def test_validate_rejects_out_of_enum_values() -> None:
    """Catches a cached response written under an older schema version."""
    payload = {
        "subject": "s",
        "body": "b",
        "decision": "maybe",
        "confidence": "high",
        "reasoning_brief": "r",
    }
    with pytest.raises(InvalidResponseError):
        validate_response(payload)

    payload["decision"] = "accept"
    payload["confidence"] = "0.87"
    with pytest.raises(InvalidResponseError):
        validate_response(payload)


def test_confidence_levels_are_ordinal() -> None:
    assert list(CONFIDENCE_LEVELS) == ["low", "medium", "high"]


# --------------------------------------------------------------------- prompt


def test_stable_prefix_is_identical_across_scenarios() -> None:
    """The whole cost model rests on this: same persona and direction, same
    prefix, byte for byte."""
    persona = _persona()
    up_scenarios = [s for s in build_scenarios() if s.direction == "up"]
    prefixes = {assemble(persona, s).stable_prefix for s in up_scenarios}
    assert len(prefixes) == 1


def test_stable_prefix_differs_across_directions() -> None:
    persona = _persona()
    by_direction = {d: build_stable_prefix(persona, d) for d in DIRECTIONS}
    assert len(set(by_direction.values())) == len(DIRECTIONS)


def test_scenario_content_stays_out_of_the_cached_prefix() -> None:
    """If scenario text leaked into the prefix, every cell would be unique and
    the cache would silently never be read."""
    persona = _persona()
    scenario = next(s for s in build_scenarios() if s.task_type == "resolve_disagreement")
    assembled = assemble(persona, scenario)
    assert scenario.incoming_message not in assembled.stable_prefix
    assert scenario.incoming_message in assembled.variable_suffix


def test_prefix_clears_the_caching_floor() -> None:
    """Below the model minimum, caching does nothing and says nothing.

    Asserted against the largest minimum across the models this prefix reaches
    (1024, from claude-sonnet-5) rather than a round number of my own choosing.
    The character heuristic under-counts prose, so clearing the bound here
    means clearing it by a wider margin against the real tokenizer -- the
    authoritative check being verify_prefix_caches() once credentials exist.
    """
    for direction in DIRECTIONS:
        prefix = build_stable_prefix(_persona(), direction)
        assert estimate_prefix_tokens(prefix) >= MAX_MODEL_CACHE_MINIMUM


def test_cache_group_pairs_persona_with_direction() -> None:
    persona = _persona()
    scenario = build_scenarios()[0]
    assembled = assemble(persona, scenario)
    assert persona.persona_id in assembled.cache_group
    assert scenario.direction in assembled.cache_group


# ----------------------------------------------------------------------- grid


def _cells(n_replicates: int = 2) -> list[GridCell]:
    return expand(
        [_persona("r3_trading", 3), _persona("r4_legal", 4, department="Legal")],
        build_scenarios(),
        [("claude-opus-5", "sim_anthropic"), ("gpt-x", "sim_openai")],
        n_replicates,
    )


def test_grid_cardinality_is_the_product_of_its_factors() -> None:
    cells = _cells(n_replicates=5)
    assert len(cells) == 2 * len(build_scenarios()) * 2 * 5


def test_cell_ids_are_unique() -> None:
    """A collision would silently overwrite one cell's result with another's."""
    cells = _cells()
    assert summarize(cells)["n_duplicate_cell_ids"] == 0


def test_cell_ids_are_deterministic() -> None:
    assert [c.cell_id for c in _cells()] == [c.cell_id for c in _cells()]


def test_cache_ordering_groups_shared_prefixes_together() -> None:
    """Scattered groups would re-write an expired prefix for every cell."""
    ordered = order_for_cache(_cells())
    seen: set[str] = set()
    previous: str | None = None
    for cell in ordered:
        if cell.cache_group != previous:
            assert cell.cache_group not in seen, "cache group is not contiguous"
            seen.add(cell.cache_group)
            previous = cell.cache_group


def test_cache_group_separates_models() -> None:
    """Caches are per-model; interleaving models would defeat the grouping."""
    cells = _cells()
    anthropic = {c.cache_group for c in cells if c.model == "claude-opus-5"}
    openai = {c.cache_group for c in cells if c.model == "gpt-x"}
    assert anthropic.isdisjoint(openai)


def test_ordering_preserves_every_cell() -> None:
    cells = _cells()
    assert {c.cell_id for c in order_for_cache(cells)} == {c.cell_id for c in cells}
