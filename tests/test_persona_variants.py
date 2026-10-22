from dataclasses import replace

import pytest

from thesis.analysis.q1 import build_q1_scenarios
from thesis.sim.memory import MemoryItem
from thesis.sim.persona import Persona, PersonaStyle, load_frozen_personas, render_persona_block
from thesis.sim.persona_variants import (
    allowed_directions,
    build_variant_cells,
    build_variants,
    persona_grid_prompt_hash,
    receiver_phrase,
    variant_stores,
    variant_traits,
)
from thesis.sim.prompt import prompt_text_hash
from thesis.sim.scenario import build_scenarios, render_scenario_block

_STYLE = PersonaStyle(
    mean_tokens=60.0,
    mean_recipients=2.0,
    imperative_ratio=0.15,
    hedge_rate=0.03,
    deference_rate=0.01,
    question_ratio=0.1,
)


def _base(rank: int = 3, department: str = "Trading") -> Persona:
    return Persona(
        persona_id=f"r{rank}_{department.lower()}",
        seniority_rank=rank,
        rank_label="Director",
        department=department,
        style=_STYLE,
        n_people=10,
        n_messages=100,
        derivation="cell",
    )


def test_persona_block_without_trait_is_unchanged() -> None:
    assert render_persona_block(_base()) == "\n".join(
        [
            "Your role: Director, in the Trading function of a large US energy company in 2001.",
            "",
            "You are a composite role archetype, not any specific individual. "
            "Write the way someone in this position would write, based on the "
            "following observed tendencies of people at this level:",
            "",
            "- Typical message length: about 60 words.",
            "- Typical number of recipients: about 2.0.",
            "- Gives direct instructions in roughly 15% of sentences.",
            "- Uses hedging language ('maybe', 'I think') in about 3%.",
            "- Uses deferential phrasing in about 1%.",
            "- Asks questions in about 10% of sentences.",
        ]
    )


def test_persona_block_puts_the_trait_after_the_role_line() -> None:
    block = render_persona_block(
        replace(_base(), trait="You joined from another energy firm about 2 years ago.")
    )
    lines = block.splitlines()
    assert lines[1] == "You joined from another energy firm about 2 years ago."
    assert block.replace(
        "You joined from another energy firm about 2 years ago.\n", ""
    ) == render_persona_block(_base())


def test_scenario_block_without_receiver_is_unchanged() -> None:
    scenario = next(s for s in build_scenarios() if s.direction == "up")
    block = render_scenario_block(scenario)
    assert block.splitlines()[3] == "You are writing to someone senior to you."


def test_scenario_block_names_the_receiver() -> None:
    scenario = replace(
        next(s for s in build_scenarios() if s.direction == "up"),
        receiver="a Vice President in Trading",
    )
    lines = render_scenario_block(scenario).splitlines()
    assert lines[3] == "You are writing to a Vice President in Trading."
    assert "senior to you" not in "\n".join(lines)


_LABELS = {1: "Employee", 2: "Manager", 3: "Director", 4: "Vice President", 5: "Managing Director"}


def test_four_variants_per_base_with_unchanged_style() -> None:
    variants = build_variants([_base()])
    assert [v.persona_id for v in variants] == [
        "r3_trading_t2_internal",
        "r3_trading_t2_external",
        "r3_trading_t12_internal",
        "r3_trading_t12_external",
    ]
    assert all(v.style == _base().style and v.seniority_rank == 3 for v in variants)
    assert len({v.trait for v in variants}) == 4


def test_variant_traits_round_trip() -> None:
    assert variant_traits("r3_trading_t12_external") == ("r3_trading", 12, "external")


def test_edge_ranks_have_two_directions() -> None:
    assert allowed_directions(1) == ("up", "lateral")
    assert allowed_directions(5) == ("lateral", "down")
    assert allowed_directions(3) == ("up", "lateral", "down")


def test_receiver_phrase_uses_the_right_rank_and_article() -> None:
    assert receiver_phrase(_LABELS, 3, "Trading", "up") == "a Vice President in Trading"
    assert receiver_phrase(_LABELS, 2, "Legal", "down") == "an Employee in Legal"
    with pytest.raises(ValueError):
        receiver_phrase(_LABELS, 5, "Legal", "up")


def test_grid_sizes_match_the_spec() -> None:
    variants = build_variants(load_frozen_personas())
    pilot = build_variant_cells(
        variants, build_q1_scenarios(), model="m", role_label="q1p", rank_labels=_LABELS
    )
    assert len(variants) == 40
    assert len(pilot) == 832
    assert all(c.scenario.receiver is not None for c in pilot)
    assert not any(c.persona.seniority_rank == 1 and c.scenario.direction == "down" for c in pilot)
    assert not any(c.persona.seniority_rank == 5 and c.scenario.direction == "up" for c in pilot)


def test_full_grid_has_4992_cells() -> None:
    variants = build_variants(load_frozen_personas())
    cells = build_variant_cells(
        variants, build_scenarios(), model="m", role_label="q1p", rank_labels=_LABELS
    )
    assert len(cells) == 4992


def test_variant_stores_reuse_the_base_memory() -> None:
    item = MemoryItem(text="Closed the Q3 review.", importance=5.0, hours_ago=24.0)
    stores = variant_stores({"r3_trading": [item]}, build_variants([_base()]))
    assert stores["r3_trading_t12_external"] == [item]


def test_persona_grid_prompt_hash_differs_from_the_default() -> None:
    assert persona_grid_prompt_hash() != prompt_text_hash()
    assert len(persona_grid_prompt_hash()) == 16
