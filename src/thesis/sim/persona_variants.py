"""Persona variants for the second Q1 experiment.

Each of the 10 role personas gets four variants from the same 2x2 set of
traits (tenure x origin). The traits are personal history, not instructions
about style, so rank stays the only thing tied to the outcome measures. Each
variant writes to a named receiver one rank up, at the same rank, or one
rank down, in its own department.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Final

from thesis.sim.grid import GridCell, order_for_cache
from thesis.sim.memory import MemoryItem
from thesis.sim.persona import RANKS, Persona
from thesis.sim.prompt import prompt_text_hash
from thesis.sim.scenario import DIRECTIONS, Direction, Scenario

TENURES: Final[tuple[int, int]] = (2, 12)
ORIGINS: Final[tuple[str, str]] = ("internal", "external")
RECEIVER_STEP: Final[dict[str, int]] = {"up": 1, "lateral": 0, "down": -1}
_TRAIT_TEMPLATES: Final[dict[str, str]] = {
    "internal": "You were promoted from within and have worked at the company for about {tenure} years.",
    "external": "You joined from another energy firm about {tenure} years ago.",
}
_RECEIVER_TEMPLATE: Final[str] = "You are writing to {receiver}."


def trait_line(tenure: int, origin: str) -> str:
    """The one-sentence trait a variant adds to its persona block."""
    return _TRAIT_TEMPLATES[origin].format(tenure=tenure)


def build_variants(personas: Sequence[Persona]) -> list[Persona]:
    """Four variants per persona, in tenure-then-origin order."""
    return [
        replace(
            persona,
            persona_id=f"{persona.persona_id}_t{tenure}_{origin}",
            trait=trait_line(tenure, origin),
        )
        for persona in personas
        for tenure in TENURES
        for origin in ORIGINS
    ]


def variant_traits(persona_id: str) -> tuple[str, int, str]:
    """Split a variant id into its base id, tenure and origin."""
    base_id, tenure, origin = persona_id.rsplit("_", 2)
    return base_id, int(tenure.removeprefix("t")), origin


def allowed_directions(rank: int) -> tuple[Direction, ...]:
    """The directions whose receiver rank exists for a sender of ``rank``."""
    return tuple(d for d in DIRECTIONS if min(RANKS) <= rank + RECEIVER_STEP[d] <= max(RANKS))


def receiver_phrase(
    rank_labels: Mapping[int, str], rank: int, department: str, direction: str
) -> str:
    """The named receiver, with its article, e.g. 'a Vice President in Trading'."""
    if direction not in allowed_directions(rank):
        msg = f"a rank-{rank} sender has no receiver for direction {direction!r}"
        raise ValueError(msg)
    label = rank_labels[rank + RECEIVER_STEP[direction]]
    article = "an" if label[0].lower() in "aeiou" else "a"
    return f"{article} {label} in {department}"


def build_variant_cells(
    variants: Sequence[Persona],
    scenarios: Sequence[Scenario],
    *,
    model: str,
    role_label: str,
    rank_labels: Mapping[int, str],
) -> list[GridCell]:
    """One cell per variant x scenario whose direction the variant can write in."""
    cells = [
        GridCell(
            cell_id=f"{role_label}__{v.persona_id}__{s.scenario_id}__r1",
            persona=v,
            scenario=replace(
                s,
                receiver=receiver_phrase(rank_labels, v.seniority_rank, v.department, s.direction),
            ),
            replicate=1,
            model=model,
            role_label=role_label,
        )
        for v in variants
        for s in scenarios
        if s.direction in allowed_directions(v.seniority_rank)
    ]
    return order_for_cache(cells)


def variant_stores(
    stores: Mapping[str, Sequence[MemoryItem]], variants: Sequence[Persona]
) -> dict[str, list[MemoryItem]]:
    """Each variant reads its base persona's memory store."""
    return {v.persona_id: list(stores.get(variant_traits(v.persona_id)[0], [])) for v in variants}


def persona_grid_prompt_hash() -> str:
    """The default template hash plus the trait and receiver templates."""
    blob = "\x00".join([prompt_text_hash(), *_TRAIT_TEMPLATES.values(), _RECEIVER_TEMPLATE])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
