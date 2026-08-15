"""Expansion of the experiment grid into individual, addressable cells.

Every generated response in the study is one cell here. Two properties matter
enough to be tested rather than assumed:

**Cell ids are deterministic and content-free.** An id is built from the
factors that define the cell, so the same design always yields the same ids.
That is what lets a run be resumed, a single cell be re-run, and a result in a
December table be traced back to the exact combination that produced it --
without depending on list position, which changes the moment the design does.

**Run order follows the cache, not the nesting.** Cells sharing a persona and
direction share a cached prompt prefix, but a cache entry expires in minutes.
Iterating in the natural nested-loop order would scatter cells from the same
group across the whole run, so each would re-write a prefix that had already
expired -- paying the write premium every time and reading almost never.
Sorting by ``cache_group`` keeps them adjacent, which is the difference between
the cache working and merely existing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

from thesis.sim.persona import Persona
from thesis.sim.scenario import Scenario


@dataclass(frozen=True, slots=True)
class GridCell:
    """One generated response: persona x scenario x replicate x model."""

    cell_id: str
    persona: Persona
    scenario: Scenario
    replicate: int
    model: str
    role_label: str

    @property
    def cache_group(self) -> str:
        """Cells sharing this string share a cachable prompt prefix.

        The model is part of the group because prompt caches are per-model: the
        same prefix sent to a different model is a different cache entry, so
        interleaving models would defeat the grouping.
        """
        return f"{self.model}__{self.persona.persona_id}__{self.scenario.direction}"


def expand(
    personas: Sequence[Persona],
    scenarios: Sequence[Scenario],
    models: Sequence[tuple[str, str]],
    n_replicates: int,
) -> list[GridCell]:
    """Expand the full grid.

    ``models`` is ``(model_id, role_label)`` so a result can name the model
    family it came from without re-deriving it from the id string -- the
    judge-swap analysis in Q3 keys on exactly that.
    """
    cells = []
    for persona, scenario, (model, role_label), replicate in product(
        personas, scenarios, models, range(1, n_replicates + 1)
    ):
        cells.append(
            GridCell(
                cell_id=(
                    f"{role_label}__{persona.persona_id}__" f"{scenario.scenario_id}__r{replicate}"
                ),
                persona=persona,
                scenario=scenario,
                replicate=replicate,
                model=model,
                role_label=role_label,
            )
        )
    return cells


def order_for_cache(cells: Sequence[GridCell]) -> list[GridCell]:
    """Sort so cells sharing a prompt prefix run consecutively.

    Within a group the order is by cell id, so a run is fully deterministic and
    two runs of the same design submit requests in the same sequence.
    """
    return sorted(cells, key=lambda c: (c.cache_group, c.cell_id))


def summarize(cells: Sequence[GridCell]) -> dict[str, object]:
    """Design counts, for the run manifest and the methods section."""
    groups = {c.cache_group for c in cells}
    return {
        "n_cells": len(cells),
        "n_personas": len({c.persona.persona_id for c in cells}),
        "n_scenarios": len({c.scenario.scenario_id for c in cells}),
        "n_models": len({c.model for c in cells}),
        "n_replicates": len({c.replicate for c in cells}),
        "n_cache_groups": len(groups),
        "mean_cells_per_cache_group": round(len(cells) / len(groups), 1) if groups else 0.0,
        "n_duplicate_cell_ids": len(cells) - len({c.cell_id for c in cells}),
    }
