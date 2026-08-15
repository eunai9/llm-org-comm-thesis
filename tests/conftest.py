"""Fixtures shared across test modules.

Lives here rather than being imported between test files: cross-importing test
modules makes the type checker see the same file under two module names, and
it couples unrelated suites together.
"""

from __future__ import annotations

import pytest

from thesis.sim.grid import GridCell, expand, order_for_cache
from thesis.sim.persona import Persona, PersonaStyle
from thesis.sim.run import RunManifest
from thesis.sim.scenario import build_scenarios


def make_persona(persona_id: str = "r3_trading") -> Persona:
    return Persona(
        persona_id=persona_id,
        seniority_rank=3,
        rank_label="Director",
        department="Trading",
        style=PersonaStyle(
            mean_tokens=50.0,
            mean_recipients=2.0,
            imperative_ratio=0.15,
            hedge_rate=0.03,
            deference_rate=0.005,
            question_ratio=0.09,
        ),
        n_people=25,
        n_messages=4236,
        derivation="cell",
    )


def make_cells(n: int = 6) -> list[GridCell]:
    cells = order_for_cache(
        expand([make_persona()], build_scenarios()[:3], [("claude-opus-5", "sim_anthropic")], 2)
    )
    return cells[:n]


def make_manifest() -> RunManifest:
    return RunManifest(
        run_id="testrun",
        started_at="2026-08-15T00:00:00+00:00",
        git_commit="abc123",
        git_dirty=False,
        config_hash="deadbeef",
        models=["claude-opus-5"],
    )


@pytest.fixture
def persona() -> Persona:
    return make_persona()


@pytest.fixture
def cells() -> list[GridCell]:
    return make_cells()
