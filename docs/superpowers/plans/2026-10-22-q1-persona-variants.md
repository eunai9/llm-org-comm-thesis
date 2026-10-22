# Q1 Persona-Variant Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and analyse a 40-persona Q1 grid (10 role cells x 4 trait variants, named receivers) for four free-tier models, next to the existing 10-persona design.

**Architecture:** Two optional fields (`Persona.trait`, `Scenario.receiver`) let the existing prompt path render a trait line and a named receiver; with both unset, every existing prompt stays byte-identical. A new `sim/persona_variants.py` builds the variants, the allowed sender/receiver pairs and the grid cells. A new `analysis/q1_personas.py` generates a grid through the existing `run_grid` path and analyses a saved grid without calling any model.

**Tech Stack:** Python 3.12, pandas, spaCy, statsmodels, pytest, the project's `thesis.sim` and `thesis.analysis` packages. Run everything inside WSL: `wsl.exe -e bash -c "cd ~/projects/thesis && source .venv/bin/activate && <command>"`.

**Spec:** `docs/superpowers/specs/2026-10-22-q1-persona-variants-design.md`

## Global Constraints

- The 10-persona experiment, its grid files and its cache entries must not change. With `trait=None` and `receiver=None`, rendered prompts are byte-identical to today.
- 4 variants per base persona: tenure in {2, 12} years x origin in {internal, external}. IDs `<base_id>_t<2|12>_<internal|external>`.
- Receiver: same department, rank +1 (up), same rank (lateral), rank -1 (down). Rank 1: lateral and up only. Rank 5: down and lateral only. 26 pairs in total.
- Size per model: pilot 832 cells (8 scenarios per direction), full 4,992 cells (48 per direction).
- Models: `llama3.2:3b` (local), `openai/gpt-oss-20b@low` (NVIDIA), `openai/gpt-oss-120b@low` and `qwen/qwen3.8-27b` (Groq). Free tier only, no paid API.
- Output files: `data/interim/q1_persona_grid_<slug>_<pilot|full>.parquet`. Never commit `data/` or `runs/`.
- Format with `black`, lint with `ruff`, type-check with `mypy`. Type hints on every signature. Docstrings only on public functions, one line unless non-obvious.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. In WSL, write the message to a file and use `git commit -F`.

## Review Focus

- Existing prompts changing by even one byte would silently invalidate every cached 10-persona reply. Golden-string tests pin both default renderings (Tasks 1 and 2).
- Rank 1 asked to write "down" or rank 5 asked to write "up" would produce a receiver that does not exist. Task 3 tests that no such cell is built.
- Variant persona IDs are not keys in the memory snapshot. Without remapping, every variant would silently get no memories. Task 3 tests `variant_stores`.
- Some replies fail validation and are skipped by `run_grid`, so a saved grid can have fewer rows than cells. The analysis must still run. Task 5 tests a grid with missing rows.
- The analysis mode must never call a model. Task 5's entry point takes a saved file only.

---

### Task 1: Optional trait line on a persona

**Files:**
- Modify: `src/thesis/sim/persona.py` (the `Persona` dataclass and `render_persona_block`)
- Test: `tests/test_persona_variants.py` (new file, shared with Tasks 2 and 3)

**Interfaces:**
- Produces: `Persona.trait: str | None = None` (last field). `render_persona_block(persona)` inserts `persona.trait` as its own line right after the "Your role: ..." line when it is set, and is unchanged when it is `None`.

- [ ] **Step 1: Write the failing tests**

```python
from dataclasses import replace

from thesis.sim.persona import Persona, PersonaStyle, render_persona_block

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
    block = render_persona_block(replace(_base(), trait="You joined from another energy firm about 2 years ago."))
    lines = block.splitlines()
    assert lines[1] == "You joined from another energy firm about 2 years ago."
    assert block.replace("You joined from another energy firm about 2 years ago.\n", "") == render_persona_block(_base())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_persona_variants.py -v`
Expected: the first test PASSES (it pins today's text) and the second FAILS with `TypeError: ... unexpected keyword argument 'trait'` from `replace`.

- [ ] **Step 3: Implement**

In `src/thesis/sim/persona.py`, add the last field to `Persona`:

```python
    derivation: Derivation
    trait: str | None = None
```

and change `render_persona_block` to build the list, then insert the trait:

```python
    style = persona.style
    lines = [
        f"Your role: {persona.rank_label}, in the {persona.department} "
        "function of a large US energy company in 2001.",
        "",
        "You are a composite role archetype, not any specific individual. "
        "Write the way someone in this position would write, based on the "
        "following observed tendencies of people at this level:",
        "",
        f"- Typical message length: about {style.mean_tokens:.0f} words.",
        f"- Typical number of recipients: about {style.mean_recipients:.1f}.",
        f"- Gives direct instructions in roughly {style.imperative_ratio:.0%} of sentences.",
        f"- Uses hedging language ('maybe', 'I think') in about {style.hedge_rate:.0%}.",
        f"- Uses deferential phrasing in about {style.deference_rate:.0%}.",
        f"- Asks questions in about {style.question_ratio:.0%} of sentences.",
    ]
    if persona.trait is not None:
        lines.insert(1, persona.trait)
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests, then the full suite**

Run: `python -m pytest tests/test_persona_variants.py -v && python -m pytest tests/ -q -p no:warnings`
Expected: all PASS. The full suite proves no existing prompt test moved.

- [ ] **Step 5: Commit**

```bash
black src/thesis/sim/persona.py tests/test_persona_variants.py && ruff check src/thesis/sim/persona.py tests/test_persona_variants.py && mypy src/thesis/sim/persona.py
git add src/thesis/sim/persona.py tests/test_persona_variants.py
git commit -F /tmp/msg.txt   # "Add an optional trait line to the persona prompt"
```

### Task 2: Optional named receiver on a scenario

**Files:**
- Modify: `src/thesis/sim/scenario.py` (the `Scenario` dataclass and `render_scenario_block`)
- Test: `tests/test_persona_variants.py`

**Interfaces:**
- Produces: `Scenario.receiver: str | None = None` (last field), holding a phrase with its article, e.g. `"a Vice President in Trading"`. `render_scenario_block` writes `f"You are writing to {receiver}."` instead of the `_DIRECTION_FRAMING` sentence when it is set.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_persona_variants.py`)

```python
from thesis.sim.scenario import build_scenarios, render_scenario_block


def test_scenario_block_without_receiver_is_unchanged() -> None:
    scenario = next(s for s in build_scenarios() if s.direction == "up")
    block = render_scenario_block(scenario)
    assert block.splitlines()[3] == "You are writing to someone senior to you."


def test_scenario_block_names_the_receiver() -> None:
    scenario = replace(next(s for s in build_scenarios() if s.direction == "up"), receiver="a Vice President in Trading")
    lines = render_scenario_block(scenario).splitlines()
    assert lines[3] == "You are writing to a Vice President in Trading."
    assert "senior to you" not in "\n".join(lines)
```

- [ ] **Step 2: Run to verify the second test fails**

Run: `python -m pytest tests/test_persona_variants.py -v -k scenario`
Expected: the first PASSES, the second FAILS with `TypeError` on `receiver`.

- [ ] **Step 3: Implement**

In `src/thesis/sim/scenario.py` add to `Scenario`:

```python
    incoming_message: str
    receiver: str | None = None
```

and in `render_scenario_block` replace `_DIRECTION_FRAMING[scenario.direction],` with:

```python
            (
                f"You are writing to {scenario.receiver}."
                if scenario.receiver is not None
                else _DIRECTION_FRAMING[scenario.direction]
            ),
```

- [ ] **Step 4: Run the tests, then the full suite**

Run: `python -m pytest tests/test_persona_variants.py -v && python -m pytest tests/ -q -p no:warnings`
Expected: all PASS.

- [ ] **Step 5: Commit** ("Add an optional named receiver to the scenario prompt"), after black, ruff and mypy on the two files.

### Task 3: Variants, allowed pairs and grid cells

**Files:**
- Create: `src/thesis/sim/persona_variants.py`
- Test: `tests/test_persona_variants.py`

**Interfaces:**
- Consumes: `Persona.trait` (Task 1), `Scenario.receiver` (Task 2), `thesis.sim.grid.GridCell`, `order_for_cache`, `thesis.sim.prompt.prompt_text_hash`, `thesis.sim.persona.RANKS`.
- Produces:
  - `TENURES: tuple[int, int] = (2, 12)`, `ORIGINS: tuple[str, str] = ("internal", "external")`
  - `trait_line(tenure: int, origin: str) -> str`
  - `build_variants(personas: Sequence[Persona]) -> list[Persona]`
  - `variant_traits(persona_id: str) -> tuple[str, int, str]` returning `(base_id, tenure, origin)`
  - `allowed_directions(rank: int) -> tuple[Direction, ...]`
  - `RECEIVER_STEP: dict[str, int] = {"up": 1, "lateral": 0, "down": -1}`
  - `receiver_phrase(rank_labels: Mapping[int, str], rank: int, department: str, direction: str) -> str`
  - `build_variant_cells(variants: Sequence[Persona], scenarios: Sequence[Scenario], *, model: str, role_label: str, rank_labels: Mapping[int, str]) -> list[GridCell]`
  - `variant_stores(stores: Mapping[str, Sequence[MemoryItem]], variants: Sequence[Persona]) -> dict[str, list[MemoryItem]]`
  - `persona_grid_prompt_hash() -> str`

- [ ] **Step 1: Write the failing tests** (append)

```python
import pytest

from thesis.analysis.q1 import build_q1_scenarios
from thesis.sim.memory import MemoryItem
from thesis.sim.persona import load_frozen_personas
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
    pilot = build_variant_cells(variants, build_q1_scenarios(), model="m", role_label="q1p", rank_labels=_LABELS)
    assert len(variants) == 40
    assert len(pilot) == 832
    assert all(c.scenario.receiver is not None for c in pilot)
    assert not any(c.persona.seniority_rank == 1 and c.scenario.direction == "down" for c in pilot)
    assert not any(c.persona.seniority_rank == 5 and c.scenario.direction == "up" for c in pilot)


def test_full_grid_has_4992_cells() -> None:
    from thesis.sim.scenario import build_scenarios

    variants = build_variants(load_frozen_personas())
    cells = build_variant_cells(variants, build_scenarios(), model="m", role_label="q1p", rank_labels=_LABELS)
    assert len(cells) == 4992


def test_variant_stores_reuse_the_base_memory() -> None:
    item = MemoryItem(text="Closed the Q3 review.", importance=5.0, hours_ago=24.0)
    stores = variant_stores({"r3_trading": [item]}, build_variants([_base()]))
    assert stores["r3_trading_t12_external"] == [item]


def test_persona_grid_prompt_hash_differs_from_the_default() -> None:
    assert persona_grid_prompt_hash() != prompt_text_hash()
    assert len(persona_grid_prompt_hash()) == 16
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_persona_variants.py -v`
Expected: collection ERROR, `ModuleNotFoundError: thesis.sim.persona_variants`.

- [ ] **Step 3: Implement `src/thesis/sim/persona_variants.py`**

```python
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


def receiver_phrase(rank_labels: Mapping[int, str], rank: int, department: str, direction: str) -> str:
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
                s, receiver=receiver_phrase(rank_labels, v.seniority_rank, v.department, s.direction)
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
```

Note: `Scenario.receiver` is rendered by Task 2 with the same sentence as `_RECEIVER_TEMPLATE`. If either changes, change both.

- [ ] **Step 4: Run the tests, then the full suite**

Run: `python -m pytest tests/test_persona_variants.py -v && python -m pytest tests/ -q -p no:warnings`
Expected: all PASS.

- [ ] **Step 5: Commit** ("Build persona variants, receiver pairs and grid cells"), after black, ruff and mypy.

### Task 4: Generate a persona grid

**Files:**
- Create: `src/thesis/analysis/q1_personas.py`
- Test: `tests/test_q1_personas.py`

**Interfaces:**
- Consumes: everything from Task 3; `thesis.analysis.q1.scenarios_for_design`, `Q1Design`; `thesis.sim.run.run_grid`, `RunManifest`; `thesis.analysis.judge_swap.client_for_model`.
- Produces:
  - `RANK_LABELS: dict[int, str]` built from the frozen personas.
  - `generate_persona_grid(client: LLMClient, *, model: str, design: Q1Design, personas: Sequence[Persona] | None = None, stores: Mapping[str, Sequence[MemoryItem]] | None = None, cache: ResponseCache | None = None, ledger: CostLedger | None = None, limit: int | None = None, progress_every: int = 100) -> pd.DataFrame` with the `run_grid` row columns plus `base_persona_id`, `tenure`, `origin`, `receiver_rank`, `prompt_text_hash`.
  - CLI: `python -m thesis.analysis.q1_personas --model MODEL --design pilot|full --out PATH [--limit N] [--cache-only] [--ollama-host URL]`.

- [ ] **Step 1: Write the failing test**

```python
from collections.abc import Sequence
from pathlib import Path

from thesis.analysis.q1_personas import generate_persona_grid
from thesis.llm.base import Capabilities, CompletionRequest, CompletionResponse, Provider, Usage
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.sim.persona_variants import persona_grid_prompt_hash


class FakeClient:
    """Answers every cell with one valid reply and never touches the network."""

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
        return CompletionResponse(
            text="{}",
            usage=Usage(input_tokens=200, output_tokens=20),
            model=f"local/{request.model}",
            stop_reason="end_turn",
            parsed={
                "subject": "Re: update",
                "body": "Send me the numbers by Friday.",
                "decision": "accept",
                "confidence": "medium",
                "reasoning_brief": "Routine.",
            },
        )

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        raise NotImplementedError

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        raise NotImplementedError


def test_generate_persona_grid_adds_variant_columns(tmp_path: Path) -> None:
    client = FakeClient()
    frame = generate_persona_grid(
        client,
        model="llama3.2:3b",
        design="pilot",
        stores={},
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        limit=12,
    )
    assert len(frame) == 12
    for col in ("base_persona_id", "tenure", "origin", "receiver_rank", "prompt_text_hash"):
        assert col in frame.columns
    assert (frame["prompt_text_hash"] == persona_grid_prompt_hash()).all()
    assert (frame["receiver_rank"] - frame["seniority_rank"]).isin([-1, 0, 1]).all()
    prompt = repr(client.calls[0])
    assert "You are writing to a" in prompt or "You are writing to an" in prompt
    assert "years" in prompt
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_q1_personas.py -v`
Expected: collection ERROR, `ModuleNotFoundError: thesis.analysis.q1_personas`.

- [ ] **Step 3: Implement the generation part of `src/thesis/analysis/q1_personas.py`**

```python
"""The second Q1 experiment: 40 persona variants writing to named receivers.

Generation goes through :func:`thesis.sim.run.run_grid`, the same path as
every other grid. The analysis mode reads a saved grid and calls no model.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from thesis.analysis.judge_swap import client_for_model
from thesis.analysis.q1 import Q1Design, scenarios_for_design
from thesis.config import load_config
from thesis.llm.base import LLMClient
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import CACHE_DIR, COST_LEDGER, ensure_dirs
from thesis.sim.memory import MemoryItem
from thesis.sim.memory_generation import load_frozen_memory
from thesis.sim.persona import Persona, load_frozen_personas
from thesis.sim.persona_variants import (
    RECEIVER_STEP,
    build_variant_cells,
    build_variants,
    persona_grid_prompt_hash,
    variant_stores,
    variant_traits,
)
from thesis.sim.run import RunManifest, run_grid

log = get_logger(__name__)

ROLE_LABEL = "sim_q1_personas"


def generate_persona_grid(
    client: LLMClient,
    *,
    model: str,
    design: Q1Design,
    personas: Sequence[Persona] | None = None,
    stores: Mapping[str, Sequence[MemoryItem]] | None = None,
    cache: ResponseCache | None = None,
    ledger: CostLedger | None = None,
    limit: int | None = None,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Generate (or load from cache) one model's persona-variant grid."""
    base = list(personas) if personas is not None else load_frozen_personas()
    stores = stores if stores is not None else load_frozen_memory()
    cache = cache if cache is not None else ResponseCache(CACHE_DIR)
    ledger = ledger if ledger is not None else CostLedger(COST_LEDGER)
    rank_labels = {p.seniority_rank: p.rank_label for p in base}
    variants = build_variants(base)
    cells = build_variant_cells(
        variants, scenarios_for_design(design), model=model, role_label=ROLE_LABEL, rank_labels=rank_labels
    )
    if limit is not None:
        cells = cells[:limit]
    prompt_hash = persona_grid_prompt_hash()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = RunManifest(
        run_id=run_id,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit="",
        git_dirty=False,
        config_hash="",
        models=[model],
        design={"kind": "q1_persona_grid", "design": design, "prompt_text_hash": prompt_hash},
        n_cells=len(cells),
    )
    rows: list[dict[str, Any]] = run_grid(
        cells,
        client,
        variant_stores(stores, variants),
        load_config(),
        run_id=run_id,
        cache=cache,
        ledger=ledger,
        manifest=manifest,
        progress_every=progress_every,
    )
    frame = pd.DataFrame.from_records(rows)
    traits = frame["persona_id"].map(variant_traits)
    frame["base_persona_id"] = traits.str[0]
    frame["tenure"] = traits.str[1]
    frame["origin"] = traits.str[2]
    frame["receiver_rank"] = frame["seniority_rank"] + frame["direction"].map(RECEIVER_STEP)
    frame["prompt_text_hash"] = prompt_hash
    log.info("%s: %d of %d cells returned a valid reply", model, len(frame), len(cells))
    return frame
```

And the generation CLI at the end of the file:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="e.g. qwen/qwen3.8-27b or llama3.2:3b")
    parser.add_argument("--design", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--ollama-host", default=None)
    args = parser.parse_args()
    configure_logging()
    ensure_dirs()
    frame = generate_persona_grid(
        client_for_model(args.model, ollama_host=args.ollama_host),
        model=args.model,
        design=args.design,
        cache=ResponseCache(CACHE_DIR, cache_only=args.cache_only),
        limit=args.limit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then the full suite**

Run: `python -m pytest tests/test_q1_personas.py -v && python -m pytest tests/ -q -p no:warnings`
Expected: all PASS.

- [ ] **Step 5: Smoke test against the real stack, no model calls**

Run: `python -m thesis.analysis.q1_personas --model llama3.2:3b --design pilot --limit 3 --cache-only --out /tmp/q1p_smoke.parquet`
Expected: it fails with a cache-miss error (nothing for these prompts is cached yet). That confirms `--cache-only` blocks live calls. Then run without `--cache-only` and `--limit 3` against WSL Ollama (`tmux new-session -d -s wsl_ollama -c ~ '~/.local/bin/ollama serve 2>&1 | tee -a ~/ollama_serve.log'` if it is not running) and check that 3 rows are written.

- [ ] **Step 6: Commit** ("Generate the Q1 persona-variant grid"), after black, ruff and mypy.

### Task 5: Analyse a saved persona grid

**Files:**
- Modify: `src/thesis/analysis/q1_personas.py` (add analysis function and a second CLI mode)
- Test: `tests/test_q1_personas.py`

**Interfaces:**
- Consumes: `thesis.analysis.q1.parse_replies`, `extract_q1_sentence_features`; `thesis.analysis.hierarchy.fit_direction_fixed_effects`; `thesis.analysis.role_cues.reply_cues`; `thesis.analysis.q1_measures.load_gilbert_phrases`, `GILBERT_PATH`.
- Produces:
  - `REPLY_OUTCOMES: tuple[str, ...] = ("imperative_ratio", "hedge_rate", "you_rate", "gratitude", "gilbert_score")`
  - `analyse_persona_grid(frame: pd.DataFrame, phrases: Mapping[tuple[str, ...], float]) -> dict[str, Any]` with keys `n_replies`, `n_personas`, `mean_sentences_per_reply`, `primary` (sentence-level `is_imperative`, logistic, persona fixed effects, clustered by persona), `reply_level` (one entry per outcome in `REPLY_OUTCOMES` that varies in the grid), `traits` (tenure and origin coefficients on `imperative_ratio`). Every contrast entry is `{"coefficient", "se", "p"}` for `down` and `up` against `lateral`.
  - CLI mode: `python -m thesis.analysis.q1_personas --analyse PATH --manifest PATH` (no model call).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_q1_personas.py`)

```python
import pandas as pd

from thesis.analysis.q1_personas import REPLY_OUTCOMES, analyse_persona_grid

_BODIES = {
    "down": ["Send the report today. Call me after.", "Could you send the report?"],
    "lateral": ["Let's talk about it this week.", "Send it when it is ready."],
    "up": ["Could you look at this when you have time?", "I think we might need more time."],
}


def _synthetic_grid(drop: int = 0) -> pd.DataFrame:
    rows = []
    for p in range(8):
        tenure, origin = (2, 12)[p % 2], ("internal", "external")[(p // 2) % 2]
        pid = f"r3_trading_t{tenure}_{origin}_{p}"
        for direction, bodies in _BODIES.items():
            for i, body in enumerate(bodies * 3):
                rows.append(
                    {
                        "cell_id": f"{pid}__{direction}__{i}",
                        "persona_id": pid,
                        "base_persona_id": "r3_trading",
                        "tenure": tenure,
                        "origin": origin,
                        "direction": direction,
                        "scenario_id": f"approve_or_decline__{direction}__high__neutral",
                        "stakes": "high",
                        "task_type": "approve_or_decline",
                        "replicate": 1,
                        "body": body,
                    }
                )
    frame = pd.DataFrame(rows)
    return frame.iloc[drop:].reset_index(drop=True)


def test_analysis_finds_more_orders_writing_down() -> None:
    result = analyse_persona_grid(_synthetic_grid(), {("thank", "you"): -1.0})
    assert result["n_personas"] == 8
    assert result["primary"]["down"]["coefficient"] > 0
    # No synthetic body thanks anyone, so gratitude never varies and is skipped.
    assert set(result["reply_level"]) == set(REPLY_OUTCOMES) - {"gratitude"}
    assert {"tenure", "origin"} <= set(result["traits"])


def test_analysis_runs_with_missing_rows() -> None:
    result = analyse_persona_grid(_synthetic_grid(drop=5), {})
    assert result["n_replies"] == len(_synthetic_grid()) - 5
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_q1_personas.py -v -k analysis`
Expected: FAIL with `ImportError: cannot import name 'analyse_persona_grid'`.

- [ ] **Step 3: Implement** (add to `src/thesis/analysis/q1_personas.py`)

New imports:

```python
import json

import statsmodels.formula.api as smf

from thesis.analysis.hierarchy import FixedEffectsResult, InsufficientDataError, fit_direction_fixed_effects
from thesis.analysis.q1 import extract_q1_sentence_features, parse_replies
from thesis.analysis.q1_measures import GILBERT_PATH, load_gilbert_phrases
from thesis.analysis.role_cues import reply_cues
```

Code:

```python
REPLY_OUTCOMES: tuple[str, ...] = ("imperative_ratio", "hedge_rate", "you_rate", "gratitude", "gilbert_score")


def _contrasts(fit: FixedEffectsResult) -> dict[str, dict[str, float]]:
    out = {}
    for level in ("down", "up"):
        coefficient, p = fit.contrast(level)
        se = next(v for k, v in fit.std_errors.items() if k.endswith(f"[T.{level}]"))
        out[level] = {"coefficient": round(coefficient, 4), "se": round(se, 4), "p": round(p, 6)}
    return out


def analyse_persona_grid(frame: pd.DataFrame, phrases: Mapping[tuple[str, ...], float]) -> dict[str, Any]:
    """Direction contrasts with persona fixed effects, clustered by persona."""
    docs = parse_replies(frame)
    sentences = extract_q1_sentence_features(frame, docs=docs)
    primary = fit_direction_fixed_effects(sentences, "is_imperative", family="logistic")

    cues = pd.DataFrame([reply_cues(body, docs[cid], phrases) for cid, body in zip(frame["cell_id"], frame["body"], strict=True)])
    reply = pd.concat([frame[["persona_id", "tenure", "origin", "direction", "base_persona_id"]].reset_index(drop=True), cues], axis=1)
    reply_level: dict[str, Any] = {}
    for outcome in REPLY_OUTCOMES:
        if reply[outcome].std() == 0:
            continue
        try:
            reply_level[outcome] = _contrasts(fit_direction_fixed_effects(reply, outcome))
        except InsufficientDataError:
            continue

    working = reply.assign(tenure=reply["tenure"].astype(str))
    trait_fit = smf.ols(
        "imperative_ratio ~ C(direction, Treatment(reference='lateral')) + C(tenure) + C(origin) + C(base_persona_id)",
        data=working,
    ).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(working["persona_id"])[0]})
    traits = {}
    for name in ("tenure", "origin"):
        term = next(t for t in trait_fit.params.index if t.startswith(f"C({name})"))
        traits[name] = {
            "term": term,
            "coefficient": round(float(trait_fit.params[term]), 4),
            "p": round(float(trait_fit.pvalues[term]), 6),
        }

    return {
        "n_replies": int(len(frame)),
        "n_personas": int(frame["persona_id"].nunique()),
        "primary": _contrasts(primary),
        "reply_level": reply_level,
        "traits": traits,
        "mean_sentences_per_reply": round(float(len(sentences) / max(len(frame), 1)), 2),
    }
```

Extend `main()`: make `--model` and `--analyse` a required mutually exclusive group, add `--manifest` (required with `--analyse`), and branch:

```python
    if args.analyse is not None:
        frame = pd.read_parquet(args.analyse)
        result = analyse_persona_grid(frame, load_gilbert_phrases(GILBERT_PATH))
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("wrote %s", args.manifest)
        return
```

- [ ] **Step 4: Run the tests, then the full suite, then ruff, black and mypy**

Run: `python -m pytest tests/test_q1_personas.py -v && python -m pytest tests/ -q -p no:warnings && black --check src tests && ruff check src/thesis/analysis/q1_personas.py tests/test_q1_personas.py && mypy src/thesis/analysis/q1_personas.py`
Expected: all PASS / clean.

- [ ] **Step 5: Commit** ("Analyse a saved persona-variant grid").

### Task 6: Pilot runs on all four models

**Files:**
- Create: `~/q1_personas_runner.sh` (outside the repo, like the other runners)
- Output (not committed): `data/interim/q1_persona_grid_<slug>_pilot.parquet`
- Output (committed): `outputs/manifests/q1_personas_<slug>_pilot.json`

**Interfaces:**
- Consumes: the CLI from Tasks 4 and 5.
- Model slugs: `llama` for `llama3.2:3b`, `gpt_oss_20b` for `openai/gpt-oss-20b@low`, `gpt_oss_120b` for `openai/gpt-oss-120b@low`, `qwen` for `qwen/qwen3.8-27b`.

- [ ] **Step 1: Write the runner** (one model per tmux session, retried until exit 0, same backoff pattern as `~/qwen_q2q3_runner.sh`)

```bash
#!/bin/bash
# Usage: q1_personas_runner.sh MODEL SLUG DESIGN
cd ~/projects/thesis
source .venv/bin/activate
set -a; source .env; set +a
MODEL="$1"; SLUG="$2"; DESIGN="$3"
OUT="data/interim/q1_persona_grid_${SLUG}_${DESIGN}.parquet"
backoff=300
until python -m thesis.analysis.q1_personas --model "$MODEL" --design "$DESIGN" --out "$OUT"; do
  echo "=== $SLUG: failed at $(date), cooling down ${backoff}s ==="
  sleep "$backoff"
  backoff=$((backoff * 2)); if [ "$backoff" -gt 1800 ]; then backoff=1800; fi
done
python -m thesis.analysis.q1_personas --analyse "$OUT" --manifest "outputs/manifests/q1_personas_${SLUG}_${DESIGN}.json"
echo "=== $SLUG $DESIGN finished $(date) ==="
```

Write it with the Write tool to the scratchpad, then copy it to `~/q1_personas_runner.sh` with `tr -d '\r'` and `chmod +x`.

- [ ] **Step 2: Start WSL Ollama if needed, then launch four tmux sessions**

```bash
tmux new-session -d -s q1p_llama -c ~/projects/thesis 'bash ~/q1_personas_runner.sh llama3.2:3b llama pilot 2>&1 | tee -a logs/q1p_llama_pilot.log'
tmux new-session -d -s q1p_gpt_oss_20b -c ~/projects/thesis 'bash ~/q1_personas_runner.sh openai/gpt-oss-20b@low gpt_oss_20b pilot 2>&1 | tee -a logs/q1p_gpt_oss_20b_pilot.log'
tmux new-session -d -s q1p_gpt_oss_120b -c ~/projects/thesis 'bash ~/q1_personas_runner.sh openai/gpt-oss-120b@low gpt_oss_120b pilot 2>&1 | tee -a logs/q1p_gpt_oss_120b_pilot.log'
tmux new-session -d -s q1p_qwen -c ~/projects/thesis 'bash ~/q1_personas_runner.sh qwen/qwen3.8-27b qwen pilot 2>&1 | tee -a logs/q1p_qwen_pilot.log'
```

Check first the existing Groq jobs (`tmux ls`): the qwen Q2/Q3 runner also uses Groq. Two Groq jobs on the same model slow each other down; that is expected, not an error.

- [ ] **Step 3: When each run finishes, report the pilot gate**

For each model: valid replies out of 832, the primary down/up contrasts, the trait coefficients. Put each primary contrast next to the same model's 10-persona contrast (`outputs/manifests/q1_models_full.json`, `models.<label>.primary` for down; the up contrast from the persona-clustered fit in section 13) and next to real email's robust fit (`real` in the same manifest). Write this into `PROGRESS_llms.md` section 19 (a short "Pilot" block per model, plain sentences), commit the manifests and the log edit, and push. Then ask the user whether to run the full design (`DESIGN=full` with the same runner).
