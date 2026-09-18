"""Q3, the judge-swap self-preference pilot, as a reusable module.

Section 23 (PROGRESS.md) approximated the plan's Q3 design -- does a judge
favor its own kind of AI over another? -- with two free, local models
standing in for two "families": **llama3.2:3b** (Meta) and **qwen2.5:3b**
(Alibaba). 120 replies were generated (10 personas x 3 directions x 2 task
types x both models as generator), each blind-scored by both models acting
as judge (240 scores), and the self-preference question was fit as the
interaction term of a ``generator x judge`` model -- exactly
:func:`thesis.analysis.hierarchy.fit_interaction_model`, reused rather than
rebuilt. That pilot found a self-preference interaction of +0.42 (p=.065),
just short of significance, weaker still (p=.20) on ``corpus_plausibility``
alone.

Like every earlier judge-swap run, the code that produced it was never
committed. This module reconstructs that design and makes it a real entry
point: ``python -m thesis.analysis.judge_swap`` generates (or serves from
cache) the same 120-reply grid against whichever personas the caller
currently has, scores every reply with both models as judge, fits the same
interaction model, and prints the section-23 numbers next to today's.

**How the design was recovered.** No commit ever recorded the exact
scenario subset (section 23's prose names counts -- "10 personas x 3
directions x 2 task types x both models" -- but not which two task types,
nor which tone or stakes level). It was recovered from the local response
cache still on this machine: every ``qwen2.5:3b`` call cached on
2026-08-24 (60 of them, and only ever 60 on that date) decodes to exactly
six scenarios -- ``approve_or_decline`` at high stakes and
``report_problem`` at routine stakes (the same pinning
:data:`thesis.analysis.q1.Q1_TASK_STAKES` uses), each x 3 directions, all
at **neutral** tone only -- x 10 personas x 1 replicate = 60. The
``llama3.2:3b`` half is not separately verifiable this way (its Aug-24
cache entries are dominated by that day's separate Q1 rerun and use
different, since-superseded persona text), but section 23's own count
(120 = 60 + 60, "both models") leaves no other way to split it, and
:func:`build_judge_swap_cells` reproduces exactly 60 cells per generator
model with the current 10 personas -- checked by a test before treating
the design as final, the same discipline ``q1.py`` used.

**What is new here, not just re-run.** The corpus was rebuilt in section 37
(quote-stripping, Lotus Notes quoting, an over-aggressive signature
stripper, all fixed in ``thesis.data.rfc822``), which changed
``n_tokens_clean`` per message, which changed persona style statistics
(``derive_personas``), which changed the rendered prompt text every cell's
cache key depends on. So the 120 replies behind section 23 no longer
reflect the current corpus, and generating against today's personas is a
fresh run, not a cache-hit replay of old data.

**Why two generation passes, not one.**
:class:`~thesis.llm.ollama_client.OllamaClient` sends the model it was
*constructed* with, not ``request.model`` -- correct for every earlier
caller (``q1.py``, ``pairs.py``, ``sim.run``), each of which only ever
asks one local model to generate. The judge-swap design needs two
different generator models in the same 120-reply grid, so
:func:`generate_judge_swap_grid` is called once per generator model (60
cells each) and the two halves are concatenated, rather than writing a
second, dispatching client only this module would need. Scoring has the
same shape: :func:`score_judge_swap_replies` is called once per judge
model over the full 120 replies.

**The two-tone design, and why section 41's replies are stale.** Section 52
re-runs this at twice the size. The design flag ``two_tone`` adds the
assertive tone next to the neutral one, which doubles the grid to 12
scenarios, 120 cells per generator, 240 replies and 480 judge calls. The
neutral-only half of that run is the same design section 41 reports, so one
run gives both numbers. The re-run was needed for a second reason: commit
``8f8df1e`` (2026-09-05) added a paragraph to ``TASK_FRAMING`` after section
41's replies were generated on 2026-09-04, and the cache is keyed on prompt
text, so every section-41 reply came from a prompt that no longer exists.
:func:`thesis.sim.prompt.prompt_text_hash` now goes into this module's run
manifest, so a later run can see that rather than having to work it out.

**Why the "old" half of the comparison is a plain 2x2 decomposition, not a
mixed-model refit.** Section 23 archived its four cell means (the
numbers behind ``docs/figures/judge_swap_interaction.png``,
:data:`HISTORICAL_CELL_MEANS` below) but not the per-persona scores that
would let its interaction model be refit today with clustering. A
saturated 2x2 table has exactly four free parameters -- an intercept, two
main effects, one interaction -- so those four published means are enough
to recover the *same* main-effect and interaction numbers the original,
clustered fit would have reported for the fixed-effects part of the model
(:func:`saturated_2x2_effects`); what cannot be recovered is the
persona-variance term, or a clustered p-value for anything but the one
interaction PROGRESS.md already quoted a p-value for.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from thesis.analysis.hierarchy import InteractionModelResult, fit_interaction_model
from thesis.analysis.q1 import Q1_TASK_STAKES
from thesis.config import load_config
from thesis.judge.prompt import JudgeItem, Variant
from thesis.judge.rubric import RUBRIC_BY_KEY
from thesis.judge.run import score_items
from thesis.llm.base import CompletionRequest, CompletionResponse, LLMClient, Provider
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import CACHE_DIR, COST_LEDGER, INTERIM_DIR, MANIFESTS_DIR, ensure_dirs
from thesis.sim.grid import GridCell, expand, order_for_cache
from thesis.sim.memory import MemoryItem
from thesis.sim.memory_generation import load_frozen_memory
from thesis.sim.persona import Persona, load_frozen_personas
from thesis.sim.prompt import prompt_text_hash
from thesis.sim.run import RunManifest, run_grid
from thesis.sim.scenario import Scenario, Tone, build_scenarios

log = get_logger(__name__)


class _CompletionClient(Protocol):
    """The one capability :func:`score_judge_swap_replies` actually needs.

    Narrower than :class:`~thesis.llm.base.LLMClient` for the same reason
    ``judge/run.py`` defines its own equivalent: stating the real dependency
    means a test double only has to implement ``complete``, not the unused
    batching/capability methods a full client provides.
    """

    provider: Provider

    def complete(self, request: CompletionRequest) -> CompletionResponse: ...


JUDGE_SWAP_GRID_PATH: Path = INTERIM_DIR / "judge_swap_grid.parquet"
JUDGE_SWAP_SCORES_PATH: Path = INTERIM_DIR / "judge_swap_scores.parquet"

# Where the two-tone run writes. Separate files, so a bigger run never
# overwrites the neutral-only files sections 23 and 41 report.
JUDGE_SWAP_TWO_TONE_GRID_PATH: Path = INTERIM_DIR / "judge_swap_grid_two_tone.parquet"
JUDGE_SWAP_TWO_TONE_SCORES_PATH: Path = INTERIM_DIR / "judge_swap_scores_two_tone.parquet"
JUDGE_SWAP_TWO_TONE_MANIFEST_PATH: Path = MANIFESTS_DIR / "judge_swap_two_tone.json"

# Which scenarios a run uses. "pilot" is the 6 neutral-tone scenarios
# sections 23 and 41 report. "two_tone" adds the assertive tone, so 12
# scenarios and, with 10 personas, 120 cells per generator model. The 6 are
# a subset of the 12, which is what lets one run report both.
JudgeSwapDesign = Literal["pilot", "two_tone"]

JUDGE_SWAP_TONES: dict[JudgeSwapDesign, tuple[Tone, ...]] = {
    "pilot": ("neutral",),
    "two_tone": ("neutral", "assertive"),
}

# The tone every pilot-design reply uses -- see the module docstring for how
# this was recovered from the response cache rather than guessed at.
JUDGE_SWAP_TONE: Tone = "neutral"

# section 41 (Sep 4): the same pilot design at 120 replies, under the prompt
# that existed before commit 8f8df1e. Stored so a later run compares with a
# recorded number instead of a remembered one. Its two main effects were
# published as p<.001 and have no exact p-value to store.
SECTION_41_EFFECTS: dict[str, dict[str, float]] = {
    "generator_quality": {"coefficient": -0.54},
    "judge_generosity": {"coefficient": 0.61},
    "self_preference_overall": {"coefficient": 0.32, "p_value": 0.142},
    "self_preference_plausibility": {"coefficient": 0.70, "p_value": 0.014},
}

# Section 23's own pair, in the order its cell means below are keyed by --
# the second model is the reference level both factors are measured against.
DEFAULT_GENERATORS: tuple[str, str] = ("llama3.2:3b", "qwen2.5:3b")

# Section 23's four cell means, read off ``docs/figures/judge_swap_interaction.png``
# (also hardcoded in ``analysis/plots.py``'s ``main()``): mean rubric score,
# keyed (generator, judge). This *is* the historical record -- the per-reply
# scores behind it were never archived in a form this module can refit.
HISTORICAL_CELL_MEANS: dict[tuple[str, str], float] = {
    ("llama3.2:3b", "llama3.2:3b"): 3.725,
    ("llama3.2:3b", "qwen2.5:3b"): 2.675,
    ("qwen2.5:3b", "llama3.2:3b"): 4.328,
    ("qwen2.5:3b", "qwen2.5:3b"): 3.697,
}

# section 23 (Aug 24): self-preference interaction, overall rubric mean
# across all 6 items, and the corpus_plausibility item alone. No coefficient
# was ever published for the plausibility-only interaction, only its
# p-value -- reported as such, not filled in with a guess.
HISTORICAL_INTERACTION_P_OVERALL = 0.065
HISTORICAL_INTERACTION_P_PLAUSIBILITY = 0.20


def build_judge_swap_scenarios(design: JudgeSwapDesign = "pilot") -> list[Scenario]:
    """The scenarios one design runs, filtered out of the full 144-scenario
    grid ``build_scenarios`` returns.

    ``pilot`` gives 6: 2 task types x 3 directions, neutral tone only.
    ``two_tone`` gives 12, adding the assertive tone. Reuses
    :data:`Q1_TASK_STAKES`'s task/stakes pinning because the cache
    archaeology in the module docstring found the identical pinning in the
    judge-swap pilot's own cached calls, not because the two designs are the
    same design.
    """
    tones = JUDGE_SWAP_TONES[design]
    return [
        s
        for s in build_scenarios()
        if s.task_type in Q1_TASK_STAKES
        and s.stakes == Q1_TASK_STAKES[s.task_type]
        and s.tone in tones
    ]


def subset_to_neutral(frame: pd.DataFrame) -> pd.DataFrame:
    """The neutral-tone rows inside a two-tone frame.

    The 6 pilot scenarios are an exact subset of the 12, so a two-tone run
    contains the design section 41 reports. Refitting on just those rows
    separates two explanations for a changed answer: more data, or a wider
    set of situations. Works on a reply frame (which carries ``scenario_id``)
    and on a score frame (which carries the scenario inside ``item_id``).
    """
    neutral_ids = {s.scenario_id for s in build_judge_swap_scenarios("pilot")}
    if "scenario_id" in frame.columns:
        keep = frame["scenario_id"].isin(neutral_ids)
    else:
        keep = frame["item_id"].map(
            lambda i: any(str(i).endswith(f"{sid}__r1") for sid in neutral_ids)
        )
    return frame[keep].reset_index(drop=True)


def _model_slug(model: str) -> str:
    """A cell-id-safe stand-in for a model id (``llama3.2:3b`` -> ``llama3_2_3b``)."""
    return model.replace(":", "_").replace("/", "_").replace(".", "_")


def build_judge_swap_cells(
    personas: Sequence[Persona],
    model: str,
    role_label: str,
    *,
    n_replicates: int = 1,
    design: JudgeSwapDesign = "pilot",
) -> list[GridCell]:
    """Expand and cache-order one generator model's half of the judge-swap
    grid: ``len(personas)`` x scenarios x ``n_replicates``. Defaults to one
    replicate and the pilot design; with 10 personas that is 60 cells,
    matching section 23's "10 personas x 3 directions x 2 task types" per
    model. ``design="two_tone"`` gives 120 cells per model."""
    scenarios = build_judge_swap_scenarios(design)
    return order_for_cache(expand(personas, scenarios, [(model, role_label)], n_replicates))


@dataclass(frozen=True, slots=True)
class JudgeSwapGrid:
    """One generator model's half of the generated (or cache-served) grid,
    plus provenance."""

    frame: pd.DataFrame
    run_id: str
    model: str
    n_cells: int
    n_from_cache: int
    n_generated: int
    design: JudgeSwapDesign = "pilot"


def generate_judge_swap_grid(
    client: LLMClient,
    *,
    model: str,
    role_label: str | None = None,
    personas: Sequence[Persona] | None = None,
    stores: Mapping[str, Sequence[MemoryItem]] | None = None,
    cache_only: bool = False,
    n_replicates: int = 1,
    limit: int | None = None,
    cache: ResponseCache | None = None,
    ledger: CostLedger | None = None,
    design: JudgeSwapDesign = "pilot",
    progress_every: int = 20,
) -> JudgeSwapGrid:
    """Generate (or load from cache) one generator model's half of the
    judge-swap grid: 60 replies under the pilot design, 120 under
    ``two_tone``.

    Runs through :func:`thesis.sim.run.run_grid`, the same code path
    ``q1.py`` and ``pairs.py`` use. Call this once per generator model (see
    the module docstring for why one client cannot serve both) and combine
    the two frames with :func:`combine_generator_grids`.
    """
    config = load_config()
    personas = personas if personas is not None else load_frozen_personas()
    stores = stores if stores is not None else load_frozen_memory()
    cache = cache if cache is not None else ResponseCache(CACHE_DIR, cache_only=cache_only)
    ledger = ledger if ledger is not None else CostLedger(COST_LEDGER)
    role_label = role_label if role_label is not None else f"gen_{_model_slug(model)}"

    cells = build_judge_swap_cells(
        personas, model, role_label, n_replicates=n_replicates, design=design
    )
    if limit is not None:
        cells = cells[:limit]

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = RunManifest(
        run_id=run_id,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit="",
        git_dirty=False,
        config_hash="",
        models=[model],
        design={
            "kind": "judge_swap_grid",
            "generator": model,
            "design": design,
            "n_scenarios": len({c.scenario.scenario_id for c in cells}),
            # Ties this run to the prompt text that produced it. Section 51
            # exists because nothing recorded this before, and section 41's
            # replies turned out to come from a prompt that no longer exists.
            "prompt_text_hash": prompt_text_hash(),
        },
        n_cells=len(cells),
    )
    rows: list[dict[str, Any]] = run_grid(
        cells,
        client,
        stores,
        config,
        run_id=run_id,
        cache=cache,
        ledger=ledger,
        manifest=manifest,
        progress_every=progress_every,
    )
    frame = pd.DataFrame.from_records(rows)
    return JudgeSwapGrid(
        frame=frame,
        run_id=run_id,
        model=model,
        n_cells=len(cells),
        n_from_cache=manifest.n_from_cache,
        n_generated=manifest.n_generated,
        design=design,
    )


def combine_generator_grids(grids: Sequence[JudgeSwapGrid]) -> pd.DataFrame:
    """Concatenate every generator model's half into the full 120-reply frame.

    Raises if any ``cell_id`` repeats across halves -- ``GridCell.cell_id``
    does not include the model, only ``role_label`` does, so two halves built
    with the same role label would silently collide here rather than merely
    at generation time.
    """
    frame = pd.concat([g.frame for g in grids], ignore_index=True)
    duplicated = frame["cell_id"].duplicated()
    if duplicated.any():
        msg = f"duplicate cell_id(s) across generator grids: {sorted(set(frame.loc[duplicated, 'cell_id']))}"
        raise ValueError(msg)
    return frame


def build_judge_items(frame: pd.DataFrame) -> list[JudgeItem]:
    """One judge item per generated reply, reply text only -- no incoming
    message, matching section 23's "reused the existing judge machinery
    unchanged" (the reply-only design, not the with-context variant
    ``judge_blindness.py`` added later)."""
    return [
        JudgeItem(
            item_id=str(row.cell_id),
            text=str(row.body),
            is_generated=True,
            source_id=str(row.persona_id),
        )
        for row in frame.itertuples()
    ]


@dataclass(frozen=True, slots=True)
class JudgeSwapScores:
    """One judge model's scores over every reply, plus provenance."""

    frame: pd.DataFrame
    run_id: str
    judge_model: str
    n_scored: int
    n_from_cache: int
    n_invalid: int


def score_judge_swap_replies(
    replies: pd.DataFrame,
    client: _CompletionClient,
    *,
    judge_model: str,
    variant: Variant = "neutral",
    cache: ResponseCache | None = None,
    ledger: CostLedger | None = None,
    run_id: str | None = None,
    progress_every: int = 20,
) -> JudgeSwapScores:
    """Blind-score every reply in ``replies`` with one judge model.

    Call once per judge model (see the module docstring for why one client
    cannot serve both) and combine the two frames with
    :func:`combine_judge_scores`. Adds ``score_overall`` -- the mean of the
    six rubric items -- to each row, alongside the per-item ``score_<key>``
    columns :func:`thesis.judge.run.results_to_rows` uses, since the
    interaction model needs one outcome column per fit.
    """
    cache = cache if cache is not None else ResponseCache(CACHE_DIR)
    ledger = ledger if ledger is not None else CostLedger(COST_LEDGER)
    run_id = run_id if run_id is not None else datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    items = build_judge_items(replies)
    results, summary = score_items(
        items,
        client,
        variant=variant,
        model=judge_model,
        cache=cache,
        ledger=ledger,
        run_id=run_id,
        progress_every=progress_every,
    )

    generator_by_item = dict(zip(replies["cell_id"], replies["model"], strict=True))
    persona_by_item = dict(zip(replies["cell_id"], replies["persona_id"], strict=True))

    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "item_id": result.item_id,
            "generator": generator_by_item[result.item_id],
            "judge": judge_model,
            "persona_id": persona_by_item[result.item_id],
        }
        for key, score in result.scores.items():
            row[f"score_{key}"] = score
        row["score_overall"] = float(np.mean([result.scores[key] for key in RUBRIC_BY_KEY]))
        rows.append(row)

    frame = pd.DataFrame.from_records(rows)
    return JudgeSwapScores(
        frame=frame,
        run_id=run_id,
        judge_model=judge_model,
        n_scored=summary.n_scored,
        n_from_cache=summary.n_from_cache,
        n_invalid=summary.n_invalid,
    )


def combine_judge_scores(scores: Sequence[JudgeSwapScores]) -> pd.DataFrame:
    """Concatenate every judge model's scores into the full 240-row frame."""
    return pd.concat([s.frame for s in scores], ignore_index=True)


def saturated_2x2_effects(
    cell_means: Mapping[tuple[str, str], float],
    *,
    generator_alt: str,
    generator_ref: str,
    judge_alt: str,
    judge_ref: str,
) -> tuple[float, float, float]:
    """Generator main effect, judge main effect, and interaction, from four
    cell means of a saturated (generator x judge) 2x2 table.

    A 2x2 table has exactly 4 free parameters -- an intercept, two main
    effects, one interaction -- the same count as its 4 cell means, so this
    recovers precisely the fixed-effects part a mixed model with the same
    reference levels would report, without needing the per-observation data
    such a model would otherwise require. Both main effects hold the *other*
    factor at its reference level, matching the convention
    :meth:`~thesis.analysis.hierarchy.InteractionModelResult.main_effect` and
    :meth:`~thesis.analysis.hierarchy.InteractionModelResult.interaction` use,
    so the two are directly comparable.
    """
    baseline = cell_means[(generator_ref, judge_ref)]
    generator_effect = cell_means[(generator_alt, judge_ref)] - baseline
    judge_effect = cell_means[(generator_ref, judge_alt)] - baseline
    interaction = (
        cell_means[(generator_alt, judge_alt)] - baseline - generator_effect - judge_effect
    )
    return generator_effect, judge_effect, interaction


def fit_judge_swap_models(
    scores: pd.DataFrame,
    *,
    generator_reference: str,
    judge_reference: str,
) -> tuple[InteractionModelResult, InteractionModelResult]:
    """Fit the ``generator x judge`` interaction model on both outcomes
    section 23 reported: the overall rubric mean, and ``corpus_plausibility``
    alone.

    ``generator``/``judge`` levels are raw model ids (``llama3.2:3b``,
    ``qwen2.5:3b``) -- these contain a colon, which once collided with
    patsy's own ``:`` interaction-term separator inside
    :func:`thesis.analysis.hierarchy.fit_interaction_model`. That is a bug
    in the shared parser, not something this module should work around by
    renaming its factor levels, so it is fixed at the source
    (``hierarchy._clean_interaction_term``) instead.
    """
    overall = fit_interaction_model(
        scores,
        "score_overall",
        factor1_col="generator",
        factor2_col="judge",
        cluster_col="persona_id",
        reference1=generator_reference,
        reference2=judge_reference,
    )
    plausibility = fit_interaction_model(
        scores,
        "score_corpus_plausibility",
        factor1_col="generator",
        factor2_col="judge",
        cluster_col="persona_id",
        reference1=generator_reference,
        reference2=judge_reference,
    )
    return overall, plausibility


@dataclass(frozen=True, slots=True)
class JudgeSwapComparison:
    """One before/after number: section 23's value next to this run's."""

    label: str
    old_value: float
    old_p_value: float | None
    new_value: float
    new_p_value: float | None


def compare_judge_swap(
    overall_model: Any,
    plausibility_model: Any,
    *,
    generator_alt: str,
    generator_ref: str,
    judge_alt: str,
    judge_ref: str,
) -> list[JudgeSwapComparison]:
    """Section 23's numbers next to this run's, for every quantity that
    pilot reported: the generator (writer) quality effect, the judge
    (rater) generosity effect, and the self-preference interaction on both
    the overall rubric mean and ``corpus_plausibility`` alone.

    ``overall_model``/``plausibility_model`` are typed loosely (rather than
    as :class:`~thesis.analysis.hierarchy.InteractionModelResult`, which
    both really are in :func:`run_judge_swap_analysis`) -- the same choice
    :func:`thesis.analysis.q1.compare_to_historical` makes, and for the same
    reason: only ``.main_effect()``/``.interaction()`` are ever called, so a
    small stand-in exposing just those methods can stand in for a real fit
    in a test.

    The "old" main effects and interaction come from
    :func:`saturated_2x2_effects` over :data:`HISTORICAL_CELL_MEANS` --
    section 23 published a coefficient for the overall-rubric interaction
    only (+0.42, the module docstring explains why the others were never
    directly stated) but the same four cell means fix every other number
    in this table too, so they are computed rather than left blank.
    ``corpus_plausibility``'s old coefficient genuinely was never
    published (only its p-value was) and stays ``nan`` here rather than
    being invented.
    """
    old_generator_effect, old_judge_effect, old_interaction = saturated_2x2_effects(
        HISTORICAL_CELL_MEANS,
        generator_alt=generator_alt,
        generator_ref=generator_ref,
        judge_alt=judge_alt,
        judge_ref=judge_ref,
    )
    new_generator_effect, generator_p = overall_model.main_effect("generator", generator_alt)
    new_judge_effect, judge_p = overall_model.main_effect("judge", judge_alt)
    new_interaction, interaction_p = overall_model.interaction(generator_alt, judge_alt)
    new_plausibility_interaction, plausibility_p = plausibility_model.interaction(
        generator_alt, judge_alt
    )

    return [
        JudgeSwapComparison(
            "generator quality (writer effect)",
            old_generator_effect,
            None,
            new_generator_effect,
            generator_p,
        ),
        JudgeSwapComparison(
            "judge generosity (rater effect)", old_judge_effect, None, new_judge_effect, judge_p
        ),
        JudgeSwapComparison(
            "self-preference interaction (overall rubric mean)",
            old_interaction,
            HISTORICAL_INTERACTION_P_OVERALL,
            new_interaction,
            interaction_p,
        ),
        JudgeSwapComparison(
            "self-preference interaction (corpus_plausibility only)",
            float("nan"),
            HISTORICAL_INTERACTION_P_PLAUSIBILITY,
            new_plausibility_interaction,
            plausibility_p,
        ),
    ]


def format_comparison_table(comparisons: Sequence[JudgeSwapComparison]) -> str:
    """A plain-text table: one row per quantity, old next to new."""

    def _fmt(value: float | None) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        return f"{value:.3f}"

    header = f"{'quantity':<52}{'old':>10}{'old p':>10}{'new':>10}{'new p':>10}"
    lines = [header, "-" * len(header)]
    for c in comparisons:
        lines.append(
            f"{c.label:<52}{_fmt(c.old_value):>10}{_fmt(c.old_p_value):>10}"
            f"{_fmt(c.new_value):>10}{_fmt(c.new_p_value):>10}"
        )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class JudgeSwapResult:
    """Everything one run of this module produces."""

    replies: pd.DataFrame
    scores: pd.DataFrame
    overall_model: InteractionModelResult
    plausibility_model: InteractionModelResult
    comparisons: list[JudgeSwapComparison]


def run_judge_swap_analysis(
    replies: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    generator_alt: str,
    generator_ref: str,
    judge_alt: str,
    judge_ref: str,
) -> JudgeSwapResult:
    """Fit both models and build the comparison table -- the analysis half
    of what :func:`main` reports, kept separate from CLI/generation so it
    can be called directly on data that already exists."""
    overall_model, plausibility_model = fit_judge_swap_models(
        scores, generator_reference=generator_ref, judge_reference=judge_ref
    )
    comparisons = compare_judge_swap(
        overall_model,
        plausibility_model,
        generator_alt=generator_alt,
        generator_ref=generator_ref,
        judge_alt=judge_alt,
        judge_ref=judge_ref,
    )
    return JudgeSwapResult(
        replies=replies,
        scores=scores,
        overall_model=overall_model,
        plausibility_model=plausibility_model,
        comparisons=comparisons,
    )


def format_report(result: JudgeSwapResult) -> str:
    """The full plain-text report :func:`main` prints."""
    own_family = result.scores[result.scores["generator"] == result.scores["judge"]]
    own_family_means = own_family.groupby("generator")["score_overall"].mean()

    sections = [
        "Q3 (judge-swap): does a judge favor its own family? (current corpus)",
        "=" * 72,
        f"{len(result.replies)} replies, {len(result.scores)} scores "
        f"({result.overall_model.n_groups} personas)",
        "",
        format_comparison_table(result.comparisons),
        "",
        f"overall-rubric persona variance: {result.overall_model.group_variance:.4f}",
        f"n observations: {result.overall_model.n_observations}",
        "",
        "own-family mean score (generator == judge):",
        own_family_means.to_string(),
    ]
    return "\n".join(sections)


# A two-sided 0.05 test reaches 80% power when the estimate sits 2.80
# standard errors away from zero: 1.96 for the test, 0.84 for the power.
POWER_Z_SUM = 2.8016


def replies_for_power(coefficient: float, std_error: float, n_replies: int) -> float:
    """How many replies an effect of this size would need for 80% power.

    Precision improves with the square root of the sample size. So the
    required size grows with the square of the gap between today's standard
    error and the one the test needs. Counts replies, not scores, because
    every reply is scored by both judges.
    """
    if coefficient == 0.0:
        return float("inf")
    needed_se = abs(coefficient) / POWER_Z_SUM
    return n_replies * (std_error / needed_se) ** 2


@dataclass(frozen=True, slots=True)
class JudgeSwapEffect:
    """One measured effect: its size, how precisely it is measured, and its
    p-value."""

    key: str
    label: str
    outcome: str
    coefficient: float
    std_error: float
    p_value: float


def effect_estimates(
    overall_model: InteractionModelResult,
    plausibility_model: InteractionModelResult,
    *,
    generator_alt: str,
    judge_alt: str,
) -> list[JudgeSwapEffect]:
    """The effects this design measures, each with a standard error.

    Generator quality is how much higher one model's replies score, for a
    fixed judge. Judge generosity is how much higher one judge scores, for a
    fixed writer. Self-preference is what is left: how much higher a judge
    scores its own family's replies than those two effects alone predict.
    That is the interaction term, and it is the one Q3 asks about. It is
    reported twice, on the rubric mean and on ``corpus_plausibility`` alone.
    """
    generator_coefficient, generator_p = overall_model.main_effect("generator", generator_alt)
    judge_coefficient, judge_p = overall_model.main_effect("judge", judge_alt)
    interaction_coefficient, interaction_p = overall_model.interaction(generator_alt, judge_alt)
    plausibility_coefficient, plausibility_p = plausibility_model.interaction(
        generator_alt, judge_alt
    )
    return [
        JudgeSwapEffect(
            "generator_quality",
            "generator quality (writer effect)",
            "score_overall",
            generator_coefficient,
            overall_model.main_effect_std_error("generator", generator_alt),
            generator_p,
        ),
        JudgeSwapEffect(
            "judge_generosity",
            "judge generosity (rater effect)",
            "score_overall",
            judge_coefficient,
            overall_model.main_effect_std_error("judge", judge_alt),
            judge_p,
        ),
        JudgeSwapEffect(
            "self_preference_overall",
            "self-preference (overall rubric mean)",
            "score_overall",
            interaction_coefficient,
            overall_model.interaction_std_error(generator_alt, judge_alt),
            interaction_p,
        ),
        JudgeSwapEffect(
            "self_preference_plausibility",
            "self-preference (corpus_plausibility only)",
            "score_corpus_plausibility",
            plausibility_coefficient,
            plausibility_model.interaction_std_error(generator_alt, judge_alt),
            plausibility_p,
        ),
    ]


@dataclass(frozen=True, slots=True)
class JudgeSwapFit:
    """One subset of a run, fitted: how big it was and what it found."""

    label: str
    n_replies: int
    n_scores: int
    overall_model: InteractionModelResult
    plausibility_model: InteractionModelResult
    effects: list[JudgeSwapEffect]

    def effect(self, key: str) -> JudgeSwapEffect:
        """One effect, by key."""
        for effect in self.effects:
            if effect.key == key:
                return effect
        msg = f"no effect {key!r}; available: {[e.key for e in self.effects]}"
        raise KeyError(msg)


def fit_subset(
    replies: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    label: str,
    generator_alt: str,
    generator_ref: str,
    judge_alt: str,
    judge_ref: str,
) -> JudgeSwapFit:
    """Fit both models on one subset of a run, and read the effects off them."""
    overall_model, plausibility_model = fit_judge_swap_models(
        scores, generator_reference=generator_ref, judge_reference=judge_ref
    )
    return JudgeSwapFit(
        label=label,
        n_replies=len(replies),
        n_scores=len(scores),
        overall_model=overall_model,
        plausibility_model=plausibility_model,
        effects=effect_estimates(
            overall_model,
            plausibility_model,
            generator_alt=generator_alt,
            judge_alt=judge_alt,
        ),
    )


@dataclass(frozen=True, slots=True)
class TwoToneResult:
    """A two-tone run: the whole grid, and the neutral-tone half inside it."""

    full: JudgeSwapFit
    neutral: JudgeSwapFit


def run_two_tone_analysis(
    replies: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    generator_alt: str,
    generator_ref: str,
    judge_alt: str,
    judge_ref: str,
) -> TwoToneResult:
    """Fit the whole two-tone grid, then the neutral-tone rows inside it.

    The neutral half is the design sections 23 and 41 report. Reporting both
    separates two things a changed answer could come from: more data, or a
    wider set of situations.
    """
    return TwoToneResult(
        full=fit_subset(
            replies,
            scores,
            label="full (both tones)",
            generator_alt=generator_alt,
            generator_ref=generator_ref,
            judge_alt=judge_alt,
            judge_ref=judge_ref,
        ),
        neutral=fit_subset(
            subset_to_neutral(replies),
            subset_to_neutral(scores),
            label="neutral tone only",
            generator_alt=generator_alt,
            generator_ref=generator_ref,
            judge_alt=judge_alt,
            judge_ref=judge_ref,
        ),
    )


def format_two_tone_report(result: TwoToneResult) -> str:
    """The two-tone report: the full grid next to its neutral-tone half."""
    header = f"{'effect':<44}{'full':>9}{'se':>7}{'p':>7}{'neutral':>10}{'se':>7}{'p':>7}"
    lines = [
        "Q3 (judge-swap), two-tone run: does a judge favor its own family?",
        "=" * len(header),
        f"full: {result.full.n_replies} replies, {result.full.n_scores} scores",
        f"neutral only: {result.neutral.n_replies} replies, {result.neutral.n_scores} scores",
        "",
        header,
        "-" * len(header),
    ]
    for effect in result.full.effects:
        same = result.neutral.effect(effect.key)
        lines.append(
            f"{effect.label:<44}{effect.coefficient:>+9.3f}{effect.std_error:>7.3f}"
            f"{effect.p_value:>7.3f}{same.coefficient:>+10.3f}{same.std_error:>7.3f}"
            f"{same.p_value:>7.3f}"
        )

    interaction = result.full.effect("self_preference_overall")
    needed = replies_for_power(
        interaction.coefficient, interaction.std_error, result.full.n_replies
    )
    lines += [
        "",
        f"persona variance (overall rubric): {result.full.overall_model.group_variance:.4f}",
        f"replies needed for 80% power on the overall interaction: {needed:.0f}",
        "",
        "section 41 (same design, 120 replies, prompt before commit 8f8df1e):",
    ]
    for key, recorded in SECTION_41_EFFECTS.items():
        old_p = recorded.get("p_value")
        old_p_text = "n/a" if old_p is None else f"{old_p:.3f}"
        lines.append(
            f"  {result.full.effect(key).label:<44}"
            f"{recorded['coefficient']:>+9.3f}{old_p_text:>7}"
        )
    return "\n".join(lines)


def _effects_payload(fit: JudgeSwapFit) -> dict[str, Any]:
    """One fitted subset, as plain JSON-safe numbers."""
    return {
        "n_replies": fit.n_replies,
        "n_scores": fit.n_scores,
        "persona_variance": round(fit.overall_model.group_variance, 4),
        "effects": {
            effect.key: {
                "label": effect.label,
                "outcome": effect.outcome,
                "coefficient": round(effect.coefficient, 4),
                "std_error": round(effect.std_error, 4),
                "p_value": float(f"{effect.p_value:.4g}"),
            }
            for effect in fit.effects
        },
    }


def build_two_tone_manifest(
    result: TwoToneResult,
    *,
    generators: Sequence[str],
    judges: Sequence[str],
    n_from_cache: int,
    n_generated: int,
) -> dict[str, Any]:
    """Everything the write-up quotes, in one file.

    Records ``prompt_text_hash``, so this run is tied to the prompt that
    produced it. Section 41 had no such record, and its replies turned out to
    come from a prompt that had changed since.
    """
    interaction = result.full.effect("self_preference_overall")
    return {
        "run": {
            "design": "two_tone",
            "generators": list(generators),
            "judges": list(judges),
            "prompt_text_hash": prompt_text_hash(),
            "n_from_cache": n_from_cache,
            "n_generated": n_generated,
        },
        "full": _effects_payload(result.full),
        "neutral_only": _effects_payload(result.neutral),
        "power": {
            "target": "80% power, two-sided 0.05, overall-rubric interaction",
            "coefficient": round(interaction.coefficient, 4),
            "std_error": round(interaction.std_error, 4),
            "replies_needed": round(
                replies_for_power(
                    interaction.coefficient, interaction.std_error, result.full.n_replies
                )
            ),
        },
        "section_41": SECTION_41_EFFECTS,
        "caveats": [
            "two 3B local models stand in for the plan's cross-provider design",
            "one draw per cell; the model reproduces its own decision 60% of the "
            "time (PROGRESS.md section 50)",
            "section 41 used a different prompt, so that comparison mixes a prompt "
            "change with draw noise",
        ],
    }


def _report(
    replies: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    design: JudgeSwapDesign,
    generators: Sequence[str],
    judges: Sequence[str],
    n_from_cache: int,
    n_generated: int,
) -> None:
    """Print one run's report. A two-tone run also gets the neutral-only
    comparison and a manifest file.

    ``n_from_cache`` and ``n_generated`` describe the process that writes the
    manifest, so an ``--analyse-only`` run records zero of both.
    """
    print(
        format_report(
            run_judge_swap_analysis(
                replies,
                scores,
                generator_alt=generators[0],
                generator_ref=generators[1],
                judge_alt=judges[0],
                judge_ref=judges[1],
            )
        )
    )
    if design != "two_tone":
        return

    two_tone = run_two_tone_analysis(
        replies,
        scores,
        generator_alt=generators[0],
        generator_ref=generators[1],
        judge_alt=judges[0],
        judge_ref=judges[1],
    )
    print()
    print(format_two_tone_report(two_tone))

    manifest = build_two_tone_manifest(
        two_tone,
        generators=generators,
        judges=judges,
        n_from_cache=n_from_cache,
        n_generated=n_generated,
    )
    JUDGE_SWAP_TWO_TONE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_SWAP_TWO_TONE_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    log.info("wrote %s", JUDGE_SWAP_TWO_TONE_MANIFEST_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generators",
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        default=list(DEFAULT_GENERATORS),
        help="The two local models standing in for two generator 'families'.",
    )
    parser.add_argument(
        "--judges",
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        default=None,
        help="The two local models acting as judge. Defaults to --generators.",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Override the Ollama server URL (default: http://127.0.0.1:11434).",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Serve only from cache; fail rather than call Ollama.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap cells per generator, for smoke tests."
    )
    parser.add_argument(
        "--design",
        choices=("pilot", "two_tone"),
        default="pilot",
        help=(
            "Which scenarios to run. 'pilot' is the 6 neutral-tone scenarios "
            "sections 23 and 41 report (120 replies with 10 personas and two "
            "generators). 'two_tone' adds the assertive tone (240 replies), "
            "and contains the pilot."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Log a progress line every N cells, and every N judge calls.",
    )
    parser.add_argument(
        "--analyse-only",
        action="store_true",
        help="Re-fit from the stored reply and score files. Calls no model.",
    )
    parser.add_argument("--out", default=None, help="Where to write the replies.")
    parser.add_argument("--scores-out", default=None, help="Where to write the scores.")
    args = parser.parse_args()
    design: JudgeSwapDesign = args.design
    judges: list[str] = args.judges if args.judges is not None else list(args.generators)

    configure_logging()
    ensure_dirs()

    default_grid = JUDGE_SWAP_GRID_PATH if design == "pilot" else JUDGE_SWAP_TWO_TONE_GRID_PATH
    default_scores = (
        JUDGE_SWAP_SCORES_PATH if design == "pilot" else JUDGE_SWAP_TWO_TONE_SCORES_PATH
    )
    out_path = Path(args.out) if args.out else default_grid
    scores_out = Path(args.scores_out) if args.scores_out else default_scores

    if args.analyse_only:
        _report(
            pd.read_parquet(out_path),
            pd.read_parquet(scores_out),
            design=design,
            generators=args.generators,
            judges=judges,
            n_from_cache=0,
            n_generated=0,
        )
        return

    from thesis.llm.ollama_client import OllamaClient, OllamaUnavailableError

    def _client(model: str) -> OllamaClient:
        return (
            OllamaClient(model, host=args.ollama_host) if args.ollama_host else OllamaClient(model)
        )

    grids: list[JudgeSwapGrid] = []
    for model in args.generators:
        client = _client(model)
        if not client.is_available() and not args.cache_only:
            msg = (
                f"no Ollama server reachable at {client.host}. Start it with "
                f"'ollama serve', and pull the model with 'ollama pull {model}'."
            )
            raise OllamaUnavailableError(msg)
        grid = generate_judge_swap_grid(
            client,
            model=model,
            cache_only=args.cache_only,
            limit=args.limit,
            design=design,
            progress_every=args.progress_every,
        )
        log.info(
            "generator %s: %d cells (%d cached, %d generated)",
            model,
            grid.n_cells,
            grid.n_from_cache,
            grid.n_generated,
        )
        grids.append(grid)

    replies = combine_generator_grids(grids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    replies.to_parquet(out_path, compression="zstd", index=False)

    cache = ResponseCache(CACHE_DIR, cache_only=args.cache_only)
    ledger = CostLedger(COST_LEDGER)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    scores_list: list[JudgeSwapScores] = []
    for judge_model in judges:
        client = _client(judge_model)
        if not client.is_available() and not args.cache_only:
            msg = (
                f"no Ollama server reachable at {client.host}. Start it with "
                f"'ollama serve', and pull the model with 'ollama pull {judge_model}'."
            )
            raise OllamaUnavailableError(msg)
        scores = score_judge_swap_replies(
            replies,
            client,
            judge_model=judge_model,
            cache=cache,
            ledger=ledger,
            run_id=run_id,
            progress_every=args.progress_every,
        )
        log.info(
            "judge %s: %d scored (%d cached, %d invalid)",
            judge_model,
            scores.n_scored,
            scores.n_from_cache,
            scores.n_invalid,
        )
        scores_list.append(scores)

    scores_frame = combine_judge_scores(scores_list)
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    scores_frame.to_parquet(scores_out, compression="zstd", index=False)

    _report(
        replies,
        scores_frame,
        design=design,
        generators=args.generators,
        judges=judges,
        n_from_cache=sum(g.n_from_cache for g in grids),
        n_generated=sum(g.n_generated for g in grids),
    )


if __name__ == "__main__":
    main()
