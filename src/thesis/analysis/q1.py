"""Q1 as a reusable module: does hierarchical direction shape directive
language?

Every earlier Q1 result (PROGRESS.md sections 7, 20, 22, 33, 34, 36) came
from the same 240 replies -- 10 personas x 3 directions x 4 incoming-message
tones x 2 fixed task types, one replicate -- generated once by ad hoc,
uncommitted code and then re-measured in place as bugs were fixed elsewhere
in the pipeline. No version of that generation step was ever a script this
project could re-run on demand.

This module reconstructs that exact design and makes it a real entry point:
``python -m thesis.analysis.q1 --local llama3.2:3b`` generates (or serves
from cache) the same 240-cell grid against whichever personas the caller
currently has, extracts the same two outcome measures section 36 settled
on -- a reply-level rate and a sentence-level binary -- fits both models,
and prints old-vs-new comparison tables next to the numbers PROGRESS.md
already reported.

**How the design was recovered.** No commit ever recorded the CLI
invocation or scenario filter behind the 240-reply pilot; ``build_scenarios``
alone produces 144 scenarios (6 task types x 3 directions x 2 stakes x 4
tones), and ``sim.run``'s ``--limit`` merely truncates an already
cache-ordered list, which cannot reproduce a specific 24-scenario subset on
its own. The actual design was recovered empirically, from the still-present
local response cache (``runs/_cache``, gitignored but never deleted on this
machine): every cached simulator call from 2026-08-30 -- the batch behind
section 33's rerun, later reused unchanged by section 36 -- decodes to
exactly two task types, each pinned to one stakes level rather than crossing
stakes as a third factor: ``approve_or_decline`` (a real risk-carrying
decision) always at "high" stakes, ``report_problem`` (a routine check-in)
always at "routine". That is :data:`Q1_TASK_STAKES` below. 2 task types x 3
directions x 4 tones x 10 personas x 1 replicate = 240, matching every
figure this project has quoted for the pilot's size.

**What is new here, not just re-run.** The corpus was rebuilt in section 37
(quote-stripping, Lotus Notes quoting, and an over-aggressive signature
stripper, all fixed in ``thesis.data.rfc822``), which changed
``n_tokens_clean`` per message, which changed persona style statistics
(``derive_personas``), which changed the rendered prompt text every cell's
cache key depends on. So the 240 replies behind sections 33/36 no longer
reflect the current corpus, and generating against today's personas is a
fresh run, not a cache-hit replay of old data -- unless it turns out the
rendered prompt text happens not to have changed after all, in which case
that is itself reported, not assumed away.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from scipy import stats
from spacy.tokens import Doc
from statsmodels.stats.multitest import multipletests

from thesis.analysis.draw_stability import GridReliability, grid_draw_reliability
from thesis.analysis.hierarchy import (
    AssociationResult,
    FixedEffectsResult,
    MixedModelResult,
    SentenceModelResult,
    aggregate_replicates,
    cell_id_without_replicate,
    direction_decision_association,
    fit_direction_fixed_effects,
    fit_direction_mixed_model,
    fit_interaction_model,
    fit_sentence_level_model,
    summarize_by_direction,
)
from thesis.analysis.plots import plot_multi_line_trend
from thesis.config import load_config
from thesis.data.features import _load_nlp, extract_features, extract_sentence_features
from thesis.llm.base import LLMClient
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import (
    CACHE_DIR,
    COST_LEDGER,
    DOCS_FIGURES_DIR,
    INTERIM_DIR,
    MANIFESTS_DIR,
    ensure_dirs,
)
from thesis.sim.grid import GridCell, expand, order_for_cache
from thesis.sim.memory import MemoryItem
from thesis.sim.memory_generation import load_frozen_memory
from thesis.sim.persona import Persona, load_frozen_personas
from thesis.sim.prompt import prompt_text_hash
from thesis.sim.run import RunManifest, run_grid
from thesis.sim.scenario import Scenario, Stakes, build_scenarios

log = get_logger(__name__)

Q1_GRID_PATH: Path = INTERIM_DIR / "q1_direction_grid.parquet"
# A separate default for NVIDIA runs, so they never overwrite a local grid.
Q1_NVIDIA_GRID_PATH: Path = INTERIM_DIR / "q1_direction_grid_nvidia.parquet"
Q1_GROQ_GRID_PATH: Path = INTERIM_DIR / "q1_direction_grid_groq.parquet"
# Where a local full-grid run writes. A separate file, so the 1,440-cell run
# never overwrites the 240-cell one every earlier section reports.
Q1_FULL_GRID_PATH: Path = INTERIM_DIR / "q1_direction_grid_full.parquet"
# Where the full-grid run's own numbers go.
Q1_FULL_MANIFEST_PATH: Path = MANIFESTS_DIR / "q1_full_grid.json"

# The real-email benchmark: what direction does in real Enron email, measured
# by thesis.analysis.q1_real and written up in PROGRESS.md section 48.
REAL_MANIFEST_PATH: Path = MANIFESTS_DIR / "q1_real.json"

# The contrasts both sides measure, keyed the way the real manifest keys them.
REAL_CONTRAST_KEYS: tuple[str, ...] = (
    "imperative_ratio:down",
    "imperative_ratio:up",
    "is_imperative:down",
    "is_imperative:up",
    "hedge_rate:down",
    "hedge_rate:up",
)

# The reconstructed design of the original 240-reply Q1 pilot -- see the
# module docstring for how this was recovered from the response cache rather
# than from any committed script. Each task type is pinned to one stakes
# level rather than crossing stakes as a third factor.
Q1_TASK_STAKES: dict[str, Stakes] = {
    "approve_or_decline": "high",
    "report_problem": "routine",
}

# section 33 (Aug 29): imperative_ratio ~ direction, linear mixed model,
# reply-level rate, corrected personas, pre-corpus-rebuild. Section 36 refit
# the same 240 cached replies at the sentence grain and did not change these.
HISTORICAL_REPLY_LEVEL: dict[str, tuple[float, float]] = {
    "up": (0.056, 0.401),
    "down": (-0.052, 0.437),
}

# section 36 (Aug 30): is_imperative ~ direction, logistic mixed model
# (variational Bayes), one row per sentence, same 240 cached replies as
# section 33. Coefficients are on the logit scale.
HISTORICAL_SENTENCE_LEVEL: dict[str, tuple[float, float]] = {
    "up": (0.195, 0.310),
    "down": (-0.126, 0.534),
}

# Which scenarios a run uses. "pilot" is the 24 every earlier Q1 section
# reports. "full" is all 144.
Q1Design = Literal["pilot", "full"]

# The analysis plan for the full-grid run, fixed before any of its numbers
# were looked at. It is written into the run manifest, so the file records
# the order rather than a claim about it. This project has been burned twice
# by deciding what counted after seeing a result (sections 33 and 39).
ANALYSIS_PLAN: dict[str, Any] = {
    "primary": {
        "outcome": "is_imperative",
        "grain": "one row per sentence",
        "model": "logistic mixed model, random intercept per persona",
        "reference": "lateral",
        "contrasts": ["down", "up"],
    },
    "secondary": {
        "outcomes": ["imperative_ratio", "hedge_rate"],
        "grain": "one row per reply",
        "model": "linear mixed model, random intercept per persona",
        "status": "reported, not the headline",
    },
    "exploratory": {
        "tests": "direction x stakes and direction x task_type interactions",
        "correction": "Holm across the whole exploratory family",
        "rule": "a single p<.05 among these is not a finding",
    },
    "decision_by_direction": {
        "test": "chi-square, not clustered by persona",
        "caveat": (
            "the model repeats its own decision only 60% of the time, so this "
            "runs on an unstable outcome; see PROGRESS.md section 50"
        ),
    },
}


def build_q1_scenarios() -> list[Scenario]:
    """The 24 scenarios (2 task types x 3 directions x 4 tones) the Q1 pilot
    uses, filtered out of the full 144-scenario grid ``build_scenarios``
    returns. See :data:`Q1_TASK_STAKES` for which task type pins which
    stakes level, and the module docstring for how that pairing was
    recovered."""
    return [
        s
        for s in build_scenarios()
        if s.task_type in Q1_TASK_STAKES and s.stakes == Q1_TASK_STAKES[s.task_type]
    ]


def scenarios_for_design(design: Q1Design) -> list[Scenario]:
    """The scenarios one design runs.

    ``pilot`` is the 24 scenarios every earlier Q1 section reports. ``full``
    is the whole 144-scenario grid: 6 task types x 3 directions x 2 stakes x
    4 tones. The 24 are an exact subset of the 144, so a full run reuses
    every reply the pilot already generated and contains the pilot inside
    it. That is what lets one run report both numbers.
    """
    return build_q1_scenarios() if design == "pilot" else build_scenarios()


def design_of_frame(frame: pd.DataFrame) -> Q1Design:
    """Which design a stored grid file holds, read off its scenario count.

    A grid file records ``scenario_id`` but not which design asked for it,
    and ``--grid`` has to label a file it did not generate.
    """
    return "pilot" if frame["scenario_id"].nunique() <= len(build_q1_scenarios()) else "full"


def prompt_hash_of_frame(frame: pd.DataFrame) -> str:
    """The prompt hash a stored grid file carries, or ``"unknown"`` for a
    grid saved before this column existed -- an older file is still usable,
    just without this provenance check.
    """
    if "prompt_text_hash" not in frame.columns:
        return "unknown"
    return str(frame["prompt_text_hash"].iloc[0])


def full_grid_path(pilot_path: Path) -> Path:
    """Where a backend's full-grid file goes, derived from its pilot file.

    Derived rather than hard-coded so a full NVIDIA run cannot overwrite a
    full local one.
    """
    return pilot_path.with_name(f"{pilot_path.stem}_full{pilot_path.suffix}")


def full_grid_manifest_path(n_draws: int) -> Path:
    """Where a full-grid run's manifest goes, derived from how many draws
    per cell the grid holds.

    A single-draw run writes :data:`Q1_FULL_MANIFEST_PATH` -- the file
    section 51 points at. A run with more draws writes its own file, so
    generating a second draw and re-running the report never overwrites the
    numbers a written-up section already cites -- the same mistake section
    40 hit with a figure, and close to the staleness problem section 51
    itself is about. Applies the same way whether the grid was just
    generated or loaded with ``--grid``, since both go through
    :func:`_report_full_grid`.
    """
    if n_draws <= 1:
        return Q1_FULL_MANIFEST_PATH
    return Q1_FULL_MANIFEST_PATH.with_name(
        f"{Q1_FULL_MANIFEST_PATH.stem}_{n_draws}draws{Q1_FULL_MANIFEST_PATH.suffix}"
    )


def full_grid_figure_path(n_draws: int) -> Path:
    """Where a full-grid run's own-family figure goes, by the same rule as
    :func:`full_grid_manifest_path` and for the same reason.

    Before this, ``_report_full_grid`` wrote every run's figure to one fixed
    name, so a second-draw run silently overwrote whatever a single-draw run
    had produced. Section 51 never happened to cite that figure, so no
    written-up number went stale -- but the bug was live, and the fix already
    made for the manifest had not been made here. Section 54 is the first run
    to actually reference this figure, which is what surfaced it.
    """
    base = DOCS_FIGURES_DIR / "q1_full_grid_vs_real.png"
    if n_draws <= 1:
        return base
    return base.with_name(f"{base.stem}_{n_draws}draws{base.suffix}")


def build_q1_cells(
    personas: Sequence[Persona],
    model: str,
    role_label: str,
    *,
    n_replicates: int = 1,
    design: Q1Design = "pilot",
) -> list[GridCell]:
    """Expand and cache-order the Q1 grid: ``len(personas)`` scenarios x
    ``n_replicates``. Defaults to one replicate and the pilot design,
    matching every earlier Q1 run -- with 10 personas that is 240 cells.
    ``design="full"`` uses all 144 scenarios, which with 10 personas is
    1,440 cells."""
    scenarios = scenarios_for_design(design)
    return order_for_cache(expand(personas, scenarios, [(model, role_label)], n_replicates))


def _tone_from_scenario_id(scenario_id: str) -> str:
    """Recover the tone level from a scenario id.

    ``Scenario.scenario_id`` is built as
    ``f"{task_type}__{direction}__{stakes}__{tone}"`` (see ``sim.scenario``),
    and the simulator's result rows carry ``scenario_id`` but no separate
    ``tone`` column -- this is the one place that needs it, so it is parsed
    here rather than adding a column every other caller of the grid would
    have to ignore.
    """
    return scenario_id.rsplit("__", 1)[1]


def subset_to_pilot(frame: pd.DataFrame) -> pd.DataFrame:
    """The 24-scenario pilot rows inside a full grid.

    The pilot scenarios are an exact subset of the full 144, so a full run
    contains the same replies every earlier Q1 section reports. Refitting on
    just those rows separates two explanations for a changed answer: more
    data, or a wider set of situations.
    """
    pilot_ids = {s.scenario_id for s in build_q1_scenarios()}
    return frame[frame["scenario_id"].isin(pilot_ids)].reset_index(drop=True)


def contrast_se(fit: Any, level: str) -> float:
    """The standard error of one direction contrast.

    The sentence model carries a posterior SD per term, so that is used
    directly. The linear mixed models do not expose one through
    ``contrast``, so it is backed out of the coefficient and its p-value --
    the same way section 48 does it for real email. Returns NaN when the
    p-value leaves the standard error undefined, which happens when it
    rounds to 0 or 1.
    """
    posterior_sd = getattr(fit, "posterior_sd", None)
    if posterior_sd is not None:
        return float(posterior_sd[f"direction[T.{level}]"])

    # Imported here, not at module level: q1_real imports this module.
    from thesis.analysis.q1_real import implied_se

    coefficient, p_value = fit.contrast(level)
    if not 0.0 < p_value < 1.0:
        return float("nan")
    return implied_se(coefficient, p_value)


@dataclass(frozen=True, slots=True)
class Q1Grid:
    """The generated (or cache-served) Q1 direction grid, plus provenance.

    ``prompt_text_hash`` ties the grid to the prompt text that produced it
    -- :func:`generate_q1_grid` always computed this (in the ``RunManifest``
    it builds for :func:`thesis.sim.run.run_grid`), but never wrote it
    anywhere the saved grid could carry it, so no file on disk recorded
    which prompt made it (PROGRESS_llms.md's Q1 next steps, item 1). It is
    also a column on ``frame``, the same way ``model`` already is, so it
    survives a round trip through parquet.
    """

    frame: pd.DataFrame
    run_id: str
    model: str
    n_cells: int
    n_from_cache: int
    n_generated: int
    prompt_text_hash: str
    design: Q1Design = "pilot"


def generate_q1_grid(
    client: LLMClient,
    *,
    model: str,
    role_label: str = "sim_q1",
    personas: Sequence[Persona] | None = None,
    stores: Mapping[str, Sequence[MemoryItem]] | None = None,
    cache_only: bool = False,
    n_replicates: int = 1,
    limit: int | None = None,
    design: Q1Design = "pilot",
    progress_every: int = 100,
    cache: ResponseCache | None = None,
    ledger: CostLedger | None = None,
) -> Q1Grid:
    """Generate (or load from cache) the Q1 direction grid.

    Runs through :func:`thesis.sim.run.run_grid`, the same code path the
    main experimental grid and the Q2 pairing module (``analysis.pairs``)
    use -- same prompt assembly, same cache, same validation -- rather than
    a private loop. Defaults to the current, committed persona snapshot
    (already reflecting the section 37 corpus rebuild) and memory snapshot,
    so a caller with no local corpus can still run this against whatever
    personas ship with the repository.

    ``cache`` and ``ledger`` default to the project's real response cache
    and cost ledger (``runs/_cache``, ``outputs/manifests/cost_ledger.csv``,
    the latter tracked in git) -- pass a test-scoped instance of each to
    avoid touching either.
    """
    config = load_config()
    personas = personas if personas is not None else load_frozen_personas()
    stores = stores if stores is not None else load_frozen_memory()
    cache = cache if cache is not None else ResponseCache(CACHE_DIR, cache_only=cache_only)
    ledger = ledger if ledger is not None else CostLedger(COST_LEDGER)

    cells = build_q1_cells(personas, model, role_label, n_replicates=n_replicates, design=design)
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
            "kind": "q1_direction_grid",
            "design": design,
            "n_task_types": len({c.scenario.task_type for c in cells}),
            "n_scenarios": len({c.scenario.scenario_id for c in cells}),
            # Ties this run to the prompt text that produced it. Section 51
            # exists because nothing recorded this before.
            "prompt_text_hash": prompt_text_hash(),
            # Written here so the manifest records the plan alongside the
            # run, before any of its numbers exist.
            "analysis_plan": ANALYSIS_PLAN,
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
    frame["prompt_text_hash"] = manifest.design["prompt_text_hash"]
    return Q1Grid(
        frame=frame,
        run_id=run_id,
        model=model,
        n_cells=len(cells),
        n_from_cache=manifest.n_from_cache,
        n_generated=manifest.n_generated,
        prompt_text_hash=manifest.design["prompt_text_hash"],
        design=design,
    )


def parse_replies(frame: pd.DataFrame) -> dict[str, Doc]:
    """Parse every reply body with spaCy once, keyed by ``cell_id``.

    Both feature grains below need a parsed document per reply; computing
    this once and passing it to both (see :func:`run_q1_analysis`) avoids
    running the same 240-odd bodies through spaCy twice in one call.
    """
    if frame.empty:
        msg = "no rows to parse"
        raise ValueError(msg)
    nlp = _load_nlp()
    docs = nlp.pipe(frame["body"].tolist())
    return dict(zip(frame["cell_id"], docs, strict=True))


def extract_q1_reply_features(
    frame: pd.DataFrame, *, docs: dict[str, Doc] | None = None
) -> pd.DataFrame:
    """Reply-level linguistic features for every generated reply, joined
    back onto ``persona_id``/``direction``/``tone``/``decision`` -- what
    :func:`thesis.analysis.hierarchy.fit_direction_mixed_model` needs.

    Runs the same rule-based markers (``thesis.data.features.extract_features``)
    the corpus-wide power-score features use, so a generated reply's
    ``imperative_ratio`` is measured exactly the way a real message's is.
    Pass ``docs`` (from :func:`parse_replies`) to reuse an already-parsed
    set rather than parsing again.
    """
    if frame.empty:
        msg = "no rows to extract features from"
        raise ValueError(msg)
    docs = docs if docs is not None else parse_replies(frame)

    feature_rows = [
        asdict(extract_features(cell_id, docs[cell_id])) for cell_id in frame["cell_id"]
    ]
    features = pd.DataFrame(feature_rows).rename(columns={"message_uid": "cell_id"})

    merged = frame.merge(features, on="cell_id", how="left", validate="one_to_one")
    merged["tone"] = merged["scenario_id"].map(_tone_from_scenario_id)
    return merged


def extract_q1_sentence_features(
    frame: pd.DataFrame, *, docs: dict[str, Doc] | None = None
) -> pd.DataFrame:
    """One row per sentence across every generated reply, joined back onto
    each sentence's reply-level ``persona_id``/``direction``/``tone`` -- what
    :func:`thesis.analysis.hierarchy.fit_sentence_level_model` needs.

    This is the fix section 34/36 made permanent: ``imperative_ratio`` is
    close to meaningless for a one-sentence reply (57.5% of the original
    pilot's replies), so the sentence-level model is fit on the data at the
    grain it actually has, rather than on a coarse ratio. Pass ``docs``
    (from :func:`parse_replies`) to reuse an already-parsed set.
    """
    if frame.empty:
        msg = "no rows to extract features from"
        raise ValueError(msg)
    docs = docs if docs is not None else parse_replies(frame)

    sentence_rows: list[dict[str, Any]] = []
    for cell_id in frame["cell_id"]:
        sentence_rows.extend(
            asdict(sentence) for sentence in extract_sentence_features(cell_id, docs[cell_id])
        )
    sentences = pd.DataFrame(sentence_rows).rename(columns={"message_uid": "cell_id"})

    # stakes and task_type come straight off the result rows (see
    # sim.run.RESULT_SCHEMA). The full grid crosses both, and the
    # exploratory models need them at this grain. replicate is carried so a
    # multi-draw grid can be split by draw (see run_q1_analysis).
    meta = frame[
        ["cell_id", "persona_id", "direction", "scenario_id", "stakes", "task_type", "replicate"]
    ].copy()
    meta["tone"] = meta["scenario_id"].map(_tone_from_scenario_id)
    return sentences.merge(meta, on="cell_id", how="left", validate="many_to_one")


@dataclass(frozen=True, slots=True)
class ContrastComparison:
    """One direction contrast, the historical (pre-rebuild) number next to
    the number this run just measured."""

    level: str
    old_coefficient: float
    old_p_value: float
    new_coefficient: float
    new_p_value: float


def compare_to_historical(
    contrast: Any,
    historical: Mapping[str, tuple[float, float]],
) -> list[ContrastComparison]:
    """Pair a fitted model's contrasts with the historical numbers named in
    ``historical`` (:data:`HISTORICAL_REPLY_LEVEL` or
    :data:`HISTORICAL_SENTENCE_LEVEL`).

    ``contrast`` is anything exposing ``.contrast(level) -> (coef, p)`` --
    :class:`~thesis.analysis.hierarchy.MixedModelResult` and
    :class:`~thesis.analysis.hierarchy.SentenceModelResult` both do -- typed
    loosely (rather than as a union of the two) so this also works against a
    small stand-in in tests that does not need a real model fit.
    """
    comparisons = []
    for level, (old_coefficient, old_p_value) in historical.items():
        new_coefficient, new_p_value = contrast.contrast(level)
        comparisons.append(
            ContrastComparison(level, old_coefficient, old_p_value, new_coefficient, new_p_value)
        )
    return sorted(comparisons, key=lambda c: c.level)


def format_comparison_table(comparisons: Sequence[ContrastComparison], *, label: str) -> str:
    """A plain-text table: one row per direction contrast, old next to new.

    Plain text rather than a DataFrame repr so the CLI output reads cleanly
    in a terminal or a log file without depending on pandas' display
    settings.
    """
    header = f"{'direction':<10}{'old coef':>12}{'old p':>10}{'new coef':>12}{'new p':>10}"
    lines = [label, header, "-" * len(header)]
    for c in comparisons:
        lines.append(
            f"{c.level:<10}{c.old_coefficient:>12.3f}{c.old_p_value:>10.3f}"
            f"{c.new_coefficient:>12.3f}{c.new_p_value:>10.3f}"
        )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Q1Result:
    """Everything one run of this module produces, in one place -- what
    :func:`main` prints and what a caller wanting the numbers rather than
    the printout should use instead of re-parsing stdout.

    ``reply_model``, ``sentence_model``, ``hedge_model`` and
    ``decision_association`` are fitted on draw 1 only, even when the grid
    holds more draws -- this is what keeps a single-draw grid's report
    identical to before, and what a multi-draw grid's headline numbers are
    compared against. The four fields below are the multi-draw analysis:
    ``None`` when the grid has one draw, since there is nothing to average
    or compare.

    ``sentence_model_persona_fe`` fits the same ``is_imperative`` outcome a
    second way, by persona-clustered fixed effects instead of
    ``sentence_model``'s variational-Bayes random intercept. The VB
    p-value comes from a posterior SD that understates uncertainty on a
    large effect (found checking the gpt-oss-120b result, PROGRESS_llms.md's
    Q1 section); this fit is the honest cross-check, kept alongside the VB
    one rather than replacing it, the same way q1_real.py already
    cross-checks the real-email benchmark.
    """

    grid: Q1Grid
    reply_features: pd.DataFrame
    sentence_features: pd.DataFrame
    reply_model: MixedModelResult
    sentence_model: SentenceModelResult
    hedge_model: MixedModelResult
    decision_association: AssociationResult
    reply_level_comparison: list[ContrastComparison]
    sentence_level_comparison: list[ContrastComparison]
    sentence_model_persona_fe: FixedEffectsResult
    n_draws: int = 1
    aggregated_reply_model: MixedModelResult | None = None
    aggregated_hedge_model: MixedModelResult | None = None
    sentence_model_clustered: SentenceModelResult | None = None
    reliability: GridReliability | None = None


def run_q1_analysis(grid: Q1Grid) -> Q1Result:
    """Extract both feature grains from a generated grid and fit every Q1
    model this project has used, in one call -- the analysis half of what
    ``main`` reports, kept separate from CLI parsing and generation so it can
    be called directly (from a notebook, or a test) on a grid that already
    exists.

    A grid with more than one draw per cell (``--replicates`` > 1) also gets
    the multi-draw analysis: the reply-level fit on draws averaged per cell,
    the sentence-level fit with a second random intercept per cell, and the
    draw 1 vs draw 2 reliability table. See :class:`Q1Result`.
    """
    docs = parse_replies(grid.frame)
    reply_features = extract_q1_reply_features(grid.frame, docs=docs)
    sentence_features = extract_q1_sentence_features(grid.frame, docs=docs)

    n_draws = int(grid.frame["replicate"].nunique())
    draw1_replies = reply_features[reply_features["replicate"] == 1]
    draw1_sentences = sentence_features[sentence_features["replicate"] == 1]

    reply_model = fit_direction_mixed_model(draw1_replies, "imperative_ratio", reference="lateral")
    hedge_model = fit_direction_mixed_model(draw1_replies, "hedge_rate", reference="lateral")
    sentence_model = fit_sentence_level_model(draw1_sentences, "is_imperative", reference="lateral")
    sentence_model_persona_fe = fit_direction_fixed_effects(
        draw1_sentences,
        "is_imperative",
        cluster_col="persona_id",
        reference="lateral",
        family="logistic",
    )
    decision_association = direction_decision_association(draw1_replies)

    aggregated_reply_model = None
    aggregated_hedge_model = None
    sentence_model_clustered = None
    reliability = None
    if n_draws > 1:
        aggregated = aggregate_replicates(
            reply_features,
            ["imperative_ratio", "hedge_rate"],
            keep_cols=["persona_id", "direction", "scenario_id", "tone", "stakes", "task_type"],
        )
        aggregated_reply_model = fit_direction_mixed_model(
            aggregated, "imperative_ratio", reference="lateral"
        )
        aggregated_hedge_model = fit_direction_mixed_model(
            aggregated, "hedge_rate", reference="lateral"
        )

        clustered_sentences = sentence_features.assign(
            cell_base_id=cell_id_without_replicate(sentence_features["cell_id"])
        )
        sentence_model_clustered = fit_sentence_level_model(
            clustered_sentences, "is_imperative", reference="lateral", nested_col="cell_base_id"
        )
        reliability = grid_draw_reliability(reply_features)

    return Q1Result(
        grid=grid,
        reply_features=reply_features,
        sentence_features=sentence_features,
        reply_model=reply_model,
        sentence_model=sentence_model,
        hedge_model=hedge_model,
        decision_association=decision_association,
        reply_level_comparison=compare_to_historical(reply_model, HISTORICAL_REPLY_LEVEL),
        sentence_level_comparison=compare_to_historical(sentence_model, HISTORICAL_SENTENCE_LEVEL),
        sentence_model_persona_fe=sentence_model_persona_fe,
        n_draws=n_draws,
        aggregated_reply_model=aggregated_reply_model,
        aggregated_hedge_model=aggregated_hedge_model,
        sentence_model_clustered=sentence_model_clustered,
        reliability=reliability,
    )


def format_report(result: Q1Result) -> str:
    """The full plain-text report :func:`main` prints: generation
    provenance, both comparison tables, and the secondary numbers (hedge
    rate, persona variance, the decision-direction association) every
    earlier Q1 write-up also reported."""
    means = summarize_by_direction(result.reply_features, ["imperative_ratio", "hedge_rate"])

    sections = [
        "Q1: does direction predict directive language? (current corpus)",
        "=" * 64,
        f"model: {result.grid.model}  |  {result.grid.n_cells} cells "
        f"({result.grid.n_from_cache} from cache, {result.grid.n_generated} generated)",
        "",
        format_comparison_table(
            result.reply_level_comparison,
            label="Reply-level (linear): imperative_ratio ~ direction",
        ),
        "",
        format_comparison_table(
            result.sentence_level_comparison,
            label="Sentence-level (logistic, logit scale): is_imperative ~ direction",
        ),
        "",
        f"reply-level persona variance: {result.reply_model.group_variance:.4f}",
        f"sentence-level persona sd:    {result.sentence_model.group_sd:.4f}",
        f"n replies: {result.reply_model.n_observations}  |  "
        f"n sentences: {result.sentence_model.n_observations}",
        "",
        "is_imperative ~ direction (persona-clustered fixed effects, cross-checks the VB fit above):",
        f"  up:   {result.sentence_model_persona_fe.contrast('up')[0]:.3f} "
        f"(p={result.sentence_model_persona_fe.contrast('up')[1]:.3f})",
        f"  down: {result.sentence_model_persona_fe.contrast('down')[0]:.3f} "
        f"(p={result.sentence_model_persona_fe.contrast('down')[1]:.3f})",
        "",
        "hedge_rate ~ direction (new run only, no historical comparison tracked):",
        f"  up:   {result.hedge_model.contrast('up')[0]:.3f} "
        f"(p={result.hedge_model.contrast('up')[1]:.3f})",
        f"  down: {result.hedge_model.contrast('down')[0]:.3f} "
        f"(p={result.hedge_model.contrast('down')[1]:.3f})",
        "",
        f"decision ~ direction: chi2={result.decision_association.statistic:.2f}, "
        f"p={result.decision_association.p_value:.3f}, "
        f"df={result.decision_association.degrees_of_freedom} "
        "(not clustered by persona -- see hierarchy.AssociationResult)",
        "",
        "mean imperative_ratio / hedge_rate by direction:",
        means.to_string(),
    ]
    report = "\n".join(sections)
    multi_draw = format_multi_draw_report(result)
    return f"{report}\n{multi_draw}" if multi_draw else report


def format_multi_draw_report(result: Q1Result) -> str:
    """The extra report for a grid with more than one draw per cell: the
    aggregated reply-level fit, the sentence-level fit clustered by cell, and
    the draw 1 vs draw 2 reliability table -- printed next to the
    draw-1-only numbers :func:`format_report` already shows. Returns an
    empty string for a single-draw grid, so :func:`format_report` prints
    nothing extra in that case.
    """
    if result.n_draws <= 1 or result.reliability is None:
        return ""
    assert result.aggregated_reply_model is not None
    assert result.aggregated_hedge_model is not None
    assert result.sentence_model_clustered is not None

    agg = result.aggregated_reply_model
    agg_hedge = result.aggregated_hedge_model
    clustered = result.sentence_model_clustered
    rel = result.reliability

    def _pair(fit: MixedModelResult | SentenceModelResult) -> str:
        up_c, up_p = fit.contrast("up")
        down_c, down_p = fit.contrast("down")
        return f"up: {up_c:+.3f} (p={up_p:.3f})  down: {down_c:+.3f} (p={down_p:.3f})"

    lines = [
        "",
        f"Multi-draw grid: {result.n_draws} draws per cell",
        "=" * 64,
        "",
        "Reply-level (linear), one row per cell, averaged over draws:",
        f"  imperative_ratio ~ direction: {_pair(agg)}",
        f"    draw 1 only, for comparison: {_pair(result.reply_model)}",
        f"  hedge_rate ~ direction:       {_pair(agg_hedge)}",
        f"    draw 1 only, for comparison: {_pair(result.hedge_model)}",
        "",
        "Sentence-level (logistic, logit scale), random intercept per persona " "and per cell:",
        f"  is_imperative ~ direction: {_pair(clustered)}",
        f"    draw 1 only, for comparison: {_pair(result.sentence_model)}",
        "",
        "Draw 1 vs draw 2, across every cell (section 50's check, generalized):",
        f"  imperative_ratio: pearson={rel.imperative_ratio.pearson:.3f} "
        f"spearman={rel.imperative_ratio.spearman:.3f} icc={rel.imperative_ratio.icc:.3f} "
        f"(2 draws ~{rel.imperative_ratio.spearman_brown_k2:.3f}, "
        f"3 draws ~{rel.imperative_ratio.spearman_brown_k3:.3f})",
        f"  hedge_rate:       pearson={rel.hedge_rate.pearson:.3f} "
        f"spearman={rel.hedge_rate.spearman:.3f} icc={rel.hedge_rate.icc:.3f} "
        f"(2 draws ~{rel.hedge_rate.spearman_brown_k2:.3f}, "
        f"3 draws ~{rel.hedge_rate.spearman_brown_k3:.3f})",
        f"  decision:         {rel.decision.share_agree:.1%} agree, "
        f"kappa={rel.decision.kappa:.3f} (n={rel.decision.n} cells)",
    ]
    return "\n".join(lines)


def grid_contrasts(result: Q1Result) -> dict[str, tuple[float, float]]:
    """Every direction contrast of one grid, keyed the way the real manifest keys them."""
    fits: dict[str, MixedModelResult | SentenceModelResult] = {
        "imperative_ratio": result.reply_model,
        "is_imperative": result.sentence_model,
        "hedge_rate": result.hedge_model,
    }
    return {
        f"{outcome}:{level}": fit.contrast(level)
        for outcome, fit in fits.items()
        for level in ("down", "up")
    }


def compare_with_real(
    contrasts: Mapping[str, tuple[float, float]], manifest: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    """Each grid contrast next to real email, with a rough z-test of the difference.

    The real side comes from the manifest :mod:`thesis.analysis.q1_real`
    writes (PROGRESS.md section 48). Rough for the reason stated there: both
    standard errors are backed out of a coefficient and a p-value, and the
    p-values are rounded.
    """
    # Imported here, not at module level: q1_real imports this module, so a
    # module-level import would be circular.
    from thesis.analysis.q1_real import implied_se

    real_side = manifest["simulator_vs_real"]
    comparison: dict[str, dict[str, float]] = {}
    for key, (coefficient, p_value) in contrasts.items():
        if key not in real_side:
            msg = f"the real manifest has no contrast {key!r}; it has {sorted(real_side)}"
            raise KeyError(msg)
        real_coefficient = float(real_side[key]["real"])
        real_p = float(real_side[key]["real_p"])
        difference = coefficient - real_coefficient
        se = math.hypot(implied_se(coefficient, p_value), implied_se(real_coefficient, real_p))
        comparison[key] = {
            "grid": round(coefficient, 4),
            "grid_p": float(f"{p_value:.4g}"),
            "real": round(real_coefficient, 4),
            "real_p": float(f"{real_p:.4g}"),
            "difference": round(difference, 4),
            # se is 0 only when a coefficient is 0 with p=1 on both sides,
            # which is a zero difference, or when a p-value rounded to 0.
            "difference_p": (
                round(float(2 * stats.norm.sf(abs(difference) / se)), 4) if se > 0 else 0.0
            ),
        }
    return comparison


def format_real_comparison(comparison: Mapping[str, Mapping[str, float]]) -> str:
    """The comparison as a plain table, in the order the real module reports."""
    lines = [
        "Against real email (PROGRESS.md section 48)",
        "-" * 76,
        f"{'contrast':24s}{'grid':>9s}{'grid p':>9s}{'real':>9s}{'real p':>9s}"
        f"{'difference':>12s}{'diff p':>9s}",
    ]
    for key in REAL_CONTRAST_KEYS:
        row = comparison.get(key)
        if row is None:
            continue
        lines.append(
            f"{key:24s}{row['grid']:>+9.3f}{row['grid_p']:>9.3f}{row['real']:>+9.3f}"
            f"{row['real_p']:>9.3f}{row['difference']:>+12.3f}{row['difference_p']:>9.3f}"
        )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ContrastEstimate:
    """One direction contrast: the estimate, how precise it is, and its p-value."""

    outcome: str
    level: str
    coefficient: float
    std_error: float
    p_value: float


@dataclass(frozen=True, slots=True)
class ExploratoryTest:
    """One exploratory interaction term, with its Holm-adjusted p-value."""

    family: str
    term: str
    coefficient: float
    p_value: float
    holm_p: float


def contrast_estimates(result: Q1Result) -> list[ContrastEstimate]:
    """Every direction contrast of one fitted grid, with standard errors.

    The standard error is the point of this function. Section 39 could not
    detect an effect it had estimated at the right size, because its
    standard error was about four times real email's. So precision, not just
    the coefficient, is what a bigger run has to be judged on.
    """
    fits: dict[str, Any] = {
        "is_imperative": result.sentence_model,
        "imperative_ratio": result.reply_model,
        "hedge_rate": result.hedge_model,
    }
    estimates = []
    for outcome, fit in fits.items():
        for level in ("down", "up"):
            coefficient, p_value = fit.contrast(level)
            estimates.append(
                ContrastEstimate(outcome, level, coefficient, contrast_se(fit, level), p_value)
            )
    return estimates


def sentence_levels_by_direction(sentence_features: pd.DataFrame) -> dict[str, float]:
    """Share of sentences that give an order, by direction.

    The design is balanced across directions, so this plain share equals the
    model's predicted probability. Section 48 reports the real side the same
    way, which is what makes the two comparable.
    """
    means = sentence_features.groupby("direction")["is_imperative"].mean()
    return {str(d): float(means[d]) for d in means.index}


def run_exploratory_tests(reply_features: pd.DataFrame) -> list[ExploratoryTest]:
    """Does the direction effect change with stakes, or with task type?

    **Exploratory.** The full grid is the first Q1 design that crosses
    stakes and runs more than two task types, so these terms could not be
    fitted before. Both models are fitted on ``imperative_ratio`` at the
    reply grain, because ``fit_interaction_model`` is a linear mixed model
    and there is no sentence-grain equivalent.

    Every interaction term from both models forms one family, and the
    p-values are Holm-adjusted across all of it. A single unadjusted p<.05
    here is not a finding, and is not to be presented as one.
    """
    fits = {
        "direction x stakes": fit_interaction_model(
            reply_features,
            "imperative_ratio",
            factor2_col="stakes",
            reference2="routine",
        ),
        "direction x task_type": fit_interaction_model(
            reply_features,
            "imperative_ratio",
            factor2_col="task_type",
            reference2=sorted(reply_features["task_type"].unique())[0],
        ),
    }
    rows = [
        (family, term, fit.coefficients[term], fit.p_values[term])
        for family, fit in fits.items()
        for term in sorted(fit.coefficients)
        if ":" in term
    ]
    if not rows:
        return []
    holm = multipletests([row[3] for row in rows], method="holm")[1]
    return [
        ExploratoryTest(family, term, coefficient, p_value, float(adjusted))
        for (family, term, coefficient, p_value), adjusted in zip(rows, holm, strict=True)
    ]


def _contrast_block(fit: MixedModelResult | SentenceModelResult) -> dict[str, dict[str, float]]:
    """Coefficient, standard error and p-value for both direction contrasts
    of one fitted model -- the shape the manifest stores a fit as."""
    block = {}
    for level in ("down", "up"):
        coefficient, p_value = fit.contrast(level)
        block[level] = {
            "coefficient": round(coefficient, 4),
            "std_error": round(contrast_se(fit, level), 4),
            "p_value": float(f"{p_value:.4g}"),
        }
    return block


def multi_draw_manifest_section(full: Q1Result) -> dict[str, Any] | None:
    """The multi-draw analysis, in the shape the manifest stores it: the
    aggregated reply-level fit and the sentence-level fit clustered by cell,
    each next to its draw-1-only baseline, plus the draw 1 vs draw 2
    reliability table. This, not the printed report alone, is what a later
    write-up should read the numbers from -- section 51 exists because a
    number lived only in a printout once already.

    ``None`` when ``full`` has one draw, since there is nothing to report.
    """
    if full.n_draws <= 1:
        return None
    assert full.aggregated_reply_model is not None
    assert full.aggregated_hedge_model is not None
    assert full.sentence_model_clustered is not None
    assert full.reliability is not None
    reliability = full.reliability

    return {
        "n_draws": full.n_draws,
        "n_rows": full.grid.n_cells,
        "n_cells": reliability.n_cells,
        "prompt_text_hash": prompt_text_hash(),
        "imperative_ratio_aggregated": _contrast_block(full.aggregated_reply_model),
        "imperative_ratio_draw1_only": _contrast_block(full.reply_model),
        "hedge_rate_aggregated": _contrast_block(full.aggregated_hedge_model),
        "hedge_rate_draw1_only": _contrast_block(full.hedge_model),
        "is_imperative_clustered": _contrast_block(full.sentence_model_clustered),
        "is_imperative_draw1_only": _contrast_block(full.sentence_model),
        "reliability": {
            "imperative_ratio": asdict(reliability.imperative_ratio),
            "hedge_rate": asdict(reliability.hedge_rate),
            "decision": asdict(reliability.decision),
        },
    }


def build_full_manifest(
    full: Q1Result, pilot: Q1Result, real_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Everything the full-grid run measured, plus the plan it was measured
    under and the two runs it has to be read against.

    ``pilot`` is the same 24-scenario design refit on the subset of rows
    inside ``full``. Reporting both keeps "more data" apart from "a wider
    set of situations". The ``"multi_draw"`` key
    (:func:`multi_draw_manifest_section`) is ``None`` unless ``full`` has
    more than one draw per cell.
    """
    from thesis.analysis.q1_real import implied_se

    real_side = real_manifest["simulator_vs_real"]
    section_39 = real_manifest["simulator_section_39"]["contrasts"]

    def _se(coefficient: float, p_value: float) -> float:
        return implied_se(coefficient, p_value) if 0.0 < p_value < 1.0 else float("nan")

    precision = {}
    for estimate in contrast_estimates(full):
        key = f"{estimate.outcome}:{estimate.level}"
        pilot_match = next(
            e
            for e in contrast_estimates(pilot)
            if e.outcome == estimate.outcome and e.level == estimate.level
        )
        old_coefficient, old_p = section_39[estimate.outcome][estimate.level]
        precision[key] = {
            "full_coefficient": round(estimate.coefficient, 4),
            "full_se": round(estimate.std_error, 4),
            "full_p": float(f"{estimate.p_value:.4g}"),
            "pilot_coefficient": round(pilot_match.coefficient, 4),
            "pilot_se": round(pilot_match.std_error, 4),
            "pilot_p": float(f"{pilot_match.p_value:.4g}"),
            "section_39_coefficient": old_coefficient,
            "section_39_se": round(_se(old_coefficient, old_p), 4),
            "real_coefficient": round(float(real_side[key]["real"]), 4),
            "real_se": round(
                _se(float(real_side[key]["real"]), float(real_side[key]["real_p"])), 4
            ),
        }

    return {
        "analysis_plan": ANALYSIS_PLAN,
        "run": {
            "model": full.grid.model,
            "design": full.grid.design,
            "prompt_text_hash": prompt_text_hash(),
            "n_draws": full.n_draws,
            "n_cells": full.grid.n_cells,
            "n_from_cache": full.grid.n_from_cache,
            "n_generated": full.grid.n_generated,
            "n_replies_full": int(full.reply_model.n_observations),
            "n_sentences_full": int(full.sentence_model.n_observations),
            "n_replies_pilot": int(pilot.reply_model.n_observations),
            "n_sentences_pilot": int(pilot.sentence_model.n_observations),
        },
        "primary": {
            f"{e.outcome}:{e.level}": {
                "coefficient": round(e.coefficient, 4),
                "std_error": round(e.std_error, 4),
                "p_value": float(f"{e.p_value:.4g}"),
            }
            for e in contrast_estimates(full)
            if e.outcome == ANALYSIS_PLAN["primary"]["outcome"]
        },
        "precision": precision,
        "exploratory": [asdict(test) for test in run_exploratory_tests(full.reply_features)],
        "levels_by_direction": {
            "full": sentence_levels_by_direction(full.sentence_features),
            "pilot": sentence_levels_by_direction(pilot.sentence_features),
            "real": real_manifest["real_levels_by_direction"]["is_imperative"],
        },
        "decision_by_direction": {
            "chi2": round(full.decision_association.statistic, 3),
            "p_value": float(f"{full.decision_association.p_value:.4g}"),
            "degrees_of_freedom": full.decision_association.degrees_of_freedom,
            "caveat": ANALYSIS_PLAN["decision_by_direction"]["caveat"],
        },
        "persona_variance": {
            "reply_level": round(full.reply_model.group_variance, 4),
            "sentence_level_sd": round(full.sentence_model.group_sd, 4),
        },
        "multi_draw": multi_draw_manifest_section(full),
    }


def format_precision_table(manifest: Mapping[str, Any]) -> str:
    """The full grid, the 24-scenario subset inside it, section 39 and real
    email, side by side, with a standard error for each."""
    lines = [
        "Precision: full grid vs the 24-scenario subset vs section 39 vs real email",
        "-" * 94,
        f"{'contrast':24s}{'full':>9s}{'se':>8s}{'subset':>9s}{'se':>8s}"
        f"{'sec 39':>9s}{'se':>8s}{'real':>9s}{'se':>8s}",
    ]
    for key in REAL_CONTRAST_KEYS:
        row = manifest["precision"].get(key)
        if row is None:
            continue
        lines.append(
            f"{key:24s}{row['full_coefficient']:>+9.3f}{row['full_se']:>8.3f}"
            f"{row['pilot_coefficient']:>+9.3f}{row['pilot_se']:>8.3f}"
            f"{row['section_39_coefficient']:>+9.3f}{row['section_39_se']:>8.3f}"
            f"{row['real_coefficient']:>+9.3f}{row['real_se']:>8.3f}"
        )
    return "\n".join(lines)


def format_exploratory_table(manifest: Mapping[str, Any]) -> str:
    """The exploratory interaction terms, labelled as exploratory."""
    lines = [
        "Exploratory (not a finding on its own): does direction depend on stakes or task type?",
        "-" * 94,
        f"{'family':24s}{'term':>44s}{'coef':>9s}{'p':>9s}{'holm p':>9s}",
    ]
    for test in manifest["exploratory"]:
        lines.append(
            f"{test['family']:24s}{test['term']:>44s}{test['coefficient']:>+9.3f}"
            f"{test['p_value']:>9.3f}{test['holm_p']:>9.3f}"
        )
    return "\n".join(lines)


def plot_full_vs_real(manifest: Mapping[str, Any], path: Path) -> Path:
    """Orders per sentence by direction: real email, the full grid, and the
    24-scenario subset. Follows section 48's figures."""
    levels = manifest["levels_by_direction"]
    order = ("down", "lateral", "up")
    return plot_multi_line_trend(
        ("writing down", "writing to a peer", "writing up"),
        {
            "real Enron email": tuple(float(levels["real"][d]) for d in order),
            "simulator, 1,440 replies": tuple(float(levels["full"][d]) for d in order),
            "simulator, 240 replies": tuple(float(levels["pilot"][d]) for d in order),
        },
        path,
        title="Orders per sentence: the bigger simulator run against real email",
        subtitle="Share of sentences that give an order. Real email peaks writing down.",
        x_label="who the writer is writing to",
        y_label="share of sentences that give an order",
        legend_loc="upper left",
    )


def _report_full_grid(grid: Q1Grid, result: Q1Result) -> None:
    """Refit the 24-scenario subset, write the manifest and the figure, and
    print both results side by side."""
    if not REAL_MANIFEST_PATH.exists():
        msg = f"no real-email benchmark at {REAL_MANIFEST_PATH}; run thesis.analysis.q1_real first"
        raise FileNotFoundError(msg)
    real_manifest = json.loads(REAL_MANIFEST_PATH.read_text(encoding="utf-8"))

    pilot_frame = subset_to_pilot(grid.frame)
    pilot_grid = Q1Grid(
        frame=pilot_frame,
        run_id=grid.run_id,
        model=grid.model,
        n_cells=len(pilot_frame),
        n_from_cache=len(pilot_frame),
        n_generated=0,
        prompt_text_hash=grid.prompt_text_hash,
        design="pilot",
    )
    pilot_result = run_q1_analysis(pilot_grid)

    manifest = build_full_manifest(result, pilot_result, real_manifest)
    manifest_path = full_grid_manifest_path(result.n_draws)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figure = plot_full_vs_real(manifest, full_grid_figure_path(result.n_draws))

    print()
    print(format_precision_table(manifest))
    print()
    print(format_exploratory_table(manifest))
    log.info("wrote %s and %s", manifest_path, figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument(
        "--local",
        metavar="MODEL",
        help=(
            "Generate with a local Ollama model (e.g. llama3.2:3b): real "
            "generated text, no key, no cost."
        ),
    )
    backend.add_argument(
        "--nvidia",
        metavar="MODEL",
        help="Generate with a free NVIDIA-hosted model (needs NVIDIA_API_KEY).",
    )
    backend.add_argument(
        "--grid",
        metavar="PATH",
        help="Analyse a grid file that already exists, without calling any model.",
    )
    backend.add_argument(
        "--groq",
        metavar="MODEL",
        help="Generate with a free Groq-hosted model (needs GROQ_API_KEY).",
    )
    parser.add_argument(
        "--compare-real",
        action="store_true",
        help="Also print each contrast next to real email (PROGRESS.md section 48).",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Override the Ollama server URL (default: http://127.0.0.1:11434).",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Serve only from cache; fail rather than call a model.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap cells, for smoke tests.")
    parser.add_argument(
        "--design",
        choices=("pilot", "full"),
        default="pilot",
        help=(
            "Which scenarios to run. 'pilot' is the 24-scenario design every "
            "earlier Q1 section reports (240 cells with 10 personas). 'full' "
            "is all 144 scenarios (1,440 cells), and contains the pilot."
        ),
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help=(
            "How many draws of each cell to generate. Each draw sends the "
            "identical prompt and differs only in the cache's draw index, so "
            "draw 1 is served from cache and only the new draws cost anything. "
            "Section 50 measured how much a single draw moves on its own; more "
            "draws average that out."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log a progress line every N cells.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the grid. Each backend has its own default file.",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    if args.grid:
        frame = pd.read_parquet(args.grid)
        grid = Q1Grid(
            frame=frame,
            run_id="from-file",
            model=str(frame["model"].iloc[0]),
            n_cells=len(frame),
            n_from_cache=len(frame),
            n_generated=0,
            prompt_text_hash=prompt_hash_of_frame(frame),
            design=design_of_frame(frame),
        )
        _report(grid, compare_real=args.compare_real)
        return

    client: LLMClient
    if args.nvidia:
        from thesis.llm.nvidia_client import NvidiaClient

        client, model, default_out = NvidiaClient(), args.nvidia, Q1_NVIDIA_GRID_PATH
    elif args.groq:
        from thesis.llm.groq_client import GroqClient

        client, model, default_out = GroqClient(), args.groq, Q1_GROQ_GRID_PATH
    else:
        from thesis.llm.ollama_client import OllamaClient, OllamaUnavailableError

        ollama = (
            OllamaClient(args.local, host=args.ollama_host)
            if args.ollama_host
            else OllamaClient(args.local)
        )
        if not ollama.is_available() and not args.cache_only:
            msg = (
                f"no Ollama server reachable at {ollama.host}. Start it with "
                f"'ollama serve', and pull the model with 'ollama pull {args.local}'."
            )
            raise OllamaUnavailableError(msg)
        client, model, default_out = ollama, args.local, Q1_GRID_PATH

    grid = generate_q1_grid(
        client,
        model=model,
        cache_only=args.cache_only,
        limit=args.limit,
        design=args.design,
        progress_every=args.progress_every,
        n_replicates=args.replicates,
    )
    if args.design == "full":
        default_out = full_grid_path(default_out)
    out_path = Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.frame.to_parquet(out_path, compression="zstd", index=False)
    log.info(
        "wrote %d rows to %s (%d cached, %d generated)",
        len(grid.frame),
        out_path,
        grid.n_from_cache,
        grid.n_generated,
    )

    _report(grid, compare_real=args.compare_real)


def _report(grid: Q1Grid, *, compare_real: bool) -> None:
    """Print the Q1 report, and the comparison with real email when asked."""
    result = run_q1_analysis(grid)
    print(format_report(result))
    if grid.design == "full":
        _report_full_grid(grid, result)
    if not compare_real:
        return
    if not REAL_MANIFEST_PATH.exists():
        msg = f"no real-email benchmark at {REAL_MANIFEST_PATH}; run thesis.analysis.q1_real first"
        raise FileNotFoundError(msg)
    manifest = json.loads(REAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    print()
    print(format_real_comparison(compare_with_real(grid_contrasts(result), manifest)))


if __name__ == "__main__":
    main()
