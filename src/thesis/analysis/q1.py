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
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from spacy.tokens import Doc

from thesis.analysis.hierarchy import (
    AssociationResult,
    MixedModelResult,
    SentenceModelResult,
    direction_decision_association,
    fit_direction_mixed_model,
    fit_sentence_level_model,
    summarize_by_direction,
)
from thesis.config import load_config
from thesis.data.features import _load_nlp, extract_features, extract_sentence_features
from thesis.llm.base import LLMClient
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import CACHE_DIR, COST_LEDGER, INTERIM_DIR, ensure_dirs
from thesis.sim.grid import GridCell, expand, order_for_cache
from thesis.sim.memory import MemoryItem
from thesis.sim.memory_generation import load_frozen_memory
from thesis.sim.persona import Persona, load_frozen_personas
from thesis.sim.run import RunManifest, run_grid
from thesis.sim.scenario import Scenario, Stakes, build_scenarios

log = get_logger(__name__)

Q1_GRID_PATH: Path = INTERIM_DIR / "q1_direction_grid.parquet"

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


def build_q1_cells(
    personas: Sequence[Persona],
    model: str,
    role_label: str,
    *,
    n_replicates: int = 1,
) -> list[GridCell]:
    """Expand and cache-order the Q1 grid: ``len(personas)`` x 24 scenarios x
    ``n_replicates``. Defaults to one replicate, matching every earlier Q1
    pilot -- with 10 personas that is 240 cells."""
    scenarios = build_q1_scenarios()
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


@dataclass(frozen=True, slots=True)
class Q1Grid:
    """The generated (or cache-served) Q1 direction grid, plus provenance."""

    frame: pd.DataFrame
    run_id: str
    model: str
    n_cells: int
    n_from_cache: int
    n_generated: int


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

    cells = build_q1_cells(personas, model, role_label, n_replicates=n_replicates)
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
        design={"kind": "q1_direction_grid", "n_task_types": len(Q1_TASK_STAKES)},
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
    )
    frame = pd.DataFrame.from_records(rows)
    return Q1Grid(
        frame=frame,
        run_id=run_id,
        model=model,
        n_cells=len(cells),
        n_from_cache=manifest.n_from_cache,
        n_generated=manifest.n_generated,
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

    meta = frame[["cell_id", "persona_id", "direction", "scenario_id"]].copy()
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
    the printout should use instead of re-parsing stdout."""

    grid: Q1Grid
    reply_features: pd.DataFrame
    reply_model: MixedModelResult
    sentence_model: SentenceModelResult
    hedge_model: MixedModelResult
    decision_association: AssociationResult
    reply_level_comparison: list[ContrastComparison]
    sentence_level_comparison: list[ContrastComparison]


def run_q1_analysis(grid: Q1Grid) -> Q1Result:
    """Extract both feature grains from a generated grid and fit every Q1
    model this project has used, in one call -- the analysis half of what
    ``main`` reports, kept separate from CLI parsing and generation so it can
    be called directly (from a notebook, or a test) on a grid that already
    exists."""
    docs = parse_replies(grid.frame)
    reply_features = extract_q1_reply_features(grid.frame, docs=docs)
    sentence_features = extract_q1_sentence_features(grid.frame, docs=docs)

    reply_model = fit_direction_mixed_model(reply_features, "imperative_ratio", reference="lateral")
    hedge_model = fit_direction_mixed_model(reply_features, "hedge_rate", reference="lateral")
    sentence_model = fit_sentence_level_model(
        sentence_features, "is_imperative", reference="lateral"
    )
    decision_association = direction_decision_association(reply_features)

    return Q1Result(
        grid=grid,
        reply_features=reply_features,
        reply_model=reply_model,
        sentence_model=sentence_model,
        hedge_model=hedge_model,
        decision_association=decision_association,
        reply_level_comparison=compare_to_historical(reply_model, HISTORICAL_REPLY_LEVEL),
        sentence_level_comparison=compare_to_historical(sentence_model, HISTORICAL_SENTENCE_LEVEL),
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
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        metavar="MODEL",
        required=True,
        help=(
            "Generate with a local Ollama model (e.g. llama3.2:3b): real "
            "generated text, no key, no cost. This project does not call a "
            "paid API -- there is no other way to generate missing cells."
        ),
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
    parser.add_argument("--limit", type=int, default=None, help="Cap cells, for smoke tests.")
    parser.add_argument("--out", default=str(Q1_GRID_PATH))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    from thesis.llm.ollama_client import OllamaClient, OllamaUnavailableError

    client = (
        OllamaClient(args.local, host=args.ollama_host)
        if args.ollama_host
        else OllamaClient(args.local)
    )
    if not client.is_available() and not args.cache_only:
        msg = (
            f"no Ollama server reachable at {client.host}. Start it with "
            f"'ollama serve', and pull the model with 'ollama pull {args.local}'."
        )
        raise OllamaUnavailableError(msg)

    grid = generate_q1_grid(
        client,
        model=args.local,
        cache_only=args.cache_only,
        limit=args.limit,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.frame.to_parquet(out_path, compression="zstd", index=False)
    log.info(
        "wrote %d rows to %s (%d cached, %d generated)",
        len(grid.frame),
        out_path,
        grid.n_from_cache,
        grid.n_generated,
    )

    result = run_q1_analysis(grid)
    print(format_report(result))


if __name__ == "__main__":
    main()
