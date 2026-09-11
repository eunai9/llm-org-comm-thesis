"""How much does the simulator follow the reply length it is told to aim for?

The persona prompt states a target: "Typical message length: about N words."
Section 32 showed the model barely reacts to it. That comparison had two
conditions, because it fell out of a persona fix rather than being designed.
Two points show low sensitivity. They cannot give a slope.

This module runs the designed version. One factor, five levels: the stated
target is set to 20, 50, 100, 200 and 400 words. Everything else is held
fixed, including the stimuli, the personas, the memories and the model.

**20 is in the grid on purpose.** The model currently writes about 20 to 23
words. A grid that only rises cannot tell "the model ignores the number" from
"the model cannot write long". Asking for 20 asks whether the instruction can
move output at all, in either direction.

**Every persona gets the same target inside a condition.** The live prompt
gives each persona its own corpus-derived number, and those run from 62 to 93
words. A dose-response design needs one dose per condition, so that spread is
removed here. The cost is that a condition no longer reproduces the live
prompt exactly. The gain is that the dose is the only thing that changes.

**The headline number is an elasticity.** Fit log(words written) on
log(words asked for), with a random intercept per item. The slope says how
much of the instruction survives. 1 means the model follows it. 0 means the
instruction does nothing.

Local models only. No paid API.

Run with ``python -m thesis.analysis.length_dose --local llama3.2:3b``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
from scipy import stats

from thesis.analysis.hierarchy import (
    DoseResponseResult,
    InsufficientDataError,
    fit_dose_response_model,
)
from thesis.analysis.pairs import build_pair_table
from thesis.llm.base import LLMClient
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import INTERIM_DIR, MANIFESTS_DIR, ensure_dirs
from thesis.sim.persona import Persona, load_frozen_personas

log = get_logger(__name__)

LENGTH_DOSE_PATH: Path = INTERIM_DIR / "length_dose_grid.parquet"
MANIFEST_PATH: Path = MANIFESTS_DIR / "length_dose.json"

# Five levels over a 20x range. 20 sits at the model's own habitual length,
# so the grid can move output down as well as up.
TARGET_LEVELS: tuple[int, ...] = (20, 50, 100, 200, 400)

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_ROLE_LABEL = "sim_local"

# 40 items per condition, 5 conditions, 200 local generations. Enough to
# estimate one slope, small enough to run on a laptop in about an hour.
DEFAULT_LIMIT = 40

# A reply "reaches" its target at 80% of it. An exact hit is not a sensible
# bar for free text, and a share measured at 100% would read as zero in every
# condition for reasons that have nothing to do with instruction following.
TARGET_HIT_FRACTION = 0.80


def word_count(text: str) -> int:
    """Whitespace-separated words, the same count the rest of this package uses."""
    return len(str(text).split())


def override_target_length(personas: Sequence[Persona], target: float) -> list[Persona]:
    """Every persona, with its stated typical length replaced by ``target``.

    Only ``style.mean_tokens`` changes. Rank, department, the other style
    statistics and the persona id are untouched, so the cell ids built from
    these personas match across conditions and the items can be paired.
    """
    return [replace(p, style=replace(p.style, mean_tokens=float(target))) for p in personas]


def generate_condition(
    client: LLMClient,
    *,
    target: int,
    model: str = DEFAULT_MODEL,
    role_label: str = DEFAULT_ROLE_LABEL,
    personas: Sequence[Persona] | None = None,
    limit: int | None = DEFAULT_LIMIT,
    cache_only: bool = False,
    progress_every: int = 20,
) -> pd.DataFrame:
    """Generate one reply per stimulus with the stated target set to ``target``.

    Each condition renders different prompt text, so each writes its own cache
    entries and no condition can be served another condition's replies.
    """
    base = personas if personas is not None else load_frozen_personas()
    table = build_pair_table(
        client,
        model=model,
        role_label=role_label,
        personas=override_target_length(base, target),
        cache_only=cache_only,
        limit=limit,
        progress_every=progress_every,
    )
    frame = table.frame.assign(instructed_target=target)
    frame["reply_words"] = frame["generated_reply"].map(word_count)
    return frame


@dataclass(frozen=True, slots=True)
class PairingReport:
    """Whether the same items really appear in every condition."""

    targets: tuple[int, ...]
    n_rows: int
    n_conditions_present: int
    n_items_any: int
    n_items_in_every_condition: int
    n_items_dropped: int

    @property
    def fully_paired(self) -> bool:
        """True when no item is missing from any condition."""
        return self.n_items_dropped == 0 and self.n_conditions_present == len(self.targets)


def check_pairing(frame: pd.DataFrame, targets: Sequence[int] = TARGET_LEVELS) -> PairingReport:
    """Count how many items appear in all conditions, before anything is fitted.

    A slope fitted on items that appear in only some conditions confounds the
    dose with which items happen to be there. The check is cheap, and it stays
    silent about a real problem if it is never run.
    """
    counts = frame.groupby("cell_id")["instructed_target"].nunique()
    complete = int((counts == len(targets)).sum())
    return PairingReport(
        targets=tuple(targets),
        n_rows=len(frame),
        n_conditions_present=int(frame["instructed_target"].nunique()),
        n_items_any=len(counts),
        n_items_in_every_condition=complete,
        n_items_dropped=len(counts) - complete,
    )


def paired_frame(frame: pd.DataFrame, targets: Sequence[int] = TARGET_LEVELS) -> pd.DataFrame:
    """Only the items present in every condition."""
    counts = frame.groupby("cell_id")["instructed_target"].nunique()
    keep = set(counts[counts == len(targets)].index)
    return frame[frame["cell_id"].isin(keep)].copy()


def summarize_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per condition: what was asked for, what came back, how spread out.

    ``share_reaching_target`` is the share of replies at or above
    :data:`TARGET_HIT_FRACTION` of the instructed target.
    """
    rows = []
    for target, group in frame.groupby("instructed_target"):
        words = group["reply_words"]
        rows.append(
            {
                "instructed_target": int(target),
                "n_replies": len(words),
                "mean_words": round(float(words.mean()), 1),
                "median_words": round(float(words.median()), 1),
                "sd_words": round(float(words.std(ddof=1)), 1),
                "iqr_words": round(float(words.quantile(0.75) - words.quantile(0.25)), 1),
                "min_words": int(words.min()),
                "max_words": int(words.max()),
                "share_reaching_target": round(
                    float((words >= TARGET_HIT_FRACTION * target).mean()), 3
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("instructed_target").reset_index(drop=True)


def spread_test(frame: pd.DataFrame) -> tuple[float, float]:
    """Levene's test: is reply length more spread out in some conditions?

    Run on the raw word counts. Returns ``(statistic, p_value)``. A flat mean
    with a changing spread is a different result from nothing moving at all,
    so the spread gets a test rather than a glance at the standard deviations.
    """
    groups = [g["reply_words"].to_numpy(dtype=float) for _, g in frame.groupby("instructed_target")]
    if len(groups) < 2:
        msg = f"need at least 2 conditions to compare spreads; got {len(groups)}"
        raise InsufficientDataError(msg)
    result = stats.levene(*groups)
    return float(result.statistic), float(result.pvalue)


def fit_elasticity(frame: pd.DataFrame) -> DoseResponseResult:
    """Fit log(words written) on log(words asked for), random intercept per item.

    Empty replies are dropped, because log(0) is undefined. Any drop shows up
    as a shortfall in ``n_observations``.
    """
    working = frame[frame["reply_words"] > 0].copy()
    working["log_words"] = [math.log(float(v)) for v in working["reply_words"]]
    working["log_target"] = [math.log(float(v)) for v in working["instructed_target"]]
    return fit_dose_response_model(working, "log_words", "log_target", cluster_col="cell_id")


@dataclass(frozen=True, slots=True)
class LengthDoseResult:
    """One full run: the pairing check, the per-condition table and the slope."""

    model: str
    pairing: PairingReport
    per_condition: pd.DataFrame
    elasticity: DoseResponseResult
    levene_statistic: float
    levene_p_value: float


def run_analysis(frame: pd.DataFrame, *, model: str = DEFAULT_MODEL) -> LengthDoseResult:
    """Check the pairing, then summarize and fit on the items that paired."""
    targets = tuple(sorted(int(t) for t in frame["instructed_target"].unique()))
    pairing = check_pairing(frame, targets)
    paired = paired_frame(frame, targets)
    statistic, p_value = spread_test(paired)
    return LengthDoseResult(
        model=model,
        pairing=pairing,
        per_condition=summarize_conditions(paired),
        elasticity=fit_elasticity(paired),
        levene_statistic=statistic,
        levene_p_value=p_value,
    )


def summary_dict(result: LengthDoseResult) -> dict[str, object]:
    """The run as plain JSON, for the manifest."""
    fit = result.elasticity
    return {
        "model": result.model,
        "targets": list(result.pairing.targets),
        "n_rows": result.pairing.n_rows,
        "n_items_in_every_condition": result.pairing.n_items_in_every_condition,
        "n_items_dropped": result.pairing.n_items_dropped,
        "fully_paired": result.pairing.fully_paired,
        "per_condition": result.per_condition.to_dict(orient="records"),
        "elasticity": {
            "slope": round(fit.slope, 4),
            "std_error": round(fit.slope_se, 4),
            "p_value": float(f"{fit.p_value:.4g}"),
            "conf_int": [round(fit.conf_low, 4), round(fit.conf_high, 4)],
            "n_observations": fit.n_observations,
            "n_items": fit.n_groups,
            "converged": fit.converged,
        },
        "spread_across_conditions": {
            "levene_statistic": round(result.levene_statistic, 3),
            "levene_p_value": float(f"{result.levene_p_value:.4g}"),
        },
    }


def format_report(result: LengthDoseResult) -> str:
    """The report printed at the end of a run."""
    fit = result.elasticity
    return "\n".join(
        [
            f"Length dose-response, model {result.model}",
            "",
            f"Items in every condition: {result.pairing.n_items_in_every_condition} "
            f"of {result.pairing.n_items_any}",
            "",
            result.per_condition.to_string(index=False),
            "",
            f"Elasticity (log words written on log words asked for): {fit.slope:.3f}",
            f"  95% CI {fit.conf_low:.3f} to {fit.conf_high:.3f}, p={fit.p_value:.4g}",
            f"  {fit.n_observations} replies, {fit.n_groups} items",
            "",
            f"Spread across conditions (Levene): W={result.levene_statistic:.2f}, "
            f"p={result.levene_p_value:.4g}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        metavar="MODEL",
        default=DEFAULT_MODEL,
        help="Local Ollama model to generate with. This project calls no paid API.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(str(t) for t in TARGET_LEVELS),
        help="Comma-separated instructed targets, in words.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Serve only from cache; fail rather than call Ollama.",
    )
    parser.add_argument(
        "--analyse-only",
        action="store_true",
        help="Skip generation and analyse the parquet at --out.",
    )
    parser.add_argument("--out", default=str(LENGTH_DOSE_PATH))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    out = Path(args.out)
    targets = [int(t) for t in args.targets.split(",")]

    if args.analyse_only:
        frame = pd.read_parquet(out)
    else:
        from thesis.llm.ollama_client import OllamaClient, OllamaUnavailableError

        client = OllamaClient(args.local)
        if not client.is_available() and not args.cache_only:
            msg = (
                f"no Ollama server reachable at {client.host}. Start it with "
                f"'ollama serve', and pull the model with 'ollama pull {args.local}'."
            )
            raise OllamaUnavailableError(msg)

        out.parent.mkdir(parents=True, exist_ok=True)
        done: list[pd.DataFrame] = []
        for target in targets:
            log.info("condition %d words: starting", target)
            done.append(
                generate_condition(
                    client,
                    target=target,
                    model=args.local,
                    limit=args.limit,
                    cache_only=args.cache_only,
                )
            )
            # Written after every condition, not once at the end, so a run
            # that dies partway still leaves the conditions it finished.
            pd.concat(done, ignore_index=True).to_parquet(out, compression="zstd", index=False)
            log.info(
                "condition %d words: done, mean reply %.1f words, wrote %s",
                target,
                float(done[-1]["reply_words"].mean()),
                out,
            )
        frame = pd.concat(done, ignore_index=True)

    result = run_analysis(frame, model=args.local)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(summary_dict(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(format_report(result))
    log.info("wrote %s and %s", out, manifest)


if __name__ == "__main__":
    main()
