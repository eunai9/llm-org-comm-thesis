"""Q1 Layer 1: do different judges give the same answer for the same email?

The Layer 1 gate only checks a judge against itself. The lens model (Körner
et al. 2024) asks a second question: consensus, i.e. whether independent
judges agree with each other. A judge can agree with itself and with other
judges and still be wrong about the true direction, so consensus is reported
next to each judge's accuracy.

Two item sets, both reused from :mod:`thesis.analysis.role_inference`:

- **gate**: the 300 Llama replies every judge scored three times for the
  self-consistency gate. These come from the cache.
- **real**: the 1,001 real emails gpt-oss-120b scored in the Layer 1 run.
  Other judges are scored on the same emails.

Run with ``python -m thesis.analysis.judge_consensus``.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from thesis.analysis.draw_stability import DecisionStability, decision_stability_n
from thesis.analysis.judge_swap import client_for_model
from thesis.analysis.q1_real import load_bodies
from thesis.analysis.role_inference import (
    GENERATING_GRIDS,
    REAL_SAMPLE_SEED,
    build_absolute_items_from_grid,
    build_absolute_items_from_real_email,
    judge_self_consistency,
    run_role_inference_absolute,
    sample_gate_frame,
    sample_real_directions,
    summarize_absolute,
)
from thesis.judge.prompt import JudgeItem
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.llm.ollama_client import OllamaClient, OllamaUnavailableError
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import CACHE_DIR, COST_LEDGER, INTERIM_DIR, MANIFESTS_DIR, ensure_dirs
from thesis.sim.scenario import DIRECTIONS

log = get_logger(__name__)

DEFAULT_JUDGES = ("qwen2.5:3b", "openai/gpt-oss-20b@low", "openai/gpt-oss-120b@low")
MANIFEST_PATH = MANIFESTS_DIR / "judge_consensus.json"


def _common_ids(answers: dict[str, dict[str, str]]) -> list[str]:
    sets = [set(by_item) for by_item in answers.values()]
    return sorted(set.intersection(*sets))


def consensus(answers: dict[str, dict[str, str]]) -> DecisionStability:
    """Agreement across judges on the items every judge answered.

    ``answers`` maps judge to {item_id: direction}. Two judges give Cohen's
    kappa, three or more give Fleiss' kappa, as in :func:`decision_stability_n`.
    """
    ids = _common_ids(answers)
    series = [pd.Series([by_item[i] for i in ids]) for by_item in answers.values()]
    return decision_stability_n(series)


def pairwise_kappa(answers: dict[str, dict[str, str]]) -> dict[str, float]:
    """Cohen's kappa for every pair of judges, on the items both answered."""
    kappas: dict[str, float] = {}
    for first, second in combinations(answers, 2):
        pair = {first: answers[first], second: answers[second]}
        ids = _common_ids(pair)
        kappas[f"{first} vs {second}"] = round(
            float(
                cohen_kappa_score(
                    [answers[first][i] for i in ids],
                    [answers[second][i] for i in ids],
                    labels=list(DIRECTIONS),
                )
            ),
            4,
        )
    return kappas


def answer_shares(by_item: dict[str, str]) -> dict[str, float]:
    """Share of items given each direction. A judge that mostly answers one
    direction can look consistent without reading the text."""
    total = len(by_item)
    return {d: round(sum(a == d for a in by_item.values()) / total, 4) for d in DIRECTIONS}


def _score_set(
    items: list[JudgeItem],
    truth: dict[str, str],
    judges: list[str],
    *,
    cache: ResponseCache,
    ledger: CostLedger,
    self_consistency: bool,
) -> dict[str, Any]:
    answers: dict[str, dict[str, str]] = {}
    per_judge: dict[str, Any] = {}
    for judge in judges:
        client = client_for_model(judge)
        if isinstance(client, OllamaClient) and not client.is_available():
            raise OllamaUnavailableError(f"no Ollama server reachable for {judge}")
        results, summary = run_role_inference_absolute(
            items,
            client,
            model=judge,
            cache=cache,
            ledger=ledger,
            run_id="judge_consensus",
            true_directions=truth,
        )
        log.info(
            "%s: %d scored, %d invalid, %d from cache",
            judge,
            summary.n_scored,
            summary.n_invalid,
            summary.n_from_cache,
        )
        answers[judge] = {r.item_id: r.inferred_direction for r in results}
        metrics = summarize_absolute(results)
        per_judge[judge] = {
            "n": metrics.n,
            "n_invalid": summary.n_invalid,
            "balanced_accuracy": metrics.accuracy,
            "ci_low": metrics.ci_low,
            "ci_high": metrics.ci_high,
            "answer_shares": answer_shares(answers[judge]),
        }
        if self_consistency:
            per_judge[judge]["self_kappa"] = round(
                judge_self_consistency(
                    items,
                    client,
                    model=judge,
                    cache=cache,
                    ledger=ledger,
                    run_id="role_inference",
                ).kappa,
                4,
            )
    agreement = consensus(answers)
    return {
        "per_judge": per_judge,
        "consensus": {
            "n_common": agreement.n,
            "share_all_agree": round(agreement.share_agree, 4),
            "share_expected": round(agreement.share_expected, 4),
            "kappa": round(agreement.kappa, 4),
        },
        "pairwise_kappa": pairwise_kappa(answers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", nargs="+", default=list(DEFAULT_JUDGES))
    parser.add_argument("--real-sample", type=int, default=1000)
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()
    cache = ResponseCache(CACHE_DIR)
    ledger = CostLedger(COST_LEDGER)

    llama = pd.read_parquet(GENERATING_GRIDS["llama3.2:3b"])
    gate_items, gate_truth = build_absolute_items_from_grid(
        sample_gate_frame(llama, n=300, random_state=1)
    )

    real_emails = pd.read_parquet(INTERIM_DIR / "q1_real_emails.parquet")
    real_directions = sample_real_directions(
        real_emails[["message_uid", "direction"]].drop_duplicates(subset="message_uid"),
        n=args.real_sample,
        seed=REAL_SAMPLE_SEED,
    )
    real_items, real_truth = build_absolute_items_from_real_email(
        load_bodies(real_directions["message_uid"]), real_directions
    )

    manifest = {
        "judges": args.judges,
        "gate (300 Llama replies)": _score_set(
            gate_items,
            gate_truth,
            args.judges,
            cache=cache,
            ledger=ledger,
            self_consistency=True,
        ),
        "real email": _score_set(
            real_items,
            real_truth,
            args.judges,
            cache=cache,
            ledger=ledger,
            self_consistency=False,
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
