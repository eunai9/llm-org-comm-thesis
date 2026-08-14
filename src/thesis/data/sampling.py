"""Draw the three frozen samples the rest of the plan depends on.

The plan specifies "one frame, one seed, three draws": a single eligible
message pool (unique, non-empty, 20-600 tokens, internal, within the study
window, sender's seniority rank known), sampled three ways with one seed:

- ``S_label``      -- messages for LLM labelling (purpose, decision_attitude,
                       sentiment, power_enacted) and power-score validation.
- ``S_shots``       -- real threads used as simulator stimuli.
- ``S_real_eval``   -- the actual real replies inside those same ``S_shots``
                       threads, so a real reply and a generated reply answer
                       the identical stimulus (a paired comparison for the
                       Q2 fidelity claim, per the plan).

Run with ``python -m thesis.data.sampling``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa

from thesis.config import Config, load_config
from thesis.data.corpus_report import _eligibility_sql
from thesis.data.identity import resolve_owners
from thesis.data.roles import build_role_index, load_employees, load_title_rank_table
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import (
    INTERIM_DIR,
    MANIFESTS_DIR,
    MESSAGES_PARQUET_GLOB,
    SAMPLES_DIR,
    ensure_dirs,
)

log = get_logger(__name__)

THREADS_PATH = INTERIM_DIR / "threads.parquet"


def _seniority_by_address(messages_glob: str) -> dict[str, int]:
    """address -> seniority_rank, reusing roles.py's join exactly (same
    pattern as power.py's ``_seniority_lookup``)."""
    employees = load_employees()
    title_ranks = load_title_rank_table()
    owners = resolve_owners(messages_glob)
    role_index, _ = build_role_index(employees, owners, title_ranks)
    return {address: role.seniority_rank for address, role in role_index.items()}


def eligible_pool(
    messages_glob: str = MESSAGES_PARQUET_GLOB,
    config: Config | None = None,
    role_by_address: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """The sampling frame shared by all three draws.

    Reuses corpus_report's eligibility SQL so the coverage numbers already
    reported and the frame sampled from here can never silently drift apart.
    ``role_by_address`` is injectable so tests can supply a synthetic
    sender->rank mapping instead of depending on the real, committed
    employee list matching synthetic fixture addresses.
    """
    config = config or load_config()
    corpus = config.data.corpus
    if role_by_address is None:
        role_by_address = _seniority_by_address(messages_glob)

    filters = [
        corpus.min_body_tokens,
        corpus.max_body_tokens,
        corpus.date_start,
        corpus.date_end,
        f"%@{corpus.internal_domains[0]}",
    ]

    roles_table = pa.Table.from_pylist(
        [{"address": address, "seniority_rank": rank} for address, rank in role_by_address.items()],
        schema=pa.schema(
            [pa.field("address", pa.string()), pa.field("seniority_rank", pa.int32())]
        ),
    )

    con = duckdb.connect()
    con.register("roles", roles_table)
    frame = con.execute(
        f"""
        SELECT m.message_uid, m.from_addr, m.date, r.seniority_rank
        FROM read_parquet(?) AS m
        JOIN roles AS r ON r.address = m.from_addr
        WHERE {_eligibility_sql("m")}
        """,
        [messages_glob, *filters],
    ).df()
    con.close()
    frame["year"] = frame["date"].dt.year
    return frame


def _stratified_sample(
    pool: pd.DataFrame, strata_cols: Sequence[str], n: int, seed: int
) -> pd.DataFrame:
    """An approximately proportional stratified sample of size ``n``.

    Per-stratum quotas are rounded independently, then any rounding
    discrepancy is corrected in the single largest stratum -- simpler than a
    full largest-remainder allocation, and the sample only needs to be
    proportional, not exact to the last message.
    """
    if n >= len(pool):
        return pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    groups = dict(tuple(pool.groupby(list(strata_cols))))
    sizes = pd.Series({key: len(group) for key, group in groups.items()})
    quotas = (sizes * (n / len(pool))).round().astype(int).clip(lower=0)
    quotas.loc[quotas.idxmax()] += n - quotas.sum()

    parts = [
        group.sample(n=min(quotas[key], len(group)), random_state=seed)
        for key, group in groups.items()
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def sample_label(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """S_label: n messages stratified by seniority_rank x year.

    The plan's own design stratifies by seniority_rank x purpose x year, but
    purpose is a Layer-C LLM label that does not exist until this sample has
    been labelled -- stratifying by it here would be circular. Stratifying
    by the two dimensions available up front is the non-circular reading;
    purpose balance can be checked, and corrected for later if needed, once
    labelling is actually done.
    """
    return _stratified_sample(pool, ["seniority_rank", "year"], n, seed)


def sample_shots_and_real_eval(
    pool: pd.DataFrame,
    n_shots: int,
    n_real_eval: int,
    seed: int,
    threads_path: Any = THREADS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S_shots: n_shots real threads used as simulator stimuli. S_real_eval:
    the actual replies inside those same threads.

    A thread qualifies only if every message in it is in the eligible pool
    -- both the stimulus and any reply must meet the same filters everything
    else was sampled from, per "one frame, one seed, three draws". The
    stimulus is each qualifying thread's first message; replies are every
    later message in the chosen threads.
    """
    threads = pd.read_parquet(threads_path)
    conv = threads[threads["is_conversation"]].copy()
    eligible_uids = set(pool["message_uid"])

    # A vectorized isin() over every row, then a vectorized groupby().all(),
    # rather than groupby().apply(lambda ...) -- the latter calls the lambda
    # once per thread (tens of thousands of Python-level calls, each doing
    # its own isin() against the 47k-item eligible set) and was measured at
    # roughly 100 minutes on the real corpus versus a fraction of a second
    # this way, for an identical result.
    conv["_is_eligible"] = conv["message_uid"].isin(eligible_uids)
    fully_eligible = conv.groupby("thread_id")["_is_eligible"].all()
    qualifying_ids = set(fully_eligible[fully_eligible].index)
    conv = conv.drop(columns="_is_eligible")

    stimuli = conv[conv["thread_id"].isin(qualifying_ids) & (conv["position_in_thread"] == 0)]
    chosen_ids = set(stimuli.sample(n=min(n_shots, len(stimuli)), random_state=seed)["thread_id"])

    shots = stimuli[stimuli["thread_id"].isin(chosen_ids)].merge(pool, on="message_uid", how="left")
    replies = conv[conv["thread_id"].isin(chosen_ids) & (conv["position_in_thread"] > 0)].merge(
        pool, on="message_uid", how="left"
    )
    if len(replies) > n_real_eval:
        replies = replies.sample(n=n_real_eval, random_state=seed)

    return shots.reset_index(drop=True), replies.reset_index(drop=True)


def main() -> None:
    configure_logging()
    ensure_dirs()
    config = load_config()
    seed = config.data.seed
    sizes = config.data.sample_sizes

    pool = eligible_pool(config=config)
    log.info("eligible pool: %d messages", len(pool))

    label = sample_label(pool, sizes.label, seed)
    shots, real_eval = sample_shots_and_real_eval(pool, sizes.shots, sizes.real_eval, seed)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    label.to_parquet(SAMPLES_DIR / "s_label.parquet", index=False)
    shots.to_parquet(SAMPLES_DIR / "s_shots.parquet", index=False)
    real_eval.to_parquet(SAMPLES_DIR / "s_real_eval.parquet", index=False)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "eligible_pool_size": len(pool),
        "s_label": {"requested": sizes.label, "drawn": len(label)},
        "s_shots": {"requested": sizes.shots, "drawn": len(shots)},
        "s_real_eval": {"requested": sizes.real_eval, "drawn": len(real_eval)},
    }
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFESTS_DIR / "sampling_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    log.info("S_label drawn:     %d / %d requested", len(label), sizes.label)
    log.info("S_shots drawn:     %d / %d requested", len(shots), sizes.shots)
    log.info("S_real_eval drawn: %d / %d requested", len(real_eval), sizes.real_eval)
    log.info("wrote %s", SAMPLES_DIR)


if __name__ == "__main__":
    main()
