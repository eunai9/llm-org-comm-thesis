"""The composite power score: Layer A + Layer B, combined per message.

Joins the per-message linguistic features (``features.py``) with the
per-sender network features (``network.py``) onto every message, z-scores
each component, and combines them with the weights frozen in
``configs/data.yaml`` -- frozen *before* this module was written, let alone
run, so there is no way the weights could have been tuned to the validation
result computed at the end of this file.

**Handling missing components.** ``reply_latency_asymmetry`` is null for any
sender with fewer than :data:`network.MIN_LATENCY_OBSERVATIONS` replies in
either direction -- a large share of senders, since most people in this
corpus do not have enough reconstructed conversation turns to support a
stable median. Rather than drop those messages (losing a lot of the corpus)
or impute a value (fabricating a signal), the composite is the **average of
the weighted z-scores that are actually available** for that message:

    power_score = sum(weight_i * z_i for available i) / sum(|weight_i| for available i)

This keeps the score on a comparable scale regardless of how many components
a given message has, at the cost of the score meaning something slightly
different across messages with different missingness patterns -- a
limitation worth stating in the thesis, not hiding.

**Validation, not tuning.** ``validate_against_seniority`` computes the mean
score by known seniority rank *after* the composite is built. If the mean
does not rise with rank, the plan is explicit: report that finding, do not
go back and adjust the weights until it does. That would be circular, and
it is exactly the trap a statistics committee is positioned to notice.

Run with ``python -m thesis.data.power``.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from thesis.config import Config, load_config
from thesis.data.identity import resolve_owners
from thesis.data.roles import build_role_index, load_employees, load_title_rank_table
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import INTERIM_DIR, MANIFESTS_DIR, MESSAGES_PARQUET_GLOB, ensure_dirs

log = get_logger(__name__)

FEATURES_PATH = INTERIM_DIR / "features.parquet"
NETWORK_PATH = INTERIM_DIR / "sender_network_features.parquet"

_LAYER_A_COLUMNS = (
    "imperative_ratio",
    "hedge_rate",
    "deference_rate",
    "commitment_rate",
    "question_ratio",
)
_LAYER_B_COLUMNS = (
    "n_recipients",
    "is_broadcast",
    "eigenvector_centrality",
    "thread_initiation_rate",
    "last_word_rate",
    "reply_latency_asymmetry",
)


def load_joined_features(
    messages_glob: str = MESSAGES_PARQUET_GLOB,
    features_path: object = FEATURES_PATH,
    network_path: object = NETWORK_PATH,
) -> pd.DataFrame:
    """One row per message: Layer A columns plus the sender's Layer B columns.

    Layer B is a per-sender aggregate broadcast onto every message that
    sender wrote (a left join, since a message's sender might not have
    enough correspondence to have network features at all).
    """
    con = duckdb.connect()
    table = con.execute(
        """
        SELECT
            m.message_uid,
            m.from_addr,
            m.n_recipients,
            a.imperative_ratio,
            a.hedge_rate,
            a.deference_rate,
            a.commitment_rate,
            a.question_ratio,
            CAST(CASE WHEN m.n_recipients >= 10 THEN 1.0 ELSE 0.0 END AS DOUBLE) AS is_broadcast,
            n.eigenvector_centrality,
            n.thread_initiation_rate,
            n.last_word_rate,
            n.reply_latency_asymmetry
        FROM read_parquet(?) AS m
        JOIN read_parquet(?) AS a USING (message_uid)
        LEFT JOIN read_parquet(?) AS n ON n.address = m.from_addr
        """,
        [messages_glob, str(features_path), str(network_path)],
    ).to_arrow_table()
    con.close()
    return table.to_pandas()


def compute_power_score(
    df: pd.DataFrame, config: Config, *, min_components: int = 3
) -> pd.DataFrame:
    """Add a ``power_score`` column: the availability-weighted mean of
    signed z-scored components, per the module docstring's formula.

    Messages with fewer than ``min_components`` available features get a
    null score rather than a score computed from a near-empty basis -- an
    average of one z-score is not a meaningful composite.
    """
    weights: dict[str, float] = {
        **config.data.power.layer_a_weights,
        **config.data.power.layer_b_weights,
    }
    columns = [c for c in (*_LAYER_A_COLUMNS, *_LAYER_B_COLUMNS) if c in weights]
    missing_weight = set(weights) - set(columns)
    if missing_weight:
        msg = f"configs/data.yaml has weights for unknown columns: {missing_weight}"
        raise ValueError(msg)

    result = df.copy()
    z_columns: list[str] = []
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        z_col = f"_z_{column}"
        if std and std > 0:
            result[z_col] = (df[column] - mean) / std
        else:
            result[z_col] = pd.NA
        z_columns.append(z_col)

    weight_array = pd.Series({f"_z_{c}": weights[c] for c in columns})
    z_frame = result[z_columns]
    available = z_frame.notna()
    weighted_sum = z_frame.mul(weight_array, axis=1).sum(axis=1, skipna=True)
    weight_denominator = available.mul(weight_array.abs(), axis=1).sum(axis=1)
    n_available = available.sum(axis=1)

    # A row where every component is unavailable (e.g. every weighted column
    # has zero variance, so nothing z-scores) has a zero denominator, which
    # would otherwise raise ZeroDivisionError rather than simply producing no
    # score. Route it to NaN before dividing -- caught by a test with a
    # single-row, all-constant-feature fixture, not discovered against real
    # data.
    safe_denominator = weight_denominator.mask(weight_denominator == 0)
    score = weighted_sum / safe_denominator
    score[n_available < min_components] = pd.NA
    result["power_score"] = score
    result["n_power_components"] = n_available

    return result.drop(columns=z_columns)


def _seniority_lookup(messages_glob: str) -> dict[str, int]:
    """address -> seniority_rank, reusing roles.py's join exactly."""
    employees = load_employees()
    title_ranks = load_title_rank_table()
    owners = resolve_owners(messages_glob)
    role_index, _ = build_role_index(employees, owners, title_ranks)
    return {address: role.seniority_rank for address, role in role_index.items()}


def validate_against_seniority(df: pd.DataFrame, messages_glob: str) -> dict[str, object]:
    """Mean power_score by known seniority_rank -- the plan's "money plot",
    reported as numbers here; a figure belongs in analysis/, built later.

    Construct validity means this should rise monotonically with rank. If it
    does not, that is reported as a finding, not adjusted away.
    """
    ranks = _seniority_lookup(messages_glob)
    scored = df.dropna(subset=["power_score"]).copy()
    scored["seniority_rank"] = scored["from_addr"].map(ranks)
    known = scored.dropna(subset=["seniority_rank"])

    by_rank = (
        known.groupby("seniority_rank")["power_score"].agg(["mean", "std", "count"]).sort_index()
    )
    means = by_rank["mean"].tolist()
    is_monotonic = all(a <= b for a, b in pairwise(means))

    spearman = (
        known["seniority_rank"].corr(known["power_score"], method="spearman")
        if len(known) > 1
        else None
    )

    return {
        "n_scored_messages": len(scored),
        "n_with_known_rank": len(known),
        "mean_power_score_by_rank": {
            int(rank): {
                "mean": round(float(row["mean"]), 4),
                "std": round(float(row["std"]), 4),
                "n": int(row["count"]),
            }
            for rank, row in by_rank.iterrows()
        },
        "is_monotonic_nondecreasing": is_monotonic,
        "spearman_rank_vs_score": round(float(spearman), 4) if spearman is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default=MESSAGES_PARQUET_GLOB)
    parser.add_argument("--features", default=str(FEATURES_PATH))
    parser.add_argument("--network", default=str(NETWORK_PATH))
    parser.add_argument("--out", default=str(INTERIM_DIR / "power_scores.parquet"))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    config = load_config()
    df = load_joined_features(args.messages, args.features, args.network)
    log.info("joined feature table: %d rows", len(df))

    scored = compute_power_score(df, config)
    n_scored = int(scored["power_score"].notna().sum())
    log.info("power_score computed for %d / %d messages", n_scored, len(scored))

    out_table = pa.Table.from_pandas(
        scored[["message_uid", "from_addr", "power_score", "n_power_components"]],
        preserve_index=False,
    )
    pq.write_table(out_table, args.out, compression="zstd")
    log.info("wrote %s", args.out)

    validation = validate_against_seniority(scored, args.messages)
    (MANIFESTS_DIR / "power_score_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    by_rank: dict[int, dict[str, float]] = validation["mean_power_score_by_rank"]  # type: ignore[assignment]
    log.info("=== validation: mean power_score by seniority_rank ===")
    for rank, stats in sorted(by_rank.items()):
        log.info(
            "  rank %d: mean=%+.4f  std=%.4f  n=%d",
            rank,
            stats["mean"],
            stats["std"],
            int(stats["n"]),
        )
    log.info("monotonic non-decreasing: %s", validation["is_monotonic_nondecreasing"])
    log.info("Spearman(rank, score):    %s", validation["spearman_rank_vs_score"])


if __name__ == "__main__":
    main()
