"""Q1 on real email: does the same person write differently up, across and down?

The simulator's Q1 test (PROGRESS.md section 39) found almost no direction
effect. That null had nothing to be compared against. This module measures
the same thing in real Enron email, with the same measures and the same
models, so the two can be put side by side.

Direction comes from job-title rank. Sender and recipient must both be on
the 156-person employee list. Up means the recipient ranks higher. Lateral
means the same rank. Down means lower.

Two samples:

- Strict (primary): exactly one To recipient, no cc or bcc, recipient ranked.
- Loose (robustness): every To recipient ranked, all in the same direction.
  Cc is allowed.

Both drop a recipient who is the sender under another address. Every ranked
address belongs to a known person, so this can be checked for every email in
either sample. Both keep only emails inside the sampling frame
(``sampling.eligible_pool``): the token band, the study dates, an internal
sender.

Direction is tied to the sender's own rank. A rank 1 employee cannot write
down. A rank 6 executive cannot write up. So the primary models control for
sender rank and add a random intercept per sender. A second check uses
sender dummies instead. Then every contrast comes only from differences
inside one sender's own email.

Features are recomputed on ``body_clean`` with the exact calls ``q1.py``
uses on generated replies. spaCy runs only on the selected emails, a few
thousand. Never on the whole corpus: that crashed WSL twice (section 37).

Decision is not measured. Real email has no decision field.

No model calls. Run with ``python -m thesis.analysis.q1_real``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats
from statsmodels.stats.multitest import multipletests

from thesis.analysis.hierarchy import (
    FixedEffectsResult,
    MixedModelResult,
    SentenceModelResult,
    fit_direction_fixed_effects,
    fit_direction_mixed_model,
    fit_sentence_level_model,
)
from thesis.analysis.plots import plot_factor_interaction
from thesis.analysis.q1 import parse_replies
from thesis.config import load_config
from thesis.data.features import extract_features, extract_sentence_features
from thesis.data.identity import resolve_owners
from thesis.data.power import FEATURES_PATH
from thesis.data.roles import build_role_index, load_employees, load_title_rank_table
from thesis.data.sampling import eligible_pool
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import (
    DOCS_FIGURES_DIR,
    INTERIM_DIR,
    MANIFESTS_DIR,
    MESSAGES_PARQUET_GLOB,
    RECIPIENTS_PARQUET,
    ensure_dirs,
)

log = get_logger(__name__)

EMAILS_PATH: Path = INTERIM_DIR / "q1_real_emails.parquet"
SENTENCES_PATH: Path = INTERIM_DIR / "q1_real_sentences.parquet"
THREADS_PATH: Path = INTERIM_DIR / "threads.parquet"
MANIFEST_PATH: Path = MANIFESTS_DIR / "q1_real.json"

DIRECTIONS: tuple[str, ...] = ("down", "lateral", "up")
REFERENCE = "lateral"
CONTRASTS: tuple[str, ...] = ("up", "down")
SENDER_COL = "sender_id"
RANK_COL = "sender_rank_level"
EMAIL_COL = "message_uid"
EMAIL_OUTCOMES: tuple[str, ...] = ("imperative_ratio", "hedge_rate")
SENTENCE_OUTCOME = "is_imperative"
OUTCOMES: tuple[str, ...] = ("imperative_ratio", SENTENCE_OUTCOME, "hedge_rate")

MESSAGE_COLUMNS: tuple[str, ...] = (
    "message_uid",
    "from_addr",
    "sender_id",
    "sender_rank",
    "direction",
    "in_frame",
)

# Section 39's simulator result, reproduced from
# data/interim/q1_direction_grid.parquet on Sep 14. Contrasts are
# (coefficient, p) against lateral. Levels are the model's own prediction per
# direction, which in that balanced design equals the plain mean.
SIMULATOR_CONTRASTS: dict[str, dict[str, tuple[float, float]]] = {
    "imperative_ratio": {"up": (0.083, 0.192), "down": (0.027, 0.672)},
    "is_imperative": {"up": (0.395, 0.046), "down": (0.163, 0.401)},
    "hedge_rate": {"up": (0.025, 0.525), "down": (0.027, 0.491)},
}
SIMULATOR_LEVELS: dict[str, dict[str, float]] = {
    "imperative_ratio": {"down": 0.317, "lateral": 0.290, "up": 0.373},
    "is_imperative": {"down": 0.317, "lateral": 0.283, "up": 0.369},
    "hedge_rate": {"down": 0.094, "lateral": 0.067, "up": 0.092},
}

_PLOT_LEVELS: tuple[str, ...] = ("writing down", "writing to a peer", "writing up")


def load_people(messages_glob: str = MESSAGES_PARQUET_GLOB) -> dict[str, tuple[int, int]]:
    """address -> (seniority rank, employee id), from roles.py's own join."""
    role_index, _ = build_role_index(
        load_employees(), resolve_owners(messages_glob), load_title_rank_table()
    )
    return {
        address: (role.seniority_rank, role.employee_id) for address, role in role_index.items()
    }


def load_to_rows(
    people: Mapping[str, tuple[int, int]],
    frame_uids: Iterable[str],
    *,
    min_tokens: int,
    max_tokens: int,
    messages_glob: str = MESSAGES_PARQUET_GLOB,
    recipients_path: Path = RECIPIENTS_PARQUET,
) -> pd.DataFrame:
    """One row per To recipient of every non-empty message in the token band
    whose sender is ranked. Recipient rank and person are null when the
    recipient is not on the list. ``in_frame`` marks the sampling frame."""
    people_table = pa.Table.from_pylist(
        [{"address": a, "rank": r, "person_id": p} for a, (r, p) in people.items()],
        schema=pa.schema(
            [
                pa.field("address", pa.string()),
                pa.field("rank", pa.int32()),
                pa.field("person_id", pa.int32()),
            ]
        ),
    )
    frame_table = pa.table({"message_uid": pa.array(sorted(set(frame_uids)), type=pa.string())})
    con = duckdb.connect()
    con.register("people", people_table)
    con.register("frame", frame_table)
    rows = con.execute(
        """
        SELECT m.message_uid, m.from_addr, m.n_to,
               coalesce(m.n_cc, 0) AS n_cc, coalesce(m.n_bcc, 0) AS n_bcc,
               ps.rank AS sender_rank, ps.person_id AS sender_id,
               rc.address AS to_addr, pr.rank AS recipient_rank, pr.person_id AS recipient_id,
               f.message_uid IS NOT NULL AS in_frame
        FROM read_parquet(?) AS m
        JOIN people AS ps ON ps.address = m.from_addr
        JOIN read_parquet(?) AS rc ON rc.message_uid = m.message_uid AND rc.field = 'to'
        LEFT JOIN people AS pr ON pr.address = rc.address
        LEFT JOIN frame AS f ON f.message_uid = m.message_uid
        WHERE NOT m.is_empty_after_clean AND m.n_tokens_clean BETWEEN ? AND ?
        """,
        [messages_glob, str(recipients_path), min_tokens, max_tokens],
    ).df()
    con.close()
    return rows


def assign_direction(sender_rank: pd.Series, recipient_rank: pd.Series) -> pd.Series:
    """Up if the recipient ranks higher, lateral if equal, down if lower."""
    sign = np.sign(recipient_rank.astype(float) - sender_rank.astype(float))
    return sign.map({1.0: "up", 0.0: "lateral", -1.0: "down"})


def drop_self_recipients(rows: pd.DataFrame, *, by_person: bool = True) -> pd.DataFrame:
    """Remove To rows that point back at the sender.

    Without ``by_person`` only the sender's own address counts. With it, the
    sender's other addresses count too: an email to yourself at a second
    address is not a message to a peer.
    """
    is_self = (rows["to_addr"] == rows["from_addr"]).to_numpy(dtype=bool)
    if by_person:
        # recipient_id is null for an unranked recipient. The comparison must
        # read that as "not self", or those rows vanish and an email with an
        # unranked recipient looks fully ranked.
        same_person = rows["recipient_id"].eq(rows["sender_id"]).fillna(False)
        is_self = is_self | same_person.to_numpy(dtype=bool)
    return rows[~is_self]


def _one_row_per_message(rows: pd.DataFrame) -> pd.DataFrame:
    first = rows.drop_duplicates("message_uid")
    return (
        first.assign(direction=assign_direction(first["sender_rank"], first["recipient_rank"]))
        .loc[:, list(MESSAGE_COLUMNS)]
        .reset_index(drop=True)
    )


def select_strict(rows: pd.DataFrame) -> pd.DataFrame:
    """Messages with exactly one To recipient, no cc or bcc, recipient ranked."""
    single = rows[
        (rows["n_to"] == 1)
        & (rows["n_cc"] == 0)
        & (rows["n_bcc"] == 0)
        & rows["recipient_rank"].notna()
    ]
    single = single[single.groupby("message_uid")["message_uid"].transform("size") == 1]
    return _one_row_per_message(single)


def select_loose(rows: pd.DataFrame) -> pd.DataFrame:
    """Messages whose To recipients are all ranked and all in one direction.
    Cc and bcc are allowed."""
    signs = np.sign(rows["recipient_rank"].astype(float) - rows["sender_rank"].astype(float))
    working = rows.assign(_sign=signs)
    per_message = working.groupby("message_uid").agg(
        n_rows=("to_addr", "size"),
        n_ranked=("recipient_rank", "count"),
        low=("_sign", "min"),
        high=("_sign", "max"),
    )
    keep = per_message.index[
        (per_message["n_rows"] == per_message["n_ranked"])
        & (per_message["low"] == per_message["high"])
    ]
    return _one_row_per_message(rows[rows["message_uid"].isin(keep)])


def mark_replies(selected: pd.DataFrame, threads: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_reply``: the email is not the first message of its
    conversation. An email missing from ``threads`` counts as not a reply."""
    is_root = selected["message_uid"].map(threads.set_index("message_uid")["is_root"])
    return selected.assign(is_reply=is_root.eq(False).to_numpy())


def direction_counts(selected: pd.DataFrame) -> dict[str, Any]:
    """Emails and senders per direction, and how many senders write in
    more than one direction."""
    per_sender = selected.groupby("sender_id")["direction"].nunique()
    by_rank = pd.crosstab(selected["sender_rank"], selected["direction"])
    return {
        "n_emails": len(selected),
        "emails_by_direction": {d: int((selected["direction"] == d).sum()) for d in DIRECTIONS},
        "senders_by_direction": {
            d: int(selected.loc[selected["direction"] == d, "sender_id"].nunique())
            for d in DIRECTIONS
        },
        "n_senders": int(selected["sender_id"].nunique()),
        "n_senders_2plus_directions": int((per_sender >= 2).sum()),
        "n_senders_all_3_directions": int((per_sender == 3).sum()),
        "emails_by_sender_rank_and_direction": {
            str(int(rank)): {d: int(row.get(d, 0)) for d in DIRECTIONS}
            for rank, row in by_rank.iterrows()
        },
    }


def load_bodies(uids: Iterable[str], messages_glob: str = MESSAGES_PARQUET_GLOB) -> pd.DataFrame:
    """``body_clean`` for the given messages only."""
    con = duckdb.connect()
    con.register("wanted", pa.table({"message_uid": pa.array(sorted(set(uids)), type=pa.string())}))
    bodies = con.execute(
        """
        SELECT m.message_uid, m.body_clean
        FROM read_parquet(?) AS m JOIN wanted USING (message_uid)
        ORDER BY m.message_uid
        """,
        [messages_glob],
    ).df()
    con.close()
    return bodies


def measure_emails(bodies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Email-level and sentence-level features, from the same calls ``q1.py``
    makes on generated replies: ``parse_replies``, ``extract_features`` and
    ``extract_sentence_features``."""
    frame = bodies.rename(columns={"message_uid": "cell_id", "body_clean": "body"})
    docs = parse_replies(frame)
    uids = frame["cell_id"].tolist()
    emails = pd.DataFrame([asdict(extract_features(uid, docs[uid])) for uid in uids])
    sentences = pd.DataFrame(
        [asdict(s) for uid in uids for s in extract_sentence_features(uid, docs[uid])]
    )
    return emails, sentences


def imperative_agreement(recomputed: pd.DataFrame, stored: pd.DataFrame) -> dict[str, Any]:
    """How closely the recomputed ``imperative_ratio`` matches the stored
    corpus value for the same messages."""
    merged = recomputed[["message_uid", "imperative_ratio"]].merge(
        stored[["message_uid", "imperative_ratio"]],
        on="message_uid",
        how="left",
        suffixes=("_new", "_stored"),
    )
    both = merged.dropna()
    diff = (both["imperative_ratio_new"] - both["imperative_ratio_stored"]).abs()
    return {
        "n_compared": len(both),
        "n_missing_from_stored": int(merged["imperative_ratio_stored"].isna().sum()),
        "share_within_0.001": round(float((diff < 1e-3).mean()), 4),
        "pearson_r": round(
            float(both["imperative_ratio_new"].corr(both["imperative_ratio_stored"])), 4
        ),
        "max_abs_diff": round(float(diff.max()), 4),
    }


def standardized_levels(
    coefficients: Mapping[str, float],
    frame: pd.DataFrame,
    *,
    covariate: str | None,
    logistic: bool,
) -> dict[str, float]:
    """Mean predicted outcome per direction, random effects at zero.

    Every row is set to each direction in turn and keeps its own covariate
    value, so all three directions are averaged over the same rank mix. With
    ``logistic`` the prediction is a probability.
    """
    base = np.full(len(frame), coefficients["Intercept"])
    if covariate is not None:
        base = base + frame[covariate].map(
            lambda value: coefficients.get(f"{covariate}[T.{value}]", 0.0)
        ).to_numpy(dtype=float)
    levels: dict[str, float] = {}
    for direction in DIRECTIONS:
        shift = 0.0 if direction == REFERENCE else coefficients[f"direction[T.{direction}]"]
        eta = base + shift
        values = 1.0 / (1.0 + np.exp(-eta)) if logistic else eta
        levels[direction] = float(np.mean(values))
    return levels


@dataclass(frozen=True, slots=True)
class ContrastRow:
    """One direction contrast from one model version."""

    version: str
    outcome: str
    level: str
    coefficient: float
    p_value: float
    n_observations: int
    n_senders: int


FitResult = MixedModelResult | SentenceModelResult | FixedEffectsResult


def _contrast_rows(version: str, result: FitResult) -> list[ContrastRow]:
    return [
        ContrastRow(
            version=version,
            outcome=result.outcome,
            level=level,
            coefficient=result.contrast(level)[0],
            p_value=result.contrast(level)[1],
            n_observations=result.n_observations,
            n_senders=result.n_groups,
        )
        for level in CONTRASTS
    ]


def fit_email_model(emails: pd.DataFrame, outcome: str, *, control_rank: bool) -> MixedModelResult:
    """Linear mixed model per email, random intercept per sender."""
    return fit_direction_mixed_model(
        emails,
        outcome,
        cluster_col=SENDER_COL,
        reference=REFERENCE,
        covariates=(RANK_COL,) if control_rank else (),
    )


def fit_sentence_model(
    sentences: pd.DataFrame, *, control_rank: bool, per_email: bool
) -> SentenceModelResult:
    """Logistic mixed model per sentence, random intercept per sender, and
    one per email inside sender when ``per_email``."""
    return fit_sentence_level_model(
        sentences,
        SENTENCE_OUTCOME,
        cluster_col=SENDER_COL,
        reference=REFERENCE,
        covariates=(RANK_COL,) if control_rank else (),
        nested_col=EMAIL_COL if per_email else None,
    )


@dataclass(frozen=True, slots=True)
class PrimaryFits:
    """The three primary models on one sample."""

    orders_per_email: MixedModelResult
    orders_per_sentence: SentenceModelResult
    hedges_per_email: MixedModelResult

    def all(self) -> list[FitResult]:
        return [self.orders_per_email, self.orders_per_sentence, self.hedges_per_email]


def fit_primary(emails: pd.DataFrame, sentences: pd.DataFrame) -> PrimaryFits:
    """Sender rank as a control, random intercept per sender, and for the
    sentence model also one per email."""
    return PrimaryFits(
        orders_per_email=fit_email_model(emails, "imperative_ratio", control_rank=True),
        orders_per_sentence=fit_sentence_model(sentences, control_rank=True, per_email=True),
        hedges_per_email=fit_email_model(emails, "hedge_rate", control_rank=True),
    )


def build_analysis_frames(
    selected: pd.DataFrame, emails: pd.DataFrame, sentences: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join features onto the selected emails, and each sentence onto its
    email's direction, sender and rank."""
    meta = selected.assign(**{RANK_COL: selected["sender_rank"].astype(int).astype(str)})
    email_frame = meta.merge(emails, on="message_uid", how="inner", validate="one_to_one")
    sentence_frame = sentences.merge(
        meta[["message_uid", "direction", SENDER_COL, RANK_COL, "is_reply"]],
        on="message_uid",
        how="inner",
        validate="many_to_one",
    )
    return email_frame, sentence_frame


def _consistency(rows: Sequence[ContrastRow]) -> dict[str, Any]:
    """Per outcome and contrast: how many versions agree on the sign, and how
    many reach p < .05."""
    summary: dict[str, Any] = {}
    for outcome in OUTCOMES:
        for level in CONTRASTS:
            chosen = [r for r in rows if r.outcome == outcome and r.level == level]
            summary[f"{outcome}:{level}"] = {
                "n_versions": len(chosen),
                "n_positive": sum(r.coefficient > 0 for r in chosen),
                "n_p_below_05": sum(r.p_value < 0.05 for r in chosen),
            }
    return summary


def _implied_se(coefficient: float, p_value: float) -> float:
    """The standard error a two-sided normal test implies from a coefficient
    and its p-value."""
    return abs(coefficient) / float(stats.norm.isf(p_value / 2))


def compare_with_simulator(rows: Sequence[ContrastRow]) -> dict[str, dict[str, float]]:
    """Each real contrast next to the simulator's, with a rough z-test of the
    difference.

    Rough because both standard errors are backed out of a coefficient and a
    p-value, and the simulator's p-values are rounded to three decimals.
    """
    comparison: dict[str, dict[str, float]] = {}
    for row in rows:
        sim_coefficient, sim_p = SIMULATOR_CONTRASTS[row.outcome][row.level]
        difference = row.coefficient - sim_coefficient
        se = float(
            np.hypot(_implied_se(row.coefficient, row.p_value), _implied_se(sim_coefficient, sim_p))
        )
        comparison[f"{row.outcome}:{row.level}"] = {
            "real": round(row.coefficient, 4),
            "real_p": float(f"{row.p_value:.4g}"),
            "simulator": sim_coefficient,
            "simulator_p": sim_p,
            "difference": round(difference, 4),
            "difference_p": round(float(2 * stats.norm.sf(abs(difference) / se)), 4),
        }
    return comparison


def run_q1_real(
    *,
    messages_glob: str = MESSAGES_PARQUET_GLOB,
    emails_out: Path = EMAILS_PATH,
    sentences_out: Path = SENTENCES_PATH,
) -> dict[str, Any]:
    """Select, measure and fit everything. Returns the manifest as a dict.
    Writes the per-email and per-sentence tables, without text, to
    ``data/interim``."""
    config = load_config()
    corpus = config.data.corpus
    people = load_people(messages_glob)
    pool = eligible_pool(messages_glob, config, {a: r for a, (r, _) in people.items()})
    rows = load_to_rows(
        people,
        pool["message_uid"],
        min_tokens=corpus.min_body_tokens,
        max_tokens=corpus.max_body_tokens,
        messages_glob=messages_glob,
    )
    log.info("loaded %d To rows from ranked senders", len(rows))

    address_only = drop_self_recipients(rows, by_person=False)
    by_person = drop_self_recipients(rows, by_person=True)
    selection_counts = {
        "strict_self_by_address": len(select_strict(address_only)),
        "loose_self_by_address": len(select_loose(address_only)),
        "strict_self_by_person": len(select_strict(by_person)),
        "loose_self_by_person": len(select_loose(by_person)),
    }

    threads = pd.read_parquet(THREADS_PATH, columns=["message_uid", "is_root"])
    samples: dict[str, pd.DataFrame] = {}
    for name, selector in (("strict", select_strict), ("loose", select_loose)):
        chosen = selector(by_person)
        samples[name] = mark_replies(chosen[chosen["in_frame"]].reset_index(drop=True), threads)
        selection_counts[f"{name}_final_in_frame"] = len(samples[name])

    all_uids = pd.concat([s["message_uid"] for s in samples.values()]).unique()
    log.info("measuring %d emails with spaCy", len(all_uids))
    features, sentences = measure_emails(load_bodies(all_uids, messages_glob))
    stored = pd.read_parquet(FEATURES_PATH, columns=["message_uid", "imperative_ratio"])
    agreement = imperative_agreement(features, stored[stored["message_uid"].isin(all_uids)])

    frames = {name: build_analysis_frames(s, features, sentences) for name, s in samples.items()}
    strict_emails, strict_sentences = frames["strict"]
    loose_emails, loose_sentences = frames["loose"]
    reply_emails = strict_emails[strict_emails["is_reply"]]
    reply_sentences = strict_sentences[strict_sentences["is_reply"]]

    primary = fit_primary(strict_emails, strict_sentences)
    sender_only = fit_sentence_model(strict_sentences, control_rank=True, per_email=False)
    no_rank: list[FitResult] = [
        fit_email_model(strict_emails, "imperative_ratio", control_rank=False),
        fit_sentence_model(strict_sentences, control_rank=False, per_email=False),
        fit_email_model(strict_emails, "hedge_rate", control_rank=False),
    ]
    fixed = [
        fit_direction_fixed_effects(
            strict_emails, "imperative_ratio", cluster_col=SENDER_COL, reference=REFERENCE
        ),
        fit_direction_fixed_effects(
            strict_sentences,
            SENTENCE_OUTCOME,
            cluster_col=SENDER_COL,
            reference=REFERENCE,
            family="logistic",
        ),
        fit_direction_fixed_effects(
            strict_emails, "hedge_rate", cluster_col=SENDER_COL, reference=REFERENCE
        ),
    ]
    loose = fit_primary(loose_emails, loose_sentences)
    replies = fit_primary(reply_emails, reply_sentences)

    versions: dict[str, list[FitResult]] = {
        "strict": primary.all(),
        "strict_sentence_sender_only": [sender_only],
        "strict_no_rank_control": no_rank,
        "strict_sender_fixed_effects": [*fixed],
        "loose": loose.all(),
        "strict_replies_only": replies.all(),
    }
    contrast_rows = [
        row for v, fits in versions.items() for fit in fits for row in _contrast_rows(v, fit)
    ]

    primary_rows = [r for r in contrast_rows if r.version == "strict"]
    holm = multipletests([r.p_value for r in primary_rows], method="holm")[1]

    real_levels = {
        "imperative_ratio": standardized_levels(
            primary.orders_per_email.coefficients, strict_emails, covariate=RANK_COL, logistic=False
        ),
        "is_imperative": standardized_levels(
            primary.orders_per_sentence.coefficients,
            strict_sentences,
            covariate=RANK_COL,
            logistic=True,
        ),
        "hedge_rate": standardized_levels(
            primary.hedges_per_email.coefficients, strict_emails, covariate=RANK_COL, logistic=False
        ),
    }
    sender_only_levels = standardized_levels(
        sender_only.coefficients, strict_sentences, covariate=RANK_COL, logistic=True
    )

    emails_out.parent.mkdir(parents=True, exist_ok=True)
    strict_emails.assign(sample="strict").pipe(
        lambda f: pd.concat([f, loose_emails.assign(sample="loose")], ignore_index=True)
    ).drop(columns=["from_addr"]).to_parquet(emails_out, compression="zstd", index=False)
    strict_sentences.to_parquet(sentences_out, compression="zstd", index=False)

    return {
        "corpus": {
            "n_ranked_addresses": len(people),
            "n_ranked_people": len({p for _, p in people.values()}),
            "n_in_sampling_frame": len(pool),
            "token_band": [corpus.min_body_tokens, corpus.max_body_tokens],
        },
        "selection_counts": selection_counts,
        "samples": {
            name: {
                **direction_counts(s),
                "n_replies": int(s["is_reply"].sum()),
                "replies_by_direction": {
                    d: int((s["is_reply"] & (s["direction"] == d)).sum()) for d in DIRECTIONS
                },
            }
            for name, s in samples.items()
        },
        "measurement": {
            "imperative_ratio_vs_stored": agreement,
            "strict_mean_sentences_per_email": round(float(strict_emails["n_sentences"].mean()), 2),
            "strict_median_sentences_per_email": float(strict_emails["n_sentences"].median()),
            "strict_n_sentences": len(strict_sentences),
            "strict_raw_means_by_direction": {
                d: {
                    "imperative_ratio": round(float(g["imperative_ratio"].mean()), 4),
                    "hedge_rate": round(float(g["hedge_rate"].mean()), 4),
                    "n_sentences": round(float(g["n_sentences"].mean()), 2),
                }
                for d, g in strict_emails.groupby("direction")
            },
        },
        "contrasts": [
            {
                **asdict(r),
                "coefficient": round(r.coefficient, 4),
                "p_value": float(f"{r.p_value:.4g}"),
            }
            for r in contrast_rows
        ],
        "primary_holm_adjusted_p": {
            f"{r.outcome}:{r.level}": round(float(p), 4)
            for r, p in zip(primary_rows, holm, strict=True)
        },
        "simulator_vs_real": compare_with_simulator(primary_rows),
        "consistency_across_versions": _consistency(contrast_rows),
        "variance_components": {
            "strict_orders_per_email_sender_var": round(primary.orders_per_email.group_variance, 5),
            "strict_hedges_per_email_sender_var": round(primary.hedges_per_email.group_variance, 5),
            "strict_sentence_sender_sd": round(primary.orders_per_sentence.group_sd, 4),
            "strict_sentence_email_sd": round(primary.orders_per_sentence.nested_sd or 0.0, 4),
            "strict_sentence_sender_only_sd": round(sender_only.group_sd, 4),
            "fixed_effects_logit_senders_dropped": fixed[1].n_groups_dropped,
        },
        "real_levels_by_direction": {
            outcome: {d: round(v, 4) for d, v in levels.items()}
            for outcome, levels in real_levels.items()
        },
        "real_sentence_sender_only_probabilities": {
            d: round(v, 4) for d, v in sender_only_levels.items()
        },
        "simulator_section_39": {
            "contrasts": {
                o: {k: list(v) for k, v in c.items()} for o, c in SIMULATOR_CONTRASTS.items()
            },
            "levels": SIMULATOR_LEVELS,
        },
    }


def plot_comparison(manifest: Mapping[str, Any], figures_dir: Path) -> list[Path]:
    """Real vs simulated, one figure per orders outcome: two lines over three
    directions. Hedges get no figure. Both sides are flat, and the table says
    that more plainly than a picture."""
    specs = (
        (
            "is_imperative",
            "orders_per_sentence",
            "Orders per sentence: real email vs the simulator",
            "Real: predicted share at the same rank mix. Simulator: section 39.",
            "share of sentences that give an order",
        ),
        (
            "imperative_ratio",
            "orders_per_email",
            "Orders per email: real email vs the simulator",
            "Real: predicted mean at the same rank mix. Simulator: section 39.",
            "mean imperative ratio",
        ),
    )
    paths = []
    for outcome, stem, title, subtitle, y_label in specs:
        real = manifest["real_levels_by_direction"][outcome]
        simulated = SIMULATOR_LEVELS[outcome]
        paths.append(
            plot_factor_interaction(
                _PLOT_LEVELS,
                {
                    "real Enron email": tuple(real[d] for d in DIRECTIONS),
                    "simulator": tuple(simulated[d] for d in DIRECTIONS),
                },
                figures_dir / f"q1_real_{stem}.png",
                title=title,
                subtitle=subtitle,
                x_label="who the writer is writing to",
                y_label=y_label,
                # The simulator line runs high and the real line low, so the
                # empty corner is top left, above the simulator's first point.
                legend_loc="upper left",
            )
        )
    return paths


def format_report(manifest: Mapping[str, Any]) -> str:
    """Plain-text summary: counts, every contrast, and the simulator side by side."""
    lines = ["Q1 on real email", "=" * 60, ""]
    lines.append("selection counts:")
    lines += [f"  {k}: {v}" for k, v in manifest["selection_counts"].items()]
    for name, sample in manifest["samples"].items():
        lines.append(
            f"{name}: {sample['n_emails']} emails {sample['emails_by_direction']}, "
            f"{sample['n_senders']} senders, {sample['n_senders_2plus_directions']} in 2+ "
            f"directions, {sample['n_senders_all_3_directions']} in all 3, "
            f"{sample['n_replies']} replies"
        )
    lines += [
        "",
        f"imperative_ratio vs stored: {manifest['measurement']['imperative_ratio_vs_stored']}",
    ]
    header = f"{'version':<30}{'outcome':<18}{'level':<6}{'coef':>9}{'p':>8}{'n':>7}{'senders':>9}"
    lines += ["", header, "-" * len(header)]
    for r in manifest["contrasts"]:
        lines.append(
            f"{r['version']:<30}{r['outcome']:<18}{r['level']:<6}{r['coefficient']:>9.3f}"
            f"{r['p_value']:>8.3f}{r['n_observations']:>7}{r['n_senders']:>9}"
        )
    lines += ["", "Holm-adjusted p, primary:", f"  {manifest['primary_holm_adjusted_p']}"]
    lines += ["", "real vs simulator contrasts (rough z-test of the difference):"]
    lines += [f"  {k}: {v}" for k, v in manifest["simulator_vs_real"].items()]
    lines += ["", "real (standardized) vs simulator levels, down / lateral / up:"]
    for outcome in OUTCOMES:
        real = manifest["real_levels_by_direction"][outcome]
        sim = SIMULATOR_LEVELS[outcome]
        lines.append(
            f"  {outcome:<18} real "
            + " / ".join(f"{real[d]:.3f}" for d in DIRECTIONS)
            + "   sim "
            + " / ".join(f"{sim[d]:.3f}" for d in DIRECTIONS)
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--figures-dir", default=str(DOCS_FIGURES_DIR))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    manifest = run_q1_real()
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figures = plot_comparison(manifest, Path(args.figures_dir))
    print(format_report(manifest))
    log.info("wrote %s and %d figures", manifest_path, len(figures))


if __name__ == "__main__":
    main()
