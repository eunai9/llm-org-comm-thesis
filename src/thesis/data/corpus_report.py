"""Produce the August go/no-go figures, from code rather than by hand.

Everything the end-of-August memo asserts about the corpus is computed here
and written to ``outputs/manifests/corpus_report.{json,md}``. Regenerating is
``python -m thesis.data.corpus_report``, so a number in the thesis can always
be traced to a run rather than to a scrollback buffer.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import duckdb

from thesis.config import load_config
from thesis.data.identity import known_address_relation, owner_address_index, resolve_owners
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import INTERIM_DIR, MANIFESTS_DIR, MESSAGES_PARQUET_GLOB, ensure_dirs

log = get_logger(__name__)


def _eligibility_sql(alias: str = "m") -> str:
    """The sampling-frame filter, defined once so report and sampler agree."""
    return (
        f"NOT {alias}.is_empty_after_clean\n"
        f"      AND {alias}.n_tokens_clean BETWEEN ? AND ?\n"
        f"      AND {alias}.date BETWEEN ? AND ?\n"
        f"      AND {alias}.from_addr LIKE ?"
    )


def build_report() -> dict[str, Any]:
    config = load_config()
    corpus = config.data.corpus
    glob = MESSAGES_PARQUET_GLOB
    filters = [
        corpus.min_body_tokens,
        corpus.max_body_tokens,
        corpus.date_start,
        corpus.date_end,
        f"%@{corpus.internal_domains[0]}",
    ]

    ingest_path = INTERIM_DIR / "ingest_report.json"
    ingest = json.loads(ingest_path.read_text(encoding="utf-8")) if ingest_path.is_file() else {}

    owners = resolve_owners(glob)
    index = owner_address_index(owners)

    con = duckdb.connect()
    known_address_relation(con, index)

    totals = con.execute(
        f"""
        SELECT
            count(*),
            count(*) FILTER (WHERE from_addr IN (SELECT address FROM known)),
            count(*) FILTER (WHERE {_eligibility_sql("t")}),
            count(*) FILTER (
                WHERE {_eligibility_sql("t")}
                  AND t.from_addr IN (SELECT address FROM known)
            )
        FROM read_parquet(?) AS t
        """,
        [*filters, *filters, glob],
    ).fetchone()
    assert totals is not None
    unique, known, eligible, eligible_known = (int(v) for v in totals)

    per_owner = con.execute(
        f"""
        SELECT count(*) AS n
        FROM read_parquet(?) AS t
        WHERE t.from_addr IN (SELECT address FROM known)
          AND {_eligibility_sql("t")}
        GROUP BY t.from_addr
        ORDER BY n DESC
        """,
        [glob, *filters],
    ).fetchall()
    con.close()

    counts = [int(row[0]) for row in per_owner]
    median = counts[len(counts) // 2] if counts else 0

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus": {
            "files_on_disk": ingest.get("files_scanned"),
            "distinct_message_ids": ingest.get("distinct_message_ids"),
            "unique_messages": unique,
            "duplication_factor": ingest.get("duplication_factor"),
            "empty_after_cleaning": ingest.get("empty_after_cleaning"),
            "messages_with_threading_headers": ingest.get("has_threading_headers"),
            "recipient_rows": ingest.get("recipient_rows"),
        },
        "identity": {
            "mailboxes_with_outgoing_mail": len(owners),
            "owner_addresses": len(index),
            "owners_with_multiple_addresses": sum(1 for o in owners if len(o.addresses) > 1),
        },
        "coverage": {
            "messages_from_known_owner": known,
            "share_of_unique": round(known / unique, 4) if unique else 0.0,
            "eligible_messages": eligible,
            "eligible_from_known_owner": eligible_known,
            "share_of_eligible": round(eligible_known / eligible, 4) if eligible else 0.0,
            "distinct_senders_in_eligible_pool": len(counts),
            "median_eligible_messages_per_sender": median,
            "senders_with_at_least_100": sum(1 for n in counts if n >= 100),
        },
        "eligibility_filters": {
            "min_body_tokens": corpus.min_body_tokens,
            "max_body_tokens": corpus.max_body_tokens,
            "date_start": corpus.date_start,
            "date_end": corpus.date_end,
            "internal_domains": list(corpus.internal_domains),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    corpus = report["corpus"]
    identity = report["identity"]
    coverage = report["coverage"]

    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    return f"""# Corpus report

Generated {report["generated_at"]} by `python -m thesis.data.corpus_report`.

## Size, after deduplication

| Quantity | Value |
|---|---:|
| Files on disk | {corpus["files_on_disk"]:,} |
| Distinct Message-IDs | {corpus["distinct_message_ids"]:,} |
| **Unique messages** | **{corpus["unique_messages"]:,}** |
| Duplication factor | {corpus["duplication_factor"]}x |
| Recipient rows | {corpus["recipient_rows"]:,} |

`distinct_message_ids` equals the file count exactly. The JavaMail export
minted one Message-ID per file, so deduplication keys on a content
fingerprint instead. **Report {corpus["unique_messages"]:,} as N, not
{corpus["files_on_disk"]:,}.**

## Corpus properties

- Messages with `In-Reply-To`/`References`: **{corpus["messages_with_threading_headers"]}**.
  The export stripped them, so header-based thread reconstruction is
  impossible and a subject-plus-participants fallback is the only option.
- Empty after quote and signature stripping: {corpus["empty_after_cleaning"]:,}
  ({pct(corpus["empty_after_cleaning"] / corpus["unique_messages"])} of unique).
  These are forwards carrying no newly authored text; they are excluded from
  the eligible pool.

## Identity resolution

| Quantity | Value |
|---|---:|
| Mailboxes with outgoing mail | {identity["mailboxes_with_outgoing_mail"]} of 150 |
| Distinct owner addresses | {identity["owner_addresses"]} |
| Owners with more than one address | {identity["owners_with_multiple_addresses"]} |

## Sender coverage — the Q1 gate

| Quantity | Value |
|---|---:|
| Unique messages from a known owner | {coverage["messages_from_known_owner"]:,} ({pct(coverage["share_of_unique"])}) |
| Eligible messages | {coverage["eligible_messages"]:,} |
| **Eligible from a known owner** | **{coverage["eligible_from_known_owner"]:,} ({pct(coverage["share_of_eligible"])})** |
| Distinct senders in that pool | {coverage["distinct_senders_in_eligible_pool"]} |
| Median messages per sender | {coverage["median_eligible_messages_per_sender"]:,} |
| Senders with at least 100 messages | {coverage["senders_with_at_least_100"]} |

**Caveat.** This is coverage by *identifiable person*, not by *title*. Joining
the published employee-title list will reduce it further, since not every
mailbox owner appears there. Treat {pct(coverage["share_of_eligible"])} as the
ceiling on empirical Q1 coverage.
"""


def main() -> None:
    configure_logging()
    ensure_dirs()
    report = build_report()

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFESTS_DIR / "corpus_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (MANIFESTS_DIR / "corpus_report.md").write_text(render_markdown(report), encoding="utf-8")

    log.info("unique messages        %8d", report["corpus"]["unique_messages"])
    log.info("eligible messages      %8d", report["coverage"]["eligible_messages"])
    log.info(
        "eligible, known sender %8d  (%.1f%%)",
        report["coverage"]["eligible_from_known_owner"],
        100 * report["coverage"]["share_of_eligible"],
    )
    log.info("wrote %s", MANIFESTS_DIR / "corpus_report.md")


if __name__ == "__main__":
    main()
