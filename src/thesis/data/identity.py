"""Resolve who sent what: mailbox owners and their address aliases.

Role analysis needs a person, but the corpus only gives addresses, and people
here have several. The 150 mailbox directories are the one unambiguous roster
available, so this module anchors identity on them: for each mailbox, the
addresses that appear as the sender in that person's own Sent folders are
taken to belong to the mailbox owner.

That anchoring is deliberately conservative. Inferring identity from
free-text display names alone would merge distinct people who share a common
surname, and every merge silently corrupts the per-sender random effects that
the hierarchical models depend on.

The output is the denominator for the coverage question that gates Q1: what
share of messages were sent by someone whose organizational role we can
actually name?
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import duckdb
import pyarrow as pa

from thesis.logging_setup import get_logger

log = get_logger(__name__)

# Folder names that hold a person's own outgoing mail. Enron mailboxes use
# several conventions, and some users have more than one of these.
SENT_FOLDERS: tuple[str, ...] = ("sent", "sent_items", "_sent_mail", "sent items")

# An address must account for at least this share of a mailbox's outgoing mail
# before it is treated as the owner's rather than as noise (a delegate sending
# on someone's behalf, or a mis-filed message).
MIN_ALIAS_SHARE = 0.02

_WHITESPACE = re.compile(r"\s+")


def normalize_display_name(raw: str) -> str:
    """Normalize an X-From display name for comparison.

    Enron display names carry routing decoration -- "Phillip K Allen/HOU/ECT@ECT"
    -- and inconsistent casing and punctuation. Strip to a comparable core.
    """
    name = raw.split("/")[0].split("<")[0]
    name = name.replace('"', " ").replace("'", " ").replace(",", " ")
    return _WHITESPACE.sub(" ", name).strip().lower()


@dataclass(frozen=True, slots=True)
class MailboxOwner:
    """One of the 150 people whose mailbox is in the corpus."""

    mailbox: str
    primary_address: str | None
    addresses: tuple[str, ...]
    display_names: tuple[str, ...]
    n_sent: int

    @property
    def is_resolved(self) -> bool:
        return self.primary_address is not None


def _sent_folder_predicate(column: str) -> str:
    quoted = ", ".join(f"'{name}'" for name in SENT_FOLDERS)
    return f"lower({column}) IN ({quoted})"


def resolve_owners(messages_glob: str) -> list[MailboxOwner]:
    """Map each mailbox directory to the addresses its owner sends from."""
    predicate = _sent_folder_predicate("split_part(source_path, '/', 2)")
    rows = duckdb.execute(
        f"""
        SELECT
            split_part(source_path, '/', 1) AS mailbox,
            from_addr,
            any_value(x_from)               AS display_name,
            count(*)                        AS n
        FROM read_parquet(?)
        WHERE {predicate}
          AND from_addr IS NOT NULL
        GROUP BY mailbox, from_addr
        ORDER BY mailbox, n DESC
        """,
        [messages_glob],
    ).fetchall()

    grouped: dict[str, list[tuple[str, str, int]]] = {}
    for mailbox, address, display_name, count in rows:
        grouped.setdefault(mailbox, []).append((address, display_name or "", int(count)))

    owners: list[MailboxOwner] = []
    for mailbox, entries in sorted(grouped.items()):
        total = sum(count for _, _, count in entries)
        kept = [
            (address, display, count)
            for address, display, count in entries
            if count / total >= MIN_ALIAS_SHARE
        ]
        if not kept:
            kept = entries[:1]
        owners.append(
            MailboxOwner(
                mailbox=mailbox,
                primary_address=kept[0][0],
                addresses=tuple(address for address, _, _ in kept),
                display_names=tuple(sorted({normalize_display_name(d) for _, d, _ in kept if d})),
                n_sent=total,
            )
        )
    return owners


def owner_address_index(owners: Sequence[MailboxOwner]) -> dict[str, str]:
    """Invert owners into address -> mailbox for fast sender lookup.

    An address claimed by more than one mailbox is dropped rather than assigned
    arbitrarily: shared or role accounts belong to no single person, and
    guessing would inject noise straight into the per-sender random effects.
    """
    claims: dict[str, set[str]] = {}
    for owner in owners:
        for address in owner.addresses:
            claims.setdefault(address, set()).add(owner.mailbox)

    contested = {address for address, mailboxes in claims.items() if len(mailboxes) > 1}
    if contested:
        log.info("dropping %d address(es) claimed by multiple mailboxes", len(contested))
    return {
        address: next(iter(mailboxes))
        for address, mailboxes in claims.items()
        if address not in contested
    }


def known_address_relation(
    con: duckdb.DuckDBPyConnection, addresses: Iterable[str], name: str = "known"
) -> None:
    """Register ``addresses`` as a queryable table on ``con``.

    Interpolating addresses into a SQL literal looks simpler and is wrong:
    real Enron addresses contain apostrophes (paul.y'barbo@enron.com), which
    terminate the string literal and raise a parser error. Registering an
    Arrow table sidesteps quoting entirely.
    """
    table = pa.table({"address": pa.array(sorted(set(addresses)), type=pa.string())})
    con.register(name, table)


def sender_coverage(messages_glob: str, index: dict[str, str]) -> dict[str, float | int]:
    """Measure how much of the corpus was sent by a known mailbox owner.

    This is the ceiling on empirical Q1 coverage: a message whose sender we
    cannot name cannot be assigned a hierarchical role either.
    """
    con = duckdb.connect()
    try:
        known_address_relation(con, index)
        row = con.execute(
            """
            SELECT
                count(*)                                      AS total,
                count(*) FILTER (
                    WHERE from_addr IN (SELECT address FROM known)
                )                                             AS known_sender,
                count(*) FILTER (WHERE from_addr IS NOT NULL) AS has_sender
            FROM read_parquet(?)
            """,
            [messages_glob],
        ).fetchone()
    finally:
        con.close()

    assert row is not None
    total, known_sender, has_sender = int(row[0]), int(row[1]), int(row[2])
    return {
        "total_messages": total,
        "messages_with_sender": has_sender,
        "messages_from_known_owner": known_sender,
        "coverage_share": round(known_sender / total, 4) if total else 0.0,
    }
