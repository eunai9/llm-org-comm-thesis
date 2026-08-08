"""Identity resolution tests.

The fixture writes a small Parquet store rather than mocking DuckDB, so the
SQL is exercised exactly as it runs in production.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from thesis.data.identity import (
    MailboxOwner,
    normalize_display_name,
    owner_address_index,
    resolve_owners,
    sender_coverage,
)


def _store(tmp_path: Path, rows: list[dict[str, object]]) -> str:
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("source_path", pa.string()),
                pa.field("from_addr", pa.string()),
                pa.field("x_from", pa.string()),
            ]
        ),
    )
    out = tmp_path / "messages"
    out.mkdir(exist_ok=True)
    pq.write_table(table, out / "part-00000.parquet")
    return str(out / "*.parquet")


@pytest.fixture
def store(tmp_path: Path) -> str:
    rows: list[dict[str, object]] = []
    # allen-p sends mostly from one address, plus a rarely used alias.
    for i in range(50):
        rows.append(
            {
                "source_path": f"allen-p/sent/{i}.",
                "from_addr": "phillip.allen@enron.com",
                "x_from": "Phillip K Allen/HOU/ECT@ECT",
            }
        )
    for i in range(5):
        rows.append(
            {
                "source_path": f"allen-p/_sent_mail/{i}.",
                "from_addr": "k..allen@enron.com",
                "x_from": "Allen, Phillip K.",
            }
        )
    # A single stray message must not be promoted to an alias.
    rows.append(
        {
            "source_path": "allen-p/sent/999.",
            "from_addr": "assistant@enron.com",
            "x_from": "Assistant",
        }
    )
    for i in range(20):
        rows.append(
            {
                "source_path": f"beck-s/sent_items/{i}.",
                "from_addr": "sally.beck@enron.com",
                "x_from": "Sally Beck",
            }
        )
    # Inbox mail must be ignored when deciding who owns a mailbox.
    for i in range(30):
        rows.append(
            {
                "source_path": f"allen-p/inbox/{i}.",
                "from_addr": "outsider@example.com",
                "x_from": "Outside Person",
            }
        )
    return _store(tmp_path, rows)


def test_resolves_one_owner_per_mailbox(store: str) -> None:
    owners = {o.mailbox: o for o in resolve_owners(store)}
    assert set(owners) == {"allen-p", "beck-s"}
    assert owners["allen-p"].primary_address == "phillip.allen@enron.com"
    assert owners["beck-s"].primary_address == "sally.beck@enron.com"


def test_keeps_frequent_alias_and_drops_stray_address(store: str) -> None:
    allen = next(o for o in resolve_owners(store) if o.mailbox == "allen-p")
    assert "k..allen@enron.com" in allen.addresses
    assert "assistant@enron.com" not in allen.addresses


def test_inbox_senders_are_not_treated_as_owners(store: str) -> None:
    owners = {o.mailbox: o for o in resolve_owners(store)}
    assert "outsider@example.com" not in owners["allen-p"].addresses


def test_display_names_are_normalized(store: str) -> None:
    allen = next(o for o in resolve_owners(store) if o.mailbox == "allen-p")
    assert "phillip k allen" in allen.display_names


def test_normalize_display_name_strips_routing_and_punctuation() -> None:
    assert normalize_display_name("Phillip K Allen/HOU/ECT@ECT") == "phillip k allen"
    assert normalize_display_name('"Beck, Sally"') == "beck sally"
    assert normalize_display_name("Tim  Belden <tim@enron.com>") == "tim belden"


def test_address_index_maps_every_alias(store: str) -> None:
    index = owner_address_index(resolve_owners(store))
    assert index["phillip.allen@enron.com"] == "allen-p"
    assert index["k..allen@enron.com"] == "allen-p"
    assert index["sally.beck@enron.com"] == "beck-s"


def test_contested_address_is_dropped_not_guessed() -> None:
    """A shared account belongs to nobody; assigning it would inject noise."""
    owners = [
        MailboxOwner("a", "shared@enron.com", ("shared@enron.com",), (), 10),
        MailboxOwner("b", "shared@enron.com", ("shared@enron.com",), (), 10),
    ]
    assert owner_address_index(owners) == {}


def test_sender_coverage_counts_known_owners(store: str) -> None:
    index = owner_address_index(resolve_owners(store))
    stats = sender_coverage(store, index)
    assert stats["total_messages"] == 106
    # 50 + 5 + 20 owner-sent; the stray and the 30 inbox messages are unknown.
    assert stats["messages_from_known_owner"] == 75
    assert stats["coverage_share"] == pytest.approx(75 / 106, abs=1e-4)


def test_sender_coverage_with_empty_index(store: str) -> None:
    stats = sender_coverage(store, {})
    assert stats["messages_from_known_owner"] == 0
    assert stats["coverage_share"] == 0.0


def test_addresses_containing_apostrophes_do_not_break_sql(tmp_path: Path) -> None:
    """Regression: paul.y'barbo@enron.com is a real address in this corpus.

    Interpolating it into a SQL string literal raises a parser error, so the
    address list must be bound rather than formatted into the query.
    """
    rows: list[dict[str, object]] = [
        {"source_path": "ybarbo-p/sent/1.", "from_addr": "paul.y'barbo@enron.com", "x_from": "P"},
        {"source_path": "ybarbo-p/inbox/2.", "from_addr": "other@enron.com", "x_from": "O"},
    ]
    glob = _store(tmp_path, rows)

    index = owner_address_index(resolve_owners(glob))
    assert "paul.y'barbo@enron.com" in index

    stats = sender_coverage(glob, index)
    assert stats["messages_from_known_owner"] == 1
    assert stats["total_messages"] == 2
