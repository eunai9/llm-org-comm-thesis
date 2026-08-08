"""Ingest tests built on a synthetic maildir.

The fixture reproduces the structural quirk that drives this module's design:
the same message stored twice, once in the sender's Sent folder and once in a
recipient's Inbox. Deduplication has to collapse those, and the counts it
reports go straight into the thesis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from thesis.data.ingest import MESSAGE_SCHEMA, RECIPIENT_SCHEMA, ingest, iter_message_files


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _message(
    *,
    message_id: str | None,
    sender: str,
    to: str,
    subject: str,
    body: str,
    date: str = "Mon, 14 May 2001 16:39:00 -0700 (PDT)",
) -> str:
    header = f"Message-ID: <{message_id}>\n" if message_id else ""
    return (
        f"{header}Date: {date}\nFrom: {sender}\nTo: {to}\n"
        f"Subject: {subject}\nX-From: Sender Name\nX-Folder: \\\\f\n\n{body}\n"
    )


@pytest.fixture
def maildir(tmp_path: Path) -> Path:
    """A maildir with 5 files representing 3 unique messages."""
    root = tmp_path / "maildir"

    shared = _message(
        message_id="dup-1@thyme",
        sender="allen-p@enron.com",
        to="belden-t@enron.com",
        subject="Storage schedule",
        body="Here is the forecast.",
    )
    # Same message, two folder copies -> must collapse to one row.
    _write(root / "allen-p" / "sent" / "1.", shared)
    _write(root / "belden-t" / "inbox" / "1.", shared)

    _write(
        root / "allen-p" / "sent" / "2.",
        _message(
            message_id="uniq-2@thyme",
            sender="allen-p@enron.com",
            to="buy-r@enron.com, skilling-j@enron.com",
            subject="Risk review",
            body="Please review by Friday.",
        ),
    )

    # No Message-ID: dedup must fall back to the content hash.
    no_id = _message(
        message_id=None,
        sender="lay-k@enron.com",
        to="allen-p@enron.com",
        subject="Board meeting",
        body="Agenda attached.",
    )
    _write(root / "lay-k" / "sent" / "1.", no_id)
    _write(root / "allen-p" / "inbox" / "9.", no_id)

    return root


def test_iter_message_files_is_sorted_and_complete(maildir: Path) -> None:
    files = list(iter_message_files(maildir))
    assert len(files) == 5
    assert files == sorted(files)


def test_deduplicates_folder_copies(maildir: Path, tmp_path: Path) -> None:
    report = ingest(maildir, tmp_path / "out")
    assert report.files_scanned == 5
    assert report.unique_messages == 3
    assert report.duplicate_copies == 2


def test_duplication_factor_reported(maildir: Path, tmp_path: Path) -> None:
    report = ingest(maildir, tmp_path / "out")
    assert report.duplication_factor == pytest.approx(5 / 3)


def test_distinct_message_ids_are_reported(maildir: Path, tmp_path: Path) -> None:
    """Evidence for keying on content: IDs do not collapse the way content does."""
    report = ingest(maildir, tmp_path / "out")
    assert report.distinct_message_ids == 2
    assert report.unique_messages == 3


def test_folder_copies_with_different_message_ids_still_dedup(tmp_path: Path) -> None:
    """The real corpus gives each folder copy its own Message-ID.

    This is the regression guard for the bug that made deduplication a no-op:
    keying on Message-ID kept both copies and doubled the corpus.
    """
    root = tmp_path / "maildir"
    shared_body = (
        "Date: Mon, 14 May 2001 16:39:00 -0700 (PDT)\nFrom: a@enron.com\n"
        "To: b@enron.com\nSubject: Storage\nX-From: A\n\nThe forecast.\n"
    )
    _write(root / "a" / "sent" / "1.", "Message-ID: <111.JavaMail@thyme>\n" + shared_body)
    _write(root / "b" / "inbox" / "1.", "Message-ID: <222.JavaMail@thyme>\n" + shared_body)

    report = ingest(root, tmp_path / "out")
    assert report.files_scanned == 2
    assert report.distinct_message_ids == 2
    assert report.unique_messages == 1
    assert report.duplicate_copies == 1


def test_writes_parquet_matching_schema(maildir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    ingest(maildir, out)
    parts = sorted((out / "messages").glob("part-*.parquet"))
    assert len(parts) == 1
    table = pq.read_table(parts[0])
    assert table.schema.equals(MESSAGE_SCHEMA)
    assert table.num_rows == 3


def test_recipients_are_exploded(maildir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    ingest(maildir, out)
    table = pq.read_table(out / "recipients.parquet")
    assert table.schema.equals(RECIPIENT_SCHEMA)
    # 1 + 2 + 1 recipients across the three unique messages.
    assert table.num_rows == 4
    assert set(table.column("field").to_pylist()) == {"to"}


def test_chunking_writes_multiple_parts(maildir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    report = ingest(maildir, out, chunk_size=2)
    assert report.parts_written == 2
    assert len(list((out / "messages").glob("part-*.parquet"))) == 2


def test_rerun_clears_stale_parts(maildir: Path, tmp_path: Path) -> None:
    """A re-run must not leave parts from a previous, larger run behind."""
    out = tmp_path / "out"
    ingest(maildir, out, chunk_size=1)
    assert len(list((out / "messages").glob("part-*.parquet"))) == 3
    ingest(maildir, out, chunk_size=10)
    assert len(list((out / "messages").glob("part-*.parquet"))) == 1


def test_limit_stops_early(maildir: Path, tmp_path: Path) -> None:
    report = ingest(maildir, tmp_path / "out", limit=2)
    assert report.files_scanned == 2


def test_report_is_written_as_json(maildir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    ingest(maildir, out)
    payload = json.loads((out / "ingest_report.json").read_text(encoding="utf-8"))
    assert payload["unique_messages"] == 3
    assert payload["files_scanned"] == 5
    assert "duplication_factor" in payload


def test_unparseable_file_does_not_abort_ingest(maildir: Path, tmp_path: Path) -> None:
    """Binary junk must be absorbed, not raised: one bad file cannot stop 500k."""
    junk = maildir / "allen-p" / "inbox" / "junk."
    junk.write_bytes(b"\x00\xff\xfe not a message at all \x00")
    report = ingest(maildir, tmp_path / "out")
    assert report.files_scanned == 6
    assert report.unique_messages >= 3


def test_empty_body_is_counted(tmp_path: Path) -> None:
    root = tmp_path / "maildir"
    _write(
        root / "u" / "inbox" / "1.",
        _message(
            message_id="fwd@thyme",
            sender="a@enron.com",
            to="b@enron.com",
            subject="FW: report",
            body="-----Original Message-----\nFrom: c@enron.com\n\nThe content.",
        ),
    )
    report = ingest(root, tmp_path / "out")
    assert report.empty_after_cleaning == 1


def test_missing_date_and_sender_are_counted(tmp_path: Path) -> None:
    root = tmp_path / "maildir"
    _write(root / "u" / "inbox" / "1.", "Subject: orphan\n\nNo headers to speak of.\n")
    report = ingest(root, tmp_path / "out")
    assert report.missing_date == 1
    assert report.missing_from == 1
