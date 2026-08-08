"""Walk the Enron maildir, parse every message, and write Parquet parts.

The corpus stores one copy of a message per mailbox folder it appears in, so
the sender's Sent copy and each recipient's Inbox copy are separate files on
disk. The raw file count is therefore roughly double the number of distinct
messages, and quoting the raw figure as the study's N would overstate it.
Deduplication happens here, and the before/after counts are written to
``ingest_report.json`` so they can be reported in the thesis.

Output:

``data/interim/messages/part-*.parquet``
    One row per unique message.
``data/interim/recipients.parquet``
    One row per message x recipient, exploded from To/Cc/Bcc. Network
    measures need this shape; keeping it separate avoids repeating the body
    text once per recipient.
``data/interim/ingest_report.json``
    Counts for the August go/no-go memo.

Run with ``python -m thesis.data.ingest``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from thesis.data.rfc822 import ParsedMessage, parse_message
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import INTERIM_DIR, RAW_DIR, ensure_dirs

log = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 20_000

MESSAGE_SCHEMA = pa.schema(
    [
        pa.field("message_uid", pa.string(), nullable=False),
        pa.field("message_id", pa.string()),
        pa.field("source_path", pa.string(), nullable=False),
        # Parquet has no second-resolution timestamp, so a "s" field is
        # silently widened to "ms" on write. Declare "ms" up front so the
        # schema we assert on is the schema that actually round-trips.
        pa.field("date", pa.timestamp("ms")),
        pa.field("from_addr", pa.string()),
        pa.field("subject", pa.string()),
        pa.field("x_from", pa.string()),
        pa.field("x_to", pa.string()),
        pa.field("x_folder", pa.string()),
        pa.field("x_origin", pa.string()),
        pa.field("n_to", pa.int32()),
        pa.field("n_cc", pa.int32()),
        pa.field("n_bcc", pa.int32()),
        pa.field("n_recipients", pa.int32()),
        pa.field("body_raw", pa.string()),
        pa.field("body_clean", pa.string()),
        pa.field("n_tokens_clean", pa.int32()),
        pa.field("is_empty_after_clean", pa.bool_()),
    ]
)

RECIPIENT_SCHEMA = pa.schema(
    [
        pa.field("message_uid", pa.string(), nullable=False),
        pa.field("address", pa.string(), nullable=False),
        pa.field("field", pa.string(), nullable=False),
    ]
)


@dataclass
class IngestReport:
    """Counts that belong in the thesis, not just in a log line."""

    files_scanned: int = 0
    files_unreadable: int = 0
    unique_messages: int = 0
    duplicate_copies: int = 0
    deduped_by_message_id: int = 0
    deduped_by_content_hash: int = 0
    empty_after_cleaning: int = 0
    missing_date: int = 0
    missing_from: int = 0
    recipient_rows: int = 0
    parts_written: int = 0

    @property
    def duplication_factor(self) -> float:
        """Raw files per unique message. Expected to be roughly 2."""
        if self.unique_messages == 0:
            return 0.0
        return self.files_scanned / self.unique_messages

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = dict(asdict(self))
        payload["duplication_factor"] = round(self.duplication_factor, 4)
        return payload


@dataclass
class _Buffer:
    """Column-oriented accumulator flushed to a Parquet part when full."""

    messages: list[dict[str, object]] = field(default_factory=list)
    recipients: list[dict[str, object]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.messages)

    def clear(self) -> None:
        self.messages.clear()
        self.recipients.clear()


def iter_message_files(maildir: Path) -> Iterator[Path]:
    """Yield every message file under the maildir, in a deterministic order.

    Sorting matters: it makes ingest reproducible, which in turn makes the
    *first* copy of a duplicated message a stable choice rather than one that
    depends on filesystem iteration order.
    """
    yield from sorted(path for path in maildir.rglob("*") if path.is_file())


def _message_row(msg: ParsedMessage, uid: str) -> dict[str, object]:
    return {
        "message_uid": uid,
        "message_id": msg.message_id,
        "source_path": msg.source_path,
        "date": msg.date,
        "from_addr": msg.from_addr,
        "subject": msg.subject,
        "x_from": msg.x_from,
        "x_to": msg.x_to,
        "x_folder": msg.x_folder,
        "x_origin": msg.x_origin,
        "n_to": len(msg.to_addrs),
        "n_cc": len(msg.cc_addrs),
        "n_bcc": len(msg.bcc_addrs),
        "n_recipients": msg.n_recipients,
        "body_raw": msg.body_raw,
        "body_clean": msg.body_clean,
        "n_tokens_clean": len(msg.body_clean.split()),
        "is_empty_after_clean": not msg.body_clean,
    }


def _recipient_rows(msg: ParsedMessage, uid: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, addresses in (("to", msg.to_addrs), ("cc", msg.cc_addrs), ("bcc", msg.bcc_addrs)):
        rows.extend(
            {"message_uid": uid, "address": address, "field": label} for address in addresses
        )
    return rows


def _flush(buffer: _Buffer, out_dir: Path, part_index: int) -> None:
    table = pa.Table.from_pylist(buffer.messages, schema=MESSAGE_SCHEMA)
    pq.write_table(table, out_dir / f"part-{part_index:05d}.parquet", compression="zstd")


def ingest(
    maildir: Path,
    out_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
) -> IngestReport:
    """Parse every message under ``maildir`` into deduplicated Parquet parts."""
    messages_dir = out_dir / "messages"
    messages_dir.mkdir(parents=True, exist_ok=True)
    for stale in messages_dir.glob("part-*.parquet"):
        stale.unlink()

    report = IngestReport()
    buffer = _Buffer()
    all_recipients: list[dict[str, object]] = []
    seen: set[str] = set()
    part_index = 0

    for path in iter_message_files(maildir):
        if limit is not None and report.files_scanned >= limit:
            break
        report.files_scanned += 1

        try:
            raw = path.read_bytes()
        except OSError:
            report.files_unreadable += 1
            continue

        relative = str(path.relative_to(maildir))
        msg = parse_message(raw, relative)
        uid = msg.dedup_key

        if uid in seen:
            report.duplicate_copies += 1
            continue
        seen.add(uid)

        if msg.message_id:
            report.deduped_by_message_id += 1
        else:
            report.deduped_by_content_hash += 1
        if msg.date is None:
            report.missing_date += 1
        if msg.from_addr is None:
            report.missing_from += 1
        if not msg.body_clean:
            report.empty_after_cleaning += 1

        report.unique_messages += 1
        buffer.messages.append(_message_row(msg, uid))
        all_recipients.extend(_recipient_rows(msg, uid))

        if len(buffer) >= chunk_size:
            _flush(buffer, messages_dir, part_index)
            log.info("wrote part %05d (%d messages)", part_index, len(buffer))
            buffer.clear()
            part_index += 1

    if len(buffer) > 0:
        _flush(buffer, messages_dir, part_index)
        log.info("wrote part %05d (%d messages)", part_index, len(buffer))
        part_index += 1

    report.parts_written = part_index
    report.recipient_rows = len(all_recipients)

    recipients_table = pa.Table.from_pylist(all_recipients, schema=RECIPIENT_SCHEMA)
    pq.write_table(recipients_table, out_dir / "recipients.parquet", compression="zstd")

    (out_dir / "ingest_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maildir", type=Path, default=RAW_DIR / "maildir")
    parser.add_argument("--out-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--limit", type=int, default=None, help="Stop after N files (for smoke tests)."
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    if not args.maildir.is_dir():
        msg = f"maildir not found at {args.maildir}; run scripts/fetch_enron.sh first"
        raise SystemExit(msg)

    report = ingest(args.maildir, args.out_dir, chunk_size=args.chunk_size, limit=args.limit)

    log.info("files scanned          %8d", report.files_scanned)
    log.info("unique messages        %8d", report.unique_messages)
    log.info("duplicate copies       %8d", report.duplicate_copies)
    log.info("duplication factor     %8.2f", report.duplication_factor)
    log.info("empty after cleaning   %8d", report.empty_after_cleaning)
    log.info("missing date           %8d", report.missing_date)
    log.info("missing from           %8d", report.missing_from)
    log.info("recipient rows         %8d", report.recipient_rows)


if __name__ == "__main__":
    main()
