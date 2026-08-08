"""Parse raw RFC-822 message blobs from the Enron corpus.

The corpus is 1999-2002 Outlook mail, so it is messy in predictable ways:
malformed or missing ``Date`` headers, non-UTF-8 bytes, display names that
disagree with addresses, and reply bodies that quote the entire preceding
thread. This module turns one raw blob into a flat, typed record.

Two design choices worth stating, because both affect downstream analysis:

* We parse with the legacy ``compat32`` policy, which returns plain strings
  and tolerates defects, rather than ``policy.default``, which raises on
  malformed structured headers. At this corpus's error rate, strictness would
  cost more messages than it saves.
* ``body_clean`` holds only newly authored text: quoted chains and signature
  blocks are removed. That is the unit of analysis for every linguistic
  feature, because quoted text is someone else's writing and would otherwise
  contaminate per-sender style measures.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import Message
from email.parser import BytesParser, Parser
from email.utils import getaddresses, parsedate_to_datetime
from typing import Final

# --------------------------------------------------------------------------
# Quoted-text and signature detection
# --------------------------------------------------------------------------

# Each pattern marks the start of quoted material. We cut at the earliest
# match, so a reply that quotes several ancestors loses all of them at once.
_QUOTE_CUT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^-+\s*Original Message\s*-+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^-+\s*Forwarded by\b.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*_{10,}\s*$", re.MULTILINE),
    re.compile(r"^\s*-{10,}\s*$", re.MULTILINE),
    re.compile(r"^On\b.{0,160}\bwrote:\s*$", re.MULTILINE),
)

# Outlook also forwards by pasting a bare header block into the body. A lone
# "From:" line is too common in ordinary prose to cut on, so we require a
# following Sent:/To:/Date: line within a few lines to confirm it.
_INLINE_FROM = re.compile(r"^\s*From:\s+\S.*$", re.MULTILINE)
_INLINE_CONFIRM = re.compile(r"^\s*(?:Sent|To|Date|Subject):\s+\S", re.MULTILINE)
_INLINE_CONFIRM_WINDOW: Final[int] = 4

_QUOTED_LINE = re.compile(r"^\s*>")

# A line of exactly "--" is the conventional signature delimiter.
_SIG_DELIMITER = re.compile(r"^--\s*$", re.MULTILINE)

# Contact-detail lines: phone/fax numbers, labelled contact fields, or a bare
# email address. Used only to trim the tail of a message.
_CONTACT_LINE = re.compile(
    r"(?:\(\d{3}\)\s*\d{3}[-.\s]?\d{4}"
    r"|\b\d{3}[-.]\d{3}[-.]\d{4}\b"
    r"|\b(?:tel|telephone|fax|phone|mobile|cell|e-?mail|ext)\b\s*[:.]?"
    r"|[\w.+-]+@[\w.-]+\.\w{2,})",
    re.IGNORECASE,
)
_SIG_SCAN_LINES: Final[int] = 6

_WHITESPACE = re.compile(r"\s+")


def strip_quoted_text(body: str) -> str:
    """Remove quoted ancestors, returning only text authored in this message."""
    cut = len(body)
    for pattern in _QUOTE_CUT_PATTERNS:
        match = pattern.search(body)
        if match is not None:
            cut = min(cut, match.start())

    for match in _INLINE_FROM.finditer(body):
        window = body[match.end() : match.end() + 400].split("\n")[:_INLINE_CONFIRM_WINDOW]
        if _INLINE_CONFIRM.search("\n".join(window)) is not None:
            cut = min(cut, match.start())
            break

    head = body[:cut]
    kept = [line for line in head.split("\n") if not _QUOTED_LINE.match(line)]
    return "\n".join(kept)


def strip_signature(body: str) -> str:
    """Remove a trailing signature block.

    Cuts at an explicit ``--`` delimiter when present; otherwise drops trailing
    contact-detail lines. Only the last few lines are considered, so a phone
    number quoted mid-message survives.
    """
    delimiter = _SIG_DELIMITER.search(body)
    if delimiter is not None:
        body = body[: delimiter.start()]

    lines = body.split("\n")
    end = len(lines)
    scanned = 0
    while end > 0 and scanned < _SIG_SCAN_LINES:
        candidate = lines[end - 1]
        if not candidate.strip():
            end -= 1
            continue
        if _CONTACT_LINE.search(candidate) is None:
            break
        end -= 1
        scanned += 1
    return "\n".join(lines[:end])


def clean_body(body: str) -> str:
    """Return newly authored text: quoted chains and signature removed."""
    return strip_signature(strip_quoted_text(body)).strip()


# --------------------------------------------------------------------------
# Header helpers
# --------------------------------------------------------------------------


def normalize_address(raw: str) -> str | None:
    """Lower-case and trim one address, returning None if it is not usable."""
    address = raw.strip().strip("<>").strip().lower()
    if not address or "@" not in address:
        return None
    return address


def extract_addresses(*header_values: str | None) -> tuple[str, ...]:
    """Parse address headers into a de-duplicated, order-preserving tuple."""
    pairs = getaddresses([value for value in header_values if value])
    seen: dict[str, None] = {}
    for _display_name, address in pairs:
        normalized = normalize_address(address)
        if normalized is not None:
            seen.setdefault(normalized, None)
    return tuple(seen)


def parse_date(value: str | None) -> datetime | None:
    """Parse a Date header into a naive UTC datetime, or None if unusable."""
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _decode(payload: bytes, charset: str | None) -> str:
    """Decode payload bytes, falling back to latin-1, which never raises."""
    for encoding in (charset, "utf-8", "cp1252"):
        if encoding:
            try:
                return payload.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
    return payload.decode("latin-1", errors="replace")


def extract_body(msg: Message) -> str:
    """Return the message's plain-text body, walking multipart if needed."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() != "text/plain":
                continue
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes):
                return _decode(payload, part.get_content_charset())
        return ""

    payload = msg.get_payload(decode=True)
    if isinstance(payload, bytes):
        return _decode(payload, msg.get_content_charset())
    raw = msg.get_payload()
    return raw if isinstance(raw, str) else ""


def content_fingerprint(
    from_addr: str | None, date: datetime | None, subject: str, body_clean: str
) -> str:
    """Stable hash used to deduplicate messages that lack a usable Message-ID.

    The corpus stores one copy per mailbox folder, so the same message recurs
    under different paths. Whitespace is normalized because Outlook re-wrapped
    bodies inconsistently between copies.
    """
    normalized_body = _WHITESPACE.sub(" ", body_clean).strip().lower()
    normalized_subject = _WHITESPACE.sub(" ", subject).strip().lower()
    parts = (
        from_addr or "",
        date.isoformat() if date else "",
        normalized_subject,
        normalized_body,
    )
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Record
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedMessage:
    """One parsed Enron message.

    ``X-*`` headers are corpus-specific and retained deliberately: ``x_from``
    carries the sender's display name, which is the strongest signal available
    for clustering address aliases onto the same person, and ``x_folder``
    records which mailbox folder this copy came from.
    """

    source_path: str
    message_id: str | None
    date: datetime | None
    from_addr: str | None
    to_addrs: tuple[str, ...]
    cc_addrs: tuple[str, ...]
    bcc_addrs: tuple[str, ...]
    subject: str
    x_from: str
    x_to: str
    x_folder: str
    x_origin: str
    body_raw: str
    body_clean: str
    content_hash: str

    @property
    def dedup_key(self) -> str:
        """Message-ID when present and trustworthy, else the content hash."""
        return self.message_id or self.content_hash

    @property
    def n_recipients(self) -> int:
        return len(self.to_addrs) + len(self.cc_addrs) + len(self.bcc_addrs)


def parse_message(raw: str | bytes, source_path: str) -> ParsedMessage:
    """Parse one raw message blob into a :class:`ParsedMessage`.

    Accepts ``bytes`` (how files are read during ingest, which lets the email
    library apply the charset declared in the headers) or ``str`` (convenient
    in tests). Never raises on malformed input: unparseable fields become
    ``None`` or empty strings so a single bad message cannot abort a
    500k-file ingest.
    """
    msg = BytesParser().parsebytes(raw) if isinstance(raw, bytes) else Parser().parsestr(raw)

    message_id_raw = msg.get("Message-ID")
    message_id = message_id_raw.strip().strip("<>") if message_id_raw else None

    from_addrs = extract_addresses(msg.get("From"))
    subject = (msg.get("Subject") or "").strip()
    body_raw = extract_body(msg)
    body_clean = clean_body(body_raw)
    date = parse_date(msg.get("Date"))
    from_addr = from_addrs[0] if from_addrs else None

    return ParsedMessage(
        source_path=source_path,
        message_id=message_id or None,
        date=date,
        from_addr=from_addr,
        to_addrs=extract_addresses(msg.get("To")),
        cc_addrs=extract_addresses(msg.get("Cc")),
        bcc_addrs=extract_addresses(msg.get("Bcc")),
        subject=subject,
        x_from=(msg.get("X-From") or "").strip(),
        x_to=(msg.get("X-To") or "").strip(),
        x_folder=(msg.get("X-Folder") or "").strip(),
        x_origin=(msg.get("X-Origin") or "").strip(),
        body_raw=body_raw,
        body_clean=body_clean,
        content_hash=content_fingerprint(from_addr, date, subject, body_clean),
    )
