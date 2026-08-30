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
# The leading ``[ \t]*`` is not cosmetic: Outlook indents the banner by one
# space often enough that anchoring it hard to column zero left the quoted
# chain in place on a quarter of the replies in the evaluation sample.
_QUOTE_CUT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^[ \t]*-+\s*Original Message\s*-+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[ \t]*-+\s*Forwarded by\b.*$", re.IGNORECASE | re.MULTILINE),
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

# Lotus Notes -- the client most of this corpus was written in -- quotes with
# no banner at all: an indented attribution line ("Marie Heard@ENRON"), an
# indented timestamp, then indented To:/cc:/Subject: lines. None of the
# patterns above sees any of that, which is why it survived into ``body_clean``
# and inflated every length statistic computed from real replies.
#
# The header line is required to be indented *and* confirmed by a second header
# line nearby, for the same reason ``_INLINE_FROM`` is confirmed: an
# unindented "Subject: ..." appears in ordinary prose, and cutting on it alone
# would delete real writing.
_NOTES_HEADER = re.compile(r"^[ \t]+(?:To|cc|Subject|Sent|From|Date):\s*\S", re.MULTILINE)
_NOTES_CONFIRM = re.compile(r"^[ \t]+(?:To|cc|Subject|Sent|From|Date):", re.MULTILINE)

# The attribution and timestamp lines sit *above* the header block, so cutting
# at the header alone would leave them behind as stray text -- a bare sender
# name and a timestamp, which then read as if the author had written them.
# Walk back over at most this many preceding lines, and only over lines that
# are indented like the rest of the block: the author's own prose starts at
# column zero, so indentation is what separates the quote from the writing.
_NOTES_INDENT = re.compile(r"^\t| {4,}")
_NOTES_BACKTRACK_LINES: Final[int] = 3

_QUOTED_LINE = re.compile(r"^\s*>")

# A line of exactly "--" is the conventional signature delimiter.
_SIG_DELIMITER = re.compile(r"^--\s*$", re.MULTILINE)

# Contact-detail lines: phone/fax numbers, labelled contact fields, or a bare
# email address. Used only to trim the tail of a message.
#
# A contact *label* only counts when something contact-shaped follows it -- a
# digit, a "+", or an address. Matching the bare word instead deleted ordinary
# closing sentences: "I received an email from Chris" and "Can you email me
# your form?" were both being discarded as signatures, taking the last thing
# the author actually said with them.
_CONTACT_LINE = re.compile(
    r"(?:\(\d{3}\)\s*\d{3}[-.\s]?\d{4}"
    r"|\b\d{3}[-.]\d{3}[-.]\d{4}\b"
    r"|\b(?:tel|telephone|fax|phone|mobile|cell|e-?mail|ext)\b\s*[:.]?\s*"
    r"(?:\+?\d|\(\d|[\w.+-]+@)"
    r"|[\w.+-]+@[\w.-]+\.\w{2,})",
    re.IGNORECASE,
)
_SIG_SCAN_LINES: Final[int] = 6

_WHITESPACE = re.compile(r"\s+")


def _notes_block_start(body: str) -> int:
    """Offset where a Lotus Notes quoted block begins, or ``len(body)`` if none.

    Finds the first confirmed indented header block, then walks back over the
    attribution and timestamp lines printed above it so the whole block goes,
    not just its headers.
    """
    match = _NOTES_HEADER.search(body)
    if match is None:
        return len(body)
    window = body[match.end() : match.end() + 400].split("\n")[1 : _INLINE_CONFIRM_WINDOW + 1]
    if _NOTES_CONFIRM.search("\n".join(window)) is None:
        return len(body)

    lines = body[: match.start()].split("\n")
    end = len(lines)
    walked = 0
    while end > 0 and walked < _NOTES_BACKTRACK_LINES:
        candidate = lines[end - 1]
        if not candidate.strip():
            end -= 1
            continue
        if _NOTES_INDENT.match(candidate) is None:
            break
        end -= 1
        walked += 1
    return len("\n".join(lines[:end]))


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

    cut = min(cut, _notes_block_start(body))

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


def header_text(msg: Message, name: str) -> str | None:
    """Return a header as plain text, or None if absent.

    Under the ``compat32`` policy ``Message.get`` usually returns ``str``, but
    it returns a ``Header`` object when the raw value carries 8-bit bytes that
    need RFC 2047 decoding -- which happens in this corpus wherever non-ASCII
    names appear in ``X-To`` and friends. Coercing here means no caller has to
    defend against a ``Header`` leaking into a string operation.
    """
    value = msg.get(name)
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


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
    from_addr: str | None, date: datetime | None, subject: str, body_raw: str
) -> str:
    """Stable hash identifying one logical message across its folder copies.

    This is *the* deduplication key -- Message-ID cannot be used. The JavaMail
    export that produced this corpus minted a fresh Message-ID for every file
    it wrote, so the sender's Sent copy and a recipient's Inbox copy of the
    same message carry different IDs. Measured on three mailboxes, Message-IDs
    disagreed in every one of 11k duplicate groups while content agreed in
    over 99% of them, so keying on Message-ID deduplicates nothing and
    overstates the corpus by roughly 2.4x.

    The hash covers sender, timestamp, subject and the *full* body. Using the
    full body rather than the quote-stripped one is deliberate: about 19% of
    messages are forwards with no newly authored text, and those all clean to
    an empty string, so a quote-stripped hash would merge distinct forwards
    that happen to share a sender, second and subject.

    Whitespace is normalized first because Outlook rewrapped bodies
    inconsistently between copies.
    """
    normalized_body = _WHITESPACE.sub(" ", body_raw).strip().lower()
    normalized_subject = _WHITESPACE.sub(" ", subject).strip().lower()
    parts = (
        from_addr or "",
        date.isoformat() if date else "",
        normalized_subject,
        normalized_body,
    )
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8"))
    return digest.hexdigest()


_MSGID_IN_ANGLES = re.compile(r"<([^>]+)>")


def normalize_message_id(raw: str) -> str | None:
    """Strip angle brackets and whitespace from one Message-ID."""
    cleaned = raw.strip().strip("<>").strip()
    return cleaned or None


def extract_message_ids(value: str | None) -> tuple[str, ...]:
    """Parse a References or In-Reply-To header into ordered, unique IDs.

    References legitimately carries several IDs; In-Reply-To should carry one
    but occasionally carries more in this corpus. Angle-bracketed IDs are
    preferred, with a whitespace split as fallback for unbracketed values.
    """
    if not value:
        return ()
    candidates = _MSGID_IN_ANGLES.findall(value) or value.split()
    seen: dict[str, None] = {}
    for candidate in candidates:
        normalized = normalize_message_id(candidate)
        if normalized is not None:
            seen.setdefault(normalized, None)
    return tuple(seen)


# --------------------------------------------------------------------------
# Record
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedMessage:
    """One parsed Enron message.

    ``message_id`` is retained as provenance for the exported file, but it
    identifies the *file*, not the message: see :func:`content_fingerprint`.
    Use :attr:`dedup_key` to identify a logical message.

    ``in_reply_to`` and ``references`` drive tier-one thread reconstruction.
    Coverage is partial in this corpus -- the JavaMail export dropped these
    headers for many messages -- so a subject-plus-participants fallback is
    needed, and the measured header coverage is worth reporting.

    ``X-*`` headers are corpus-specific and retained deliberately: ``x_from``
    carries the sender's display name, which is the strongest signal available
    for clustering address aliases onto the same person, and ``x_folder``
    records which mailbox folder this copy came from.
    """

    source_path: str
    message_id: str | None
    in_reply_to: str | None
    references: tuple[str, ...]
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
        """Always the content hash; see :func:`content_fingerprint`.

        Message-ID is deliberately *not* consulted. It is unique per exported
        file rather than per message in this corpus, so using it would silently
        disable deduplication.
        """
        return self.content_hash

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

    message_id_raw = header_text(msg, "Message-ID")
    message_id = normalize_message_id(message_id_raw) if message_id_raw else None
    in_reply_to_ids = extract_message_ids(header_text(msg, "In-Reply-To"))

    from_addrs = extract_addresses(header_text(msg, "From"))
    subject = (header_text(msg, "Subject") or "").strip()
    body_raw = extract_body(msg)
    body_clean = clean_body(body_raw)
    date = parse_date(header_text(msg, "Date"))
    from_addr = from_addrs[0] if from_addrs else None

    return ParsedMessage(
        source_path=source_path,
        message_id=message_id,
        in_reply_to=in_reply_to_ids[0] if in_reply_to_ids else None,
        references=extract_message_ids(header_text(msg, "References")),
        date=date,
        from_addr=from_addr,
        to_addrs=extract_addresses(header_text(msg, "To")),
        cc_addrs=extract_addresses(header_text(msg, "Cc")),
        bcc_addrs=extract_addresses(header_text(msg, "Bcc")),
        subject=subject,
        x_from=(header_text(msg, "X-From") or "").strip(),
        x_to=(header_text(msg, "X-To") or "").strip(),
        x_folder=(header_text(msg, "X-Folder") or "").strip(),
        x_origin=(header_text(msg, "X-Origin") or "").strip(),
        body_raw=body_raw,
        body_clean=body_clean,
        content_hash=content_fingerprint(from_addr, date, subject, body_raw),
    )
