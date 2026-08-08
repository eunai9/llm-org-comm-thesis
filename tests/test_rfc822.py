"""Parser tests built from the failure modes this corpus actually exhibits.

Each fixture below is a shape observed in Enron mail: Outlook reply chains,
inline forwarded headers, missing or malformed Date values, non-UTF-8 bytes,
and multipart messages. The parser must never raise, because one bad message
must not abort a 500k-file ingest.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from thesis.data.rfc822 import (
    ParsedMessage,
    clean_body,
    content_fingerprint,
    extract_addresses,
    normalize_address,
    parse_date,
    parse_message,
    strip_quoted_text,
    strip_signature,
)

WELL_FORMED = """Message-ID: <1234567.1075855377439.JavaMail.evans@thyme>
Date: Mon, 14 May 2001 16:39:00 -0700 (PDT)
From: phillip.allen@enron.com
To: tim.belden@enron.com, john.doe@enron.com
Cc: Rick.Buy@ENRON.com
Subject: Re: Gas storage schedule
X-From: Phillip K Allen
X-To: Tim Belden <Tim Belden/Enron@EnronXGate>
X-Folder: \\Phillip_Allen_Jan2002_1\\Allen, Phillip K.\\'Sent Mail
X-Origin: Allen-P

Here is our forecast for the week. Please review before Friday.

Phillip
"""


def test_parses_core_headers() -> None:
    msg = parse_message(WELL_FORMED, "maildir/allen-p/sent/1.")
    assert msg.message_id == "1234567.1075855377439.JavaMail.evans@thyme"
    assert msg.from_addr == "phillip.allen@enron.com"
    assert msg.to_addrs == ("tim.belden@enron.com", "john.doe@enron.com")
    assert msg.cc_addrs == ("rick.buy@enron.com",)
    assert msg.subject == "Re: Gas storage schedule"
    assert msg.x_from == "Phillip K Allen"
    assert msg.x_origin == "Allen-P"
    assert msg.source_path == "maildir/allen-p/sent/1."


def test_addresses_are_lowercased_and_deduplicated() -> None:
    assert extract_addresses("A@ENRON.com, a@enron.com, B@enron.com") == (
        "a@enron.com",
        "b@enron.com",
    )


def test_normalize_address_rejects_non_addresses() -> None:
    assert normalize_address("  <Foo@Enron.COM> ") == "foo@enron.com"
    assert normalize_address("not-an-address") is None
    assert normalize_address("   ") is None


def test_date_is_converted_to_naive_utc() -> None:
    parsed = parse_date("Mon, 14 May 2001 16:39:00 -0700 (PDT)")
    assert parsed == datetime(2001, 5, 14, 23, 39, 0)
    assert parsed is not None
    assert parsed.tzinfo is None


@pytest.mark.parametrize("value", [None, "", "not a date", "Mon, 32 Foo 2001"])
def test_malformed_dates_return_none(value: str | None) -> None:
    assert parse_date(value) is None


def test_missing_date_header_does_not_raise() -> None:
    raw = "From: a@enron.com\nTo: b@enron.com\nSubject: No date\n\nBody text here.\n"
    msg = parse_message(raw, "p")
    assert msg.date is None
    assert msg.body_clean == "Body text here."


def test_missing_message_id_falls_back_to_content_hash() -> None:
    raw = "From: a@enron.com\nSubject: s\n\nBody.\n"
    msg = parse_message(raw, "p")
    assert msg.message_id is None
    assert msg.dedup_key == msg.content_hash
    assert len(msg.content_hash) == 64


def test_outlook_original_message_chain_is_removed() -> None:
    body = """My answer is yes.

-----Original Message-----
From: someone@enron.com
Sent: Monday, May 14, 2001 9:00 AM
To: me@enron.com
Subject: Question

Do you agree?
"""
    assert strip_quoted_text(body).strip() == "My answer is yes."


def test_forwarded_by_banner_is_removed() -> None:
    body = """Please handle this.

---------------------- Forwarded by Phillip K Allen/HOU/ECT on 05/14/2001 ---
Original content that is not mine.
"""
    assert strip_quoted_text(body).strip() == "Please handle this."


def test_inline_forwarded_header_block_is_removed() -> None:
    body = """See below.

From: someone@enron.com
Sent: Monday, May 14, 2001 9:00 AM
To: me@enron.com

Quoted content.
"""
    assert strip_quoted_text(body).strip() == "See below."


def test_bare_from_line_in_prose_is_not_treated_as_a_forward() -> None:
    """A lone 'From:' without a confirming header line must not truncate."""
    body = "The email came From: an external vendor and we should reply.\n"
    assert "external vendor" in strip_quoted_text(body)


def test_angle_quoted_lines_are_removed() -> None:
    body = "I agree.\n> previous line one\n> previous line two\n"
    assert strip_quoted_text(body).strip() == "I agree."


def test_on_wrote_chain_is_removed() -> None:
    body = "Sounds good.\n\nOn Mon, 14 May 2001, Tim Belden wrote:\nEarlier text.\n"
    assert strip_quoted_text(body).strip() == "Sounds good."


def test_signature_delimiter_is_removed() -> None:
    body = "Approved.\n\n--\nPhillip Allen\nEnron Corp\n"
    assert strip_signature(body).strip() == "Approved."


def test_trailing_contact_lines_are_removed() -> None:
    body = "Approved.\nPhillip Allen\n(713) 853-7041\nphillip.allen@enron.com\n"
    assert strip_signature(body).strip().startswith("Approved.")
    assert "853-7041" not in strip_signature(body)


def test_phone_number_mid_body_survives() -> None:
    """Only the tail is scanned, so contact details in prose are preserved."""
    body = "Call the desk at (713) 853-7041 before noon.\n" + "\n".join(
        f"Point {i} of the analysis." for i in range(10)
    )
    assert "853-7041" in strip_signature(body)


def test_clean_body_strips_quotes_and_signature_together() -> None:
    body = """Yes, proceed.

--
Phillip

-----Original Message-----
From: x@enron.com

Should we proceed?
"""
    assert clean_body(body) == "Yes, proceed."


def test_forward_only_message_cleans_to_empty() -> None:
    """Bare forwards become empty and are excluded downstream."""
    raw = (
        "From: a@enron.com\nSubject: FW: report\n\n"
        "-----Original Message-----\nFrom: b@enron.com\n\nThe actual content.\n"
    )
    assert parse_message(raw, "p").body_clean == ""


def test_multipart_message_uses_text_plain_part() -> None:
    raw = """From: a@enron.com
Subject: Multipart
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="BOUND"

--BOUND
Content-Type: text/plain; charset="us-ascii"

The plain text version.

--BOUND
Content-Type: text/html; charset="us-ascii"

<html><body>The HTML version.</body></html>

--BOUND--
"""
    msg = parse_message(raw, "p")
    assert "plain text version" in msg.body_clean
    assert "HTML version" not in msg.body_clean


def test_non_utf8_bytes_do_not_raise() -> None:
    """cp1252 smart quotes are common in this corpus and must decode."""
    raw = (
        "From: a@enron.com\nSubject: Encoding\n"
        'Content-Type: text/plain; charset="iso-8859-1"\n\n'
        "Caf\xe9 meeting at 3pm.\n"
    )
    msg = parse_message(raw, "p")
    assert "meeting at 3pm" in msg.body_clean


def test_crlf_line_endings_are_handled() -> None:
    raw = "From: a@enron.com\r\nSubject: CRLF\r\n\r\nLine one.\r\nLine two.\r\n"
    msg = parse_message(raw, "p")
    assert "Line one." in msg.body_clean
    assert "Line two." in msg.body_clean


def test_empty_input_does_not_raise() -> None:
    msg = parse_message("", "p")
    assert isinstance(msg, ParsedMessage)
    assert msg.from_addr is None
    assert msg.body_clean == ""


def test_recipient_count_spans_to_cc_and_bcc() -> None:
    raw = (
        "From: a@enron.com\nTo: b@enron.com, c@enron.com\n"
        "Cc: d@enron.com\nBcc: e@enron.com\nSubject: s\n\nBody.\n"
    )
    assert parse_message(raw, "p").n_recipients == 4


def test_fingerprint_ignores_whitespace_rewrapping() -> None:
    """Outlook rewrapped bodies between folder copies; dedup must survive it."""
    date = datetime(2001, 5, 14, 23, 39)
    a = content_fingerprint("a@enron.com", date, "Subject", "one two three")
    b = content_fingerprint("a@enron.com", date, "Subject", "one  two\n three\n")
    assert a == b


def test_fingerprint_separates_different_content() -> None:
    date = datetime(2001, 5, 14, 23, 39)
    a = content_fingerprint("a@enron.com", date, "Subject", "approved")
    b = content_fingerprint("a@enron.com", date, "Subject", "denied")
    assert a != b


def test_parsed_message_is_immutable() -> None:
    msg = parse_message(WELL_FORMED, "p")
    with pytest.raises((AttributeError, TypeError)):
        msg.subject = "changed"  # type: ignore[misc]
