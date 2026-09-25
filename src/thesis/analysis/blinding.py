"""Strip identity markers from a reply before it reaches the role-inference
judge: a greeting naming the recipient, a sign-off block naming the writer,
and a job title mentioned anywhere in the body.

Generated replies rarely carry any of these -- the median reply is one
sentence and has no room for a sign-off. Real Enron email is the reason this
exists: a signature block naming a Vice President is exactly the kind of
leak that would let a judge guess direction from provenance rather than from
how the message is actually written, which is the whole point of the blind
task in :mod:`thesis.analysis.role_inference`.

The title lexicon is the project's own 36-entry roster
(:func:`thesis.data.roles.load_title_rank_table`), not a hand-built list --
these are the titles that can actually appear in this corpus's signatures.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from thesis.data.roles import load_title_rank_table

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|dear)\s+[A-Z][a-zA-Z'-]*[,:]?\s*$", re.IGNORECASE | re.MULTILINE
)
_SIGNOFF_START_RE = re.compile(
    r"^\s*(best|regards|sincerely|thanks|thank you|cheers)[,.]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _default_titles() -> tuple[str, ...]:
    return tuple(load_title_rank_table().keys())


def _strip_signoff(text: str) -> str:
    """Everything from the first sign-off line onward, dropped."""
    match = _SIGNOFF_START_RE.search(text)
    return text[: match.start()].rstrip() if match else text


def _strip_greeting(text: str) -> str:
    return _GREETING_RE.sub("", text)


def _strip_titles(text: str, titles: Iterable[str]) -> str:
    """Replace each title string, longest first so "Director" doesn't eat
    part of "Managing Director" before the longer match gets a chance."""
    result = text
    for title in sorted(titles, key=len, reverse=True):
        pattern = re.compile(re.escape(title), re.IGNORECASE)
        result = pattern.sub("", result)
    return result


def strip_identity(text: str, *, titles: Iterable[str] | None = None) -> str:
    """Remove greeting, sign-off, and job-title mentions from a reply body.

    ``titles`` defaults to the project's real 36-title roster. Order of
    operations matters: the sign-off is stripped whole (removing a trailing
    name along with any title next to it) before the mid-body title pass
    runs on what's left, so a title inside a stripped sign-off is never
    double-processed and a title still standing in the body (e.g. someone
    referring to their own role mid-sentence) is still caught.
    """
    resolved_titles = tuple(titles) if titles is not None else _default_titles()
    text = _strip_signoff(text)
    text = _strip_greeting(text)
    text = _strip_titles(text, resolved_titles)
    return re.sub(r"[ \t]+", " ", text).strip()
