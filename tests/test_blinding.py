from __future__ import annotations

from thesis.analysis.blinding import strip_identity

TEST_TITLES = ("Vice President", "Managing Director", "Director")


def test_removes_a_signoff_block() -> None:
    text = "Sure, I'll get that over today.\n\nBest,\nJohn Smith"
    result = strip_identity(text, titles=TEST_TITLES)
    assert "John Smith" not in result
    assert "Sure, I'll get that over today." in result


def test_removes_a_signoff_with_a_title() -> None:
    text = "Approved.\n\nRegards,\nSarah Lee, Vice President of Trading"
    result = strip_identity(text, titles=TEST_TITLES)
    assert "Sarah Lee" not in result
    assert "Vice President" not in result


def test_removes_a_greeting_naming_the_recipient() -> None:
    text = "Hi Sarah,\n\nCan you send the figures over today?"
    result = strip_identity(text, titles=TEST_TITLES)
    assert "Sarah" not in result
    assert "Can you send the figures over today?" in result


def test_removes_a_title_mentioned_mid_body() -> None:
    text = "As the Managing Director I've decided to approve this."
    result = strip_identity(text, titles=TEST_TITLES)
    assert "Managing Director" not in result
    assert "decided to approve this" in result


def test_does_not_touch_ordinary_text_with_no_markers() -> None:
    text = "I need the trading team to confirm they can accommodate the new date."
    assert strip_identity(text, titles=TEST_TITLES) == text


def test_uses_the_real_title_roster_by_default() -> None:
    """No titles argument: falls back to the project's own 36-title roster,
    so a real Enron signature ("VP Trading", "Mng Dir Trading", ...) is
    caught without the caller enumerating titles by hand."""
    text = "Sounds good.\n\nBest,\nTom - VP Trading"
    result = strip_identity(text)
    assert "VP Trading" not in result
