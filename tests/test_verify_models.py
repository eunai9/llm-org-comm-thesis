"""Model-id verification tests.

The check runs before money is spent, so its failure modes matter: it must not
crash when a provider is unreachable, and it must not report an unverifiable
id as verified.
"""

from __future__ import annotations

from thesis.llm.verify_models import ProviderModels, configured_models, suggest


def test_unreachable_provider_is_not_reachable() -> None:
    listing = ProviderModels("openai", error="no key")
    assert listing.reachable is False
    assert listing.available == ()


def test_reachable_provider_reports_ids() -> None:
    listing = ProviderModels("openai", available=("gpt-a", "gpt-b"))
    assert listing.reachable is True


def test_suggest_matches_on_model_family() -> None:
    """A retired gpt-* is replaced by a current gpt-*, not by whatever string
    happens to be textually closest."""
    # Sorted, because both listing functions always return sorted ids and the
    # suggestion order should reflect what the caller will actually see.
    available = tuple(sorted(("claude-opus-5", "gpt-5-mini", "gpt-5", "o4-mini")))
    assert suggest("gpt-4o", available) == ["gpt-5", "gpt-5-mini"]


def test_suggest_returns_empty_when_family_absent() -> None:
    assert suggest("gpt-4o", ("claude-opus-5",)) == []


def test_suggest_respects_limit() -> None:
    available = tuple(f"gpt-{i}" for i in range(50))
    assert len(suggest("gpt-4o", available, limit=3)) == 3


def test_configured_models_reads_every_role() -> None:
    """Simulator, judge, and labeller ids must all be checked."""
    pinned = configured_models()
    assert "anthropic" in pinned
    assert any(m.startswith("claude") for m in pinned["anthropic"])
