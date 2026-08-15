"""Check that every model id in the config actually exists on the account.

A pinned model id is a claim about the outside world, and the outside world
moves: providers retire ids, rename them, and gate some behind account
verification. A stale pin does not fail quietly at import time -- it fails at
the first paid call, which on a batch run means after the submission has
already been accepted.

This module asks each provider what *this account* can actually see, which is
the only authoritative answer. The published docs describe what exists in
general; the account listing describes what is callable here, and those differ
whenever a model needs org verification or a higher tier.

Run with ``python -m thesis.llm.verify_models``. It makes no paid calls -- the
model listing endpoints are free -- so it is safe to run as often as wanted,
and worth running before any batch submission.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from thesis.config import load_config
from thesis.logging_setup import configure_logging, get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProviderModels:
    """What one provider reports as available, or why it could not be asked."""

    provider: str
    available: tuple[str, ...] = ()
    error: str | None = None

    @property
    def reachable(self) -> bool:
        return self.error is None


def list_anthropic_models() -> ProviderModels:
    """Model ids visible to this Anthropic account."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return ProviderModels("anthropic", error="no ANTHROPIC_API_KEY in environment")
    try:
        import anthropic

        client = anthropic.Anthropic()
        return ProviderModels(
            "anthropic",
            available=tuple(sorted(m.id for m in client.models.list(limit=1000))),
        )
    except Exception as exc:  # Any failure is reported, never crashes the check.
        return ProviderModels("anthropic", error=f"{type(exc).__name__}: {exc}")


def list_openai_models() -> ProviderModels:
    """Model ids visible to this OpenAI account."""
    if not os.environ.get("OPENAI_API_KEY"):
        return ProviderModels("openai", error="no OPENAI_API_KEY in environment")
    try:
        import openai

        client = openai.OpenAI()
        return ProviderModels(
            "openai",
            available=tuple(sorted(m.id for m in client.models.list())),
        )
    except Exception as exc:  # Any failure is reported, never crashes the check.
        return ProviderModels("openai", error=f"{type(exc).__name__}: {exc}")


def configured_models() -> dict[str, set[str]]:
    """Every model id the config pins, grouped by provider."""
    config = load_config()
    pinned: dict[str, set[str]] = {}
    specs = [*config.models.simulator, *config.models.judge, config.models.labeller]
    for spec in specs:
        pinned.setdefault(spec.provider, set()).add(spec.model_id)
    return pinned


def suggest(missing: str, available: tuple[str, ...], limit: int = 12) -> list[str]:
    """Plausible replacements for a missing id.

    Matches on the leading alphabetic token (``gpt``, ``claude``) rather than
    on edit distance: a retired ``gpt-4o`` is far more likely to be replaced by
    a current ``gpt-*`` than by whatever id happens to be textually closest.
    """
    family = missing.split("-")[0].lower()
    return [m for m in available if m.lower().startswith(family)][:limit]


def check() -> tuple[bool, list[str]]:
    """Compare configured ids against what each account reports.

    Returns ``(ok, lines)``. ``ok`` is False only when a configured id is
    confirmed absent -- an unreachable provider is reported but does not count
    as a failure, since "could not check" is not the same as "wrong".
    """
    listings = {"anthropic": list_anthropic_models(), "openai": list_openai_models()}
    pinned = configured_models()

    ok = True
    lines: list[str] = []
    for provider, ids in sorted(pinned.items()):
        listing = listings[provider]
        lines.append(f"[{provider}]")

        if not listing.reachable:
            lines.append(f"  ? could not verify: {listing.error}")
            lines.extend(f"  ? {model_id} (unchecked)" for model_id in sorted(ids))
            lines.append("")
            continue

        for model_id in sorted(ids):
            if model_id in listing.available:
                lines.append(f"  OK      {model_id}")
            else:
                ok = False
                lines.append(f"  MISSING {model_id}")
                for candidate in suggest(model_id, listing.available):
                    lines.append(f"            candidate: {candidate}")
        lines.append("")
    return ok, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print every model id each account can see, then exit.",
    )
    args = parser.parse_args()
    configure_logging()

    if args.list:
        for listing in (list_anthropic_models(), list_openai_models()):
            print(f"[{listing.provider}]")
            if not listing.reachable:
                print(f"  unavailable: {listing.error}")
            else:
                for model_id in listing.available:
                    print(f"  {model_id}")
            print()
        return

    ok, lines = check()
    for line in lines:
        print(line)

    if ok:
        print("All configured model ids were found (or could not be checked).")
    else:
        print(
            "At least one configured model id does not exist on the account.\n"
            "Update configs/models.yaml, and record the date you confirmed it."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
