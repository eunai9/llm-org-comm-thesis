"""A costless, keyless client for exercising the pipeline end to end.

Its purpose is to answer "does the machinery work?" -- does the grid expand,
does the cache hit, does the ledger balance, does the Parquet come out with the
right schema -- without an API key and without spending anything.

**Its output is not data.** The replies are assembled from templates, not
generated, so nothing produced through this client may appear in the thesis as
a result. Two safeguards make that hard to forget rather than merely
documented:

- Every response is stamped with a model id beginning ``stub-``, which flows
  into the results Parquet and the run manifest, so any downstream table built
  from stub output is visibly stub output.
- :func:`is_stub_model` gives the analysis layer a single check to refuse such
  rows, rather than each script inventing its own.

The replies do vary by persona rank, direction, and stakes. Not for realism --
they would not fool anyone -- but so that a pipeline bug which collapses every
cell onto one response, or crosses two cells' inputs, shows up as identical
output where it should differ. A stub that returned one constant string would
hide exactly the class of bug this is meant to surface.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from thesis.llm.base import (
    Capabilities,
    CompletionRequest,
    CompletionResponse,
    Provider,
    Usage,
)

STUB_MODEL_PREFIX = "stub-"


def is_stub_model(model: str) -> bool:
    """Whether a model id came from this client.

    Analysis code should call this and refuse the row, so stub output cannot
    reach a results table by being forgotten about.
    """
    return model.startswith(STUB_MODEL_PREFIX)


_BODIES: dict[str, str] = {
    "accept": "Fine by me - go ahead. Numbers are attached.",
    "decline": "I don't think we can do that on this timeline. Let's keep the original date.",
    "defer": "Let me check the figures before I commit to anything. Back to you tomorrow.",
    "escalate": "This one is above my line - I'll put it in front of the desk head today.",
    "none": "Noted, thanks.",
}


def _deterministic_choice(seed_text: str, options: Sequence[str]) -> str:
    """Pick an option by hashing the prompt.

    Deterministic so that the same request always yields the same stub reply:
    a cache hit and a fresh stub call must agree, or cache tests would pass or
    fail depending on which path ran first.
    """
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return options[digest[0] % len(options)]


class StubClient:
    """Satisfies the client protocol; never touches the network."""

    provider: Provider = "anthropic"

    def __init__(self, *, model_label: str = "stub-sim") -> None:
        self.model_label = (
            model_label if model_label.startswith(STUB_MODEL_PREFIX) else f"stub-{model_label}"
        )
        self.n_completions = 0

    def capabilities(self, model: str) -> Capabilities:
        # Mirrors claude-opus-5's real constraints, so code paths that branch
        # on capabilities behave here the way they will in a paid run.
        return Capabilities(
            supports_sampling_params=False,
            min_cacheable_prompt_tokens=512,
            thinking_on_by_default=True,
        )

    def count_tokens(self, request: CompletionRequest) -> int:
        """Character-based approximation. Adequate only for shape-checking."""
        text = (request.system or "") + "".join(m.content for m in request.messages)
        return max(1, len(text) // 4)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        prompt = (request.system or "") + "".join(m.content for m in request.messages)
        # The variant is folded into the seed so replicates differ, matching
        # the real client's behaviour of returning independent draws.
        seed = f"{prompt}|{request.variant}"
        decision = _deterministic_choice(seed, list(_BODIES))

        payload = {
            "subject": "Re: the item below",
            "body": _BODIES[decision],
            "decision": decision,
            "confidence": _deterministic_choice(seed + "c", ["low", "medium", "high"]),
            "reasoning_brief": "Stub response - generated offline, not model output.",
        }
        text = json.dumps(payload)
        self.n_completions += 1

        return CompletionResponse(
            text=text,
            usage=Usage(input_tokens=self.count_tokens(request), output_tokens=len(text) // 4),
            model=self.model_label,
            stop_reason="end_turn",
            parsed=payload,
        )

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        return "stub-batch"

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        return {}
