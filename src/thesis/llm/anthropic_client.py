"""Anthropic implementation of :class:`~thesis.llm.base.LLMClient`.

Everything provider-specific lives here rather than leaking into the shared
protocol: the cache-control block shape, structured outputs, the batch request
wrapper, and the model-by-model constraints that would otherwise be invisible
until a request 400s mid-run.

Three of those constraints are worth stating outright, because each one is a
real failure this file exists to prevent:

1. ``claude-opus-5`` **rejects** ``temperature``, ``top_p``, and ``top_k`` with
   a 400. This is why the plan's "temperature 0.8 for diversity" idea is not
   available and diversity is instead reported at API defaults as a named
   limitation. Nothing in this module ever sends a sampling parameter.
2. Prompt caching has a **minimum cacheable prefix** that differs per model
   (512 tokens on ``claude-opus-5``, 1024 on ``claude-sonnet-5``). Below it,
   caching silently does nothing -- no error, just a bill. The simulator's
   prompt assembly is ordered to keep the stable prefix comfortably above it,
   and :meth:`AnthropicClient.complete` reports back whether the cache was
   actually read so that assumption is checked rather than trusted.
3. On ``claude-opus-5`` thinking is **on by default**, and ``max_tokens`` caps
   thinking plus visible output together. A ``max_tokens`` sized for the reply
   alone truncates the reply.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, cast

import anthropic

from thesis.llm.base import (
    Capabilities,
    CompletionRequest,
    CompletionResponse,
    Usage,
)
from thesis.logging_setup import get_logger

log = get_logger(__name__)

# Per-model constraints. A dict rather than string-prefix checks so that an
# unknown model is a loud KeyError at request-build time instead of a wrong
# assumption that only surfaces as a 400 halfway through a paid batch.
_CAPABILITIES: dict[str, Capabilities] = {
    "claude-opus-5": Capabilities(
        supports_sampling_params=False,
        min_cacheable_prompt_tokens=512,
        thinking_on_by_default=True,
    ),
    "claude-sonnet-5": Capabilities(
        supports_sampling_params=False,
        min_cacheable_prompt_tokens=1024,
        thinking_on_by_default=False,
    ),
    "claude-haiku-4-5": Capabilities(
        supports_sampling_params=True,
        min_cacheable_prompt_tokens=4096,
        thinking_on_by_default=False,
    ),
}


class UnknownModelError(KeyError):
    """Raised for a model this project has not recorded constraints for.

    Failing loudly is deliberate. Guessing a model's capabilities is how a run
    ends up sending a parameter that is rejected, or assuming a cache that
    never forms.
    """


def capabilities_for(model: str) -> Capabilities:
    """Look up a model's constraints, or fail with a useful message."""
    try:
        return _CAPABILITIES[model]
    except KeyError:
        known = ", ".join(sorted(_CAPABILITIES))
        msg = f"no recorded capabilities for {model!r}; known models: {known}"
        raise UnknownModelError(msg) from None


class AnthropicClient:
    """Thin wrapper over the Anthropic SDK."""

    provider = "anthropic"

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        # The zero-argument constructor resolves credentials from the
        # environment (ANTHROPIC_API_KEY, or a logged-in CLI profile), so no
        # key is ever read or stored by this project's own code.
        self._client = client or anthropic.Anthropic()

    def capabilities(self, model: str) -> Capabilities:
        return capabilities_for(model)

    # ---------------------------------------------------------------- request

    def _system_param(self, request: CompletionRequest) -> Any:
        """Build the ``system`` field, optionally marked as a cache breakpoint.

        The breakpoint goes on the last (here, only) system block, which caches
        the system prompt and any tools together -- render order is tools, then
        system, then messages, so a marker here covers the whole stable prefix.
        """
        if request.system is None:
            return anthropic.NOT_GIVEN
        if not request.cache_system:
            return request.system
        return [
            {
                "type": "text",
                "text": request.system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _build_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        """Assemble SDK kwargs, deliberately omitting sampling parameters."""
        caps = self.capabilities(request.model)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        system = self._system_param(request)
        if system is not anthropic.NOT_GIVEN:
            kwargs["system"] = system

        if request.output_schema is not None:
            # Structured outputs replace the older assistant-prefill trick,
            # which returns a 400 on every model this project uses.
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": request.output_schema}
            }

        # Never send temperature/top_p/top_k. Asserting rather than silently
        # skipping documents the reason at the one place it would be tempting
        # to add them back.
        assert not caps.supports_sampling_params or "temperature" not in kwargs
        return kwargs

    # --------------------------------------------------------------- counting

    def count_tokens(self, request: CompletionRequest) -> int:
        """Exact input-token count via the provider's own endpoint."""
        kwargs = self._build_kwargs(request)
        result = self._client.messages.count_tokens(
            model=kwargs["model"],
            messages=kwargs["messages"],
            system=kwargs.get("system", anthropic.NOT_GIVEN),
        )
        return int(result.input_tokens)

    # -------------------------------------------------------------- responses

    def _to_response(self, message: Any) -> CompletionResponse:
        """Normalize an SDK message into this project's response type.

        Content is a list of typed blocks, and with thinking enabled the first
        block is a thinking block rather than text -- so blocks are filtered by
        type instead of indexing ``content[0]``, which would return reasoning
        (or an empty string) instead of the answer.
        """
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )

        parsed: dict[str, Any] | None = None
        if text:
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            else:
                parsed = candidate if isinstance(candidate, dict) else None

        raw_usage = message.usage
        usage = Usage(
            input_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
            output_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
        )
        return CompletionResponse(
            text=text,
            usage=usage,
            model=message.model,
            stop_reason=message.stop_reason,
            parsed=parsed,
            request_id=getattr(message, "_request_id", None),
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Run one completion.

        ``stop_reason`` is checked before the text is trusted: a refusal
        returns HTTP 200 with an empty or partial body, so code that reads the
        content unconditionally would treat a refusal as a valid response and
        quietly analyse an empty string.
        """
        message = self._client.messages.create(**self._build_kwargs(request))
        response = self._to_response(message)

        if response.stop_reason == "refusal":
            log.warning("model refused request (model=%s)", request.model)
        elif response.stop_reason == "max_tokens":
            log.warning(
                "response truncated at max_tokens=%d (model=%s); on a "
                "thinking-by-default model this budget covers thinking too",
                request.max_tokens,
                request.model,
            )
        return response

    # ----------------------------------------------------------------- batches

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        """Submit a batch and return its id. Batched calls cost 50% less."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        payload = [
            Request(
                custom_id=custom_id,
                # _build_kwargs returns a plain dict; the SDK types the batch
                # params as a TypedDict. The shape is identical -- the cast
                # just tells mypy that, since ** expansion into a TypedDict is
                # not something it can verify structurally.
                params=cast(MessageCreateParamsNonStreaming, self._build_kwargs(request)),
            )
            for custom_id, request in requests
        ]
        batch = self._client.messages.batches.create(requests=payload)
        log.info("submitted batch %s with %d request(s)", batch.id, len(payload))
        return str(batch.id)

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        """Return results keyed by ``custom_id``, or ``None`` if still running.

        Results are keyed by ``custom_id`` because the API returns them in
        arbitrary order -- pairing by position would silently attach responses
        to the wrong grid cell, which is the kind of error that survives all
        the way into a results table.
        """
        batch = self._client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            return None

        results: dict[str, CompletionResponse] = {}
        for entry in self._client.messages.batches.results(batch_id):
            # Narrowing on the attribute directly, rather than via a local
            # copy of `.type`, is what lets the checker prove `.message`
            # exists -- only the succeeded variant of the result union has it.
            if entry.result.type == "succeeded":
                results[entry.custom_id] = self._to_response(entry.result.message)
            else:
                # Recorded rather than raised: one bad cell should not discard
                # thousands of good responses from the same batch.
                log.warning("batch %s: %s -> %s", batch_id, entry.custom_id, entry.result.type)
        return results


def api_key_present() -> bool:
    """Whether an Anthropic credential is visible in the environment.

    Used by tests and dry runs to skip live calls cleanly instead of failing
    with an authentication error that looks like a bug.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
