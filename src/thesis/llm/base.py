"""The provider-agnostic surface every LLM call in this project goes through.

Deliberately thin. The plan's instruction was to abstract over the *shape* of a
request and response -- prompt in, text plus token counts out -- and to keep
provider-specific features (extended thinking, cache-control syntax,
structured-output syntax) inside each client rather than inventing a
lowest-common-denominator wrapper for them. An over-general provider
abstraction is a classic time sink with no thesis value: the research questions
are about what the models *say*, not about how elegantly the SDKs are wrapped.

What does need to be uniform is the part the analysis depends on:

- :class:`Usage` -- token counts, normalized across providers, because the cost
  ledger and the "how much did this thesis cost to run" number in the write-up
  have to add up across two SDKs that name these fields differently.
- :class:`Capabilities` -- the per-model facts that change how a request must be
  built. These are data rather than scattered ``if model.startswith(...)``
  checks so that a wrong assumption fails a test instead of silently producing
  a 400 in the middle of a paid batch run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# Includes local execution alongside the two paid families. config.Provider
# stays restricted to the paid ones on purpose: a local model is a development
# tool, not something the experimental design may be configured to run on.
Provider = Literal["anthropic", "openai", "ollama"]
Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """One conversational turn. Content is plain text -- this project never
    sends images or documents, so the multimodal content-block forms of either
    SDK would be unused complexity."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for a single call, normalized across providers.

    Cache fields are Anthropic-specific in origin but kept on the shared type
    because the cost calculation needs them and a zero on OpenAI is honest
    rather than missing: OpenAI's automatic caching is not something this
    project requests or controls, so it is reported as uncached.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        """Every input token, cached or not.

        ``input_tokens`` from the API is the *uncached remainder* only, so
        summing the three is the only way to recover the true prompt size --
        an easy and expensive thing to get wrong when reporting corpus-level
        token counts in the thesis.
        """
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    def __add__(self, other: Usage) -> Usage:
        """Accumulate usage across calls, so a run total is a plain ``sum()``."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
        )

    def __radd__(self, other: int) -> Usage:
        """Support ``sum(usages)``, whose start value is the integer 0."""
        if other == 0:
            return self
        raise TypeError("Usage can only be summed with other Usage values")


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Per-model facts that change how a request must be constructed.

    Each field exists because getting it wrong is a hard failure or a silent
    one, not because it is interesting in itself:

    ``supports_sampling_params``
        ``claude-opus-5`` **rejects** ``temperature``/``top_p``/``top_k`` with a
        400. The plan anticipated this and named the consequence as a stated
        limitation: temperature is dropped as an experimental factor and
        diversity is reported at API defaults. Encoding it here means the
        simulator cannot accidentally send a parameter that would fail an
        entire batch.

    ``min_cacheable_prompt_tokens``
        A prompt shorter than this silently does not cache -- no error, just no
        saving, and ``cache_read_input_tokens`` stays 0. The value differs by
        model (512 on ``claude-opus-5``, 1024 on ``claude-sonnet-5``), which is
        exactly the kind of detail that turns into an unexplained cost overrun
        if assumed uniform.

    ``thinking_on_by_default``
        On ``claude-opus-5`` a request that omits ``thinking`` still thinks, and
        ``max_tokens`` caps thinking *plus* visible output together. A
        max_tokens tuned as if it bounded the reply alone can truncate the
        reply mid-sentence.
    """

    supports_sampling_params: bool
    min_cacheable_prompt_tokens: int
    thinking_on_by_default: bool
    supports_batch: bool = True


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """One completion, described independently of any SDK.

    ``system`` is separate from ``messages`` because both providers treat it as
    a distinct field, and because it is the part that stays byte-identical
    across a grid cell -- which is what makes prompt caching work at all.
    """

    model: str
    messages: Sequence[Message]
    max_tokens: int
    system: str | None = None
    output_schema: dict[str, Any] | None = None
    cache_system: bool = False

    # Which independent draw this is for an otherwise identical prompt.
    #
    # Replicates exist to measure how much the model's output *varies* when
    # asked the same thing repeatedly. Their prompts are byte-identical by
    # construction, so without something to tell them apart the cache would
    # serve draw 1 for every replicate: every replicate would be the same
    # text, measured variance would be exactly zero, and the diversity
    # analysis would silently be reporting a property of the cache rather
    # than of the model. This field participates in the cache key so each
    # draw is its own entry.
    variant: int = 0

    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    """The result of one completion.

    ``parsed`` is populated only when the request carried an ``output_schema``;
    it holds the decoded JSON object. ``text`` is always the raw response text,
    kept even when parsing succeeded so that a failure to parse downstream can
    be diagnosed against exactly what the model returned.
    """

    text: str
    usage: Usage
    model: str
    stop_reason: str | None = None
    parsed: dict[str, Any] | None = None
    request_id: str | None = None
    from_cache: bool = False


@runtime_checkable
class BatchClient(Protocol):
    """The subset of a client the batch pass actually uses.

    Separated from :class:`LLMClient` so the batch code states its real
    dependency. Requiring the full client there would force test doubles to
    stub methods that are never called, and would type-admit clients that
    cannot batch at all -- the local Ollama client raises on submission, and
    the type system should be able to say so.
    """

    provider: Provider

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        """Submit ``(custom_id, request)`` pairs; return the provider batch id."""
        ...

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        """Return results keyed by ``custom_id``, or ``None`` if still running."""
        ...


@runtime_checkable
class LLMClient(Protocol):
    """What every provider client must offer.

    Batch submission is part of the protocol rather than an optional extra
    because the plan's cost model depends on it: batching halves the price, and
    at ~13,300 planned calls that is the difference between a comfortable
    budget and an uncomfortable one.
    """

    provider: Provider

    def capabilities(self, model: str) -> Capabilities:
        """Per-model constraints, so callers can build a legal request."""
        ...

    def count_tokens(self, request: CompletionRequest) -> int:
        """Exact input-token count for cost projection.

        Must use the provider's own counting endpoint. Approximating Claude
        token counts with ``tiktoken`` undercounts by 15-20% on prose and
        considerably more on code, which would make every pre-flight cost
        guard in this project optimistic in the wrong direction.
        """
        ...

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Run one completion synchronously."""
        ...

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        """Submit ``(custom_id, request)`` pairs; return the provider batch id."""
        ...

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        """Return results keyed by ``custom_id``, or ``None`` if still running.

        Results come back in arbitrary order on both providers, so the return
        type is a mapping rather than a list -- keying by position is a real
        and silent way to mismatch a response to the wrong grid cell.
        """
        ...
