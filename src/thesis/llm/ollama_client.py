"""A local-model client, for real generated text at no cost.

Ollama runs an open-weights model on this machine. That makes it the middle
rung between the stub client (free, but templated) and a frontier API (real,
but paid): the text here is genuinely *generated*, so prompts can be judged on
whether they elicit sensible email before any money is spent on the real run.

**Local output is still not thesis data**, and the reason is not just quality.
The research questions name specific, pinned, citable models; a reply from a 3B
model running on a laptop cannot stand in for one of those in a results table.
So local responses are marked the same way stub responses are -- a ``local/``
prefix on the model id, recorded in the results file -- and priced at zero,
because nothing was billed.

Two practical notes about running models locally:

- Memory is the binding constraint, not speed. A model needs roughly its
  parameter count in gigabytes at 8-bit, or about half that at 4-bit, plus
  room for context. This project's WSL instance is capped at 8GB, so a 3B
  model is comfortable and anything past about 8B is not.
- Ollama exposes an OpenAI-compatible endpoint, but its **native** API is used
  here because the native ``format`` field accepts a full JSON schema, which
  is what constrains the response to the shape the analysis expects. The
  compatibility layer's support for structured output is less consistent
  across versions.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx

from thesis.llm.base import (
    Capabilities,
    CompletionRequest,
    CompletionResponse,
    Provider,
    Usage,
)
from thesis.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_HOST = "http://127.0.0.1:11434"
LOCAL_MODEL_PREFIX = "local/"

# Generous: a small model on CPU is slow, and a timeout that fires mid-run
# looks like a bug rather than the hardware simply being modest.
DEFAULT_TIMEOUT_SECONDS = 300.0


def is_local_model(model: str) -> bool:
    """Whether a model id came from a locally-run model.

    Analysis code checks this, alongside the stub check, to refuse rows that
    did not come from a pinned provider model.
    """
    return model.startswith(LOCAL_MODEL_PREFIX)


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaClient:
    """Satisfies the client protocol against a locally-running Ollama server."""

    provider: Provider = "ollama"

    def __init__(
        self,
        model: str = "llama3.2:3b",
        *,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self._client = httpx.Client(base_url=self.host, timeout=timeout)

    # ------------------------------------------------------------- lifecycle

    def is_available(self) -> bool:
        """Whether the server is reachable. Never raises."""
        try:
            self._client.get("/api/tags").raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    def installed_models(self) -> list[str]:
        """Model names already pulled onto this machine."""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            msg = f"could not reach Ollama at {self.host}: {exc}"
            raise OllamaUnavailableError(msg) from exc
        payload: dict[str, Any] = response.json()
        return sorted(entry["name"] for entry in payload.get("models", []))

    # ---------------------------------------------------------- capabilities

    def capabilities(self, model: str) -> Capabilities:
        """Local models accept sampling parameters and never cache prompts.

        ``min_cacheable_prompt_tokens`` is deliberately enormous rather than
        zero: there is no prompt cache here, and reporting a small threshold
        would let calling code believe caching is available and silently
        expect savings that cannot occur.
        """
        return Capabilities(
            supports_sampling_params=True,
            min_cacheable_prompt_tokens=10**9,
            thinking_on_by_default=False,
            supports_batch=False,
        )

    def count_tokens(self, request: CompletionRequest) -> int:
        """Approximate. Ollama exposes no pre-flight counting endpoint.

        Only ever used for projections, and a local run costs nothing, so the
        approximation cannot mislead a spending decision.
        """
        text = (request.system or "") + "".join(m.content for m in request.messages)
        return max(1, len(text) // 4)

    # -------------------------------------------------------------- requests

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": request.max_tokens},
        }
        if request.output_schema is not None:
            # The native API takes a JSON schema directly, which is what keeps
            # a small model's output in the shape the analysis expects.
            payload["format"] = request.output_schema
        return payload

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            response = self._client.post("/api/chat", json=self._payload(request))
            response.raise_for_status()
        except httpx.HTTPError as exc:
            msg = (
                f"Ollama request failed against {self.host}: {exc}. Is the "
                f"server running, and has {self.model!r} been pulled?"
            )
            raise OllamaUnavailableError(msg) from exc

        body: dict[str, Any] = response.json()
        text = body.get("message", {}).get("content", "")

        parsed: dict[str, Any] | None = None
        if text:
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                # Small models drift from a schema more often than frontier
                # ones. Recorded rather than raised so one malformed reply does
                # not abort a whole prototype run.
                log.warning("local model returned non-JSON output (%d chars)", len(text))
            else:
                parsed = candidate if isinstance(candidate, dict) else None

        usage = Usage(
            input_tokens=int(body.get("prompt_eval_count", 0) or 0),
            output_tokens=int(body.get("eval_count", 0) or 0),
        )
        return CompletionResponse(
            text=text,
            usage=usage,
            model=f"{LOCAL_MODEL_PREFIX}{self.model}",
            stop_reason=body.get("done_reason"),
            parsed=parsed,
        )

    # --------------------------------------------------------------- batches

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        """Not supported: there is no queue to submit to, and no discount.

        Failing loudly beats silently looping, which would look like batching
        while providing none of its properties.
        """
        msg = "Ollama has no batch API; run cells individually."
        raise NotImplementedError(msg)

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        msg = "Ollama has no batch API."
        raise NotImplementedError(msg)
