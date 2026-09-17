"""One client body for the free, OpenAI-compatible APIs this project uses.

NVIDIA's catalog and Groq both speak the OpenAI chat-completions format, so
the request shape, the rate spacing, the retry rule and the response mapping
are identical. Only the address, the key, the marker prefix and a couple of
per-service extras differ, and those live in the subclasses.

Neither service is billed on the tiers this project uses, so both mark their
responses with a prefix (``nim/``, ``groq/``). The cost ledger prices a
marked reply at zero instead of failing on a missing price.

A model id may end in ``@low``, ``@medium`` or ``@high``, for example
``openai/gpt-oss-120b@low``. The client sends the plain id and asks for that
reasoning effort. The full id stays on every response and in the cache key,
so replies made with different efforts are never mixed up.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

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

# A normal reply takes under 15 seconds. A free tier sometimes holds a request
# without answering, so a long timeout only delays the retry.
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 2.0
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

REASONING_EFFORTS = frozenset({"low", "medium", "high"})


class FreeTierUnavailableError(RuntimeError):
    """Raised when a free-tier API is unreachable or refuses a request."""


def split_reasoning_effort(model: str) -> tuple[str, str | None]:
    """Split ``name@effort`` into the model name the API expects and the effort."""
    name, separator, effort = model.rpartition("@")
    if not separator:
        return model, None
    if effort not in REASONING_EFFORTS:
        msg = (
            f"unknown reasoning effort {effort!r} in {model!r}; "
            f"use one of {sorted(REASONING_EFFORTS)}"
        )
        raise ValueError(msg)
    return name, effort


class OpenAICompatibleClient:
    """Satisfies the client protocol against a free, OpenAI-compatible API.

    The model is taken from each request, so one client can serve several
    models of the same service in one run.
    """

    # Not a ClassVar: the client protocol expects an instance variable, and a
    # class variable does not satisfy it.
    provider: Provider
    base_url: ClassVar[str]
    api_key_env: ClassVar[str]
    # Marks a response as coming from this service, e.g. "nim/" or "groq/".
    model_prefix: ClassVar[str]
    # Used in log lines and error messages.
    service_name: ClassVar[str]
    min_seconds_between_calls: ClassVar[float]
    error: ClassVar[type[FreeTierUnavailableError]] = FreeTierUnavailableError
    # Fields every request of this service carries, beyond the common ones.
    extra_payload: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        min_interval: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = api_key or os.environ.get(self.api_key_env)
        if not key:
            msg = f"no {self.api_key_env} in environment; add it to .env"
            raise self.error(msg)
        self.min_interval = self.min_seconds_between_calls if min_interval is None else min_interval
        self._sleep = sleep
        self._last_call = float("-inf")
        self._client = httpx.Client(
            base_url=base_url or self.base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}"},
        )

    # ---------------------------------------------------------- capabilities

    def capabilities(self, model: str) -> Capabilities:
        """These models accept sampling parameters and offer no prompt cache."""
        return Capabilities(
            supports_sampling_params=True,
            min_cacheable_prompt_tokens=10**9,
            thinking_on_by_default=False,
            supports_batch=False,
        )

    def count_tokens(self, request: CompletionRequest) -> int:
        """Approximate. These APIs have no counting endpoint, and calls are free."""
        text = (request.system or "") + "".join(m.content for m in request.messages)
        return max(1, len(text) // 4)

    # -------------------------------------------------------------- requests

    def _response_format(self, schema: dict[str, Any]) -> dict[str, Any]:
        """How this service is asked for a reply in a fixed JSON shape."""
        return {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}}

    def _payload(self, request: CompletionRequest) -> dict[str, Any]:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)

        model, effort = split_reasoning_effort(request.model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if effort is not None:
            payload["reasoning_effort"] = effort
        if request.output_schema is not None:
            payload["response_format"] = self._response_format(request.output_schema)
        payload.update(self.extra_payload)
        return payload

    def _wait_for_slot(self) -> None:
        wait = self._last_call + self.min_interval - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last_call = time.monotonic()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error = ""
        for attempt in range(MAX_RETRIES + 1):
            self._wait_for_slot()
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                last_error = str(exc)
            else:
                if response.status_code not in RETRY_STATUS:
                    if response.is_error:
                        msg = (
                            f"{self.service_name} API returned "
                            f"{response.status_code}: {response.text[:500]}"
                        )
                        raise self.error(msg)
                    body: dict[str, Any] = response.json()
                    return body
                last_error = f"HTTP {response.status_code}"
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_SECONDS * 2**attempt
                log.warning(
                    "%s request failed (%s); retrying in %.0fs",
                    self.service_name,
                    last_error,
                    delay,
                )
                self._sleep(delay)
        msg = f"{self.service_name} request failed after {MAX_RETRIES} retries: {last_error}"
        raise self.error(msg)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        body = self._post(self._payload(request))
        choice: dict[str, Any] = (body.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""

        parsed: dict[str, Any] | None = None
        if text and request.output_schema is not None:
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                log.warning(
                    "%s model returned non-JSON output (%d chars)", self.service_name, len(text)
                )
            else:
                parsed = candidate if isinstance(candidate, dict) else None

        usage_body: dict[str, Any] = body.get("usage") or {}
        usage = Usage(
            input_tokens=int(usage_body.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage_body.get("completion_tokens", 0) or 0),
        )
        return CompletionResponse(
            text=text,
            usage=usage,
            model=f"{self.model_prefix}{request.model}",
            stop_reason=choice.get("finish_reason"),
            parsed=parsed,
            request_id=body.get("id"),
        )

    # --------------------------------------------------------------- batches

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        """Not supported: these free tiers have no batch API."""
        msg = f"{self.service_name} has no batch API; run cells individually."
        raise NotImplementedError(msg)

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        msg = f"{self.service_name} has no batch API."
        raise NotImplementedError(msg)
