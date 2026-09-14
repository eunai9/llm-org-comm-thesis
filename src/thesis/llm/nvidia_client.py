"""A client for NVIDIA's hosted model catalog (build.nvidia.com).

The catalog serves open-weight models through an OpenAI-compatible API. The
free Developer Program tier costs nothing, so every response is priced at
zero. Response model ids get a ``nim/`` prefix. This keeps free-tier rows easy
to find and stops the cost ledger from pricing them as paid calls.

The free tier allows about 40 requests per minute. The client spaces its calls
to stay under that limit and retries when the server answers 429 or 5xx.

A model id may end in ``@low``, ``@medium`` or ``@high``, for example
``openai/gpt-oss-20b@low``. The client then sends the plain id and asks for
that reasoning effort. The full id stays on every response and in the cache
key, so replies made with different efforts are never mixed up.
gpt-oss-20b needs ``@low``: without it, some replies run past the JSON until
the token limit.

The API key is read from ``NVIDIA_API_KEY``. It belongs in ``.env``, which is
gitignored.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
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

BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY_ENV = "NVIDIA_API_KEY"
NIM_MODEL_PREFIX = "nim/"

# A normal reply takes under 15 seconds. The free tier sometimes holds a
# request without answering, so a long timeout only delays the retry.
DEFAULT_TIMEOUT_SECONDS = 120.0
# 40 requests per minute is one call every 1.5 seconds.
MIN_SECONDS_BETWEEN_CALLS = 1.5
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 2.0
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


REASONING_EFFORTS = frozenset({"low", "medium", "high"})


def is_nim_model(model: str) -> bool:
    """Whether a model id came from the free NVIDIA catalog."""
    return model.startswith(NIM_MODEL_PREFIX)


def split_reasoning_effort(model: str) -> tuple[str, str | None]:
    """Split ``name@effort`` into the model name the API expects and the effort."""
    name, separator, effort = model.rpartition("@")
    if not separator:
        return model, None
    if effort not in REASONING_EFFORTS:
        msg = f"unknown reasoning effort {effort!r} in {model!r}; use one of {sorted(REASONING_EFFORTS)}"
        raise ValueError(msg)
    return name, effort


class NvidiaUnavailableError(RuntimeError):
    """Raised when the NVIDIA API is unreachable or refuses a request."""


class NvidiaClient:
    """Satisfies the client protocol against NVIDIA's hosted catalog.

    The model is taken from each request, so one client can serve several
    catalog models in the same run.
    """

    provider: Provider = "nvidia"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        min_interval: float = MIN_SECONDS_BETWEEN_CALLS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = api_key or os.environ.get(API_KEY_ENV)
        if not key:
            msg = f"no {API_KEY_ENV} in environment; add it to .env"
            raise NvidiaUnavailableError(msg)
        self.min_interval = min_interval
        self._sleep = sleep
        self._last_call = float("-inf")
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {key}"},
        )

    # ---------------------------------------------------------- capabilities

    def capabilities(self, model: str) -> Capabilities:
        """Catalog models accept sampling parameters and offer no prompt cache."""
        return Capabilities(
            supports_sampling_params=True,
            min_cacheable_prompt_tokens=10**9,
            thinking_on_by_default=False,
            supports_batch=False,
        )

    def count_tokens(self, request: CompletionRequest) -> int:
        """Approximate. The API has no counting endpoint, and calls are free."""
        text = (request.system or "") + "".join(m.content for m in request.messages)
        return max(1, len(text) // 4)

    # -------------------------------------------------------------- requests

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
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": request.output_schema},
            }
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
                        msg = f"NVIDIA API returned {response.status_code}: {response.text[:500]}"
                        raise NvidiaUnavailableError(msg)
                    body: dict[str, Any] = response.json()
                    return body
                last_error = f"HTTP {response.status_code}"
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_SECONDS * 2**attempt
                log.warning("NVIDIA request failed (%s); retrying in %.0fs", last_error, delay)
                self._sleep(delay)
        msg = f"NVIDIA request failed after {MAX_RETRIES} retries: {last_error}"
        raise NvidiaUnavailableError(msg)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        body = self._post(self._payload(request))
        choice: dict[str, Any] = (body.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""

        parsed: dict[str, Any] | None = None
        if text and request.output_schema is not None:
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                log.warning("NVIDIA model returned non-JSON output (%d chars)", len(text))
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
            model=f"{NIM_MODEL_PREFIX}{request.model}",
            stop_reason=choice.get("finish_reason"),
            parsed=parsed,
            request_id=body.get("id"),
        )

    # --------------------------------------------------------------- batches

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        """Not supported: the catalog has no batch API."""
        msg = "NVIDIA catalog has no batch API; run cells individually."
        raise NotImplementedError(msg)

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        msg = "NVIDIA catalog has no batch API."
        raise NotImplementedError(msg)
