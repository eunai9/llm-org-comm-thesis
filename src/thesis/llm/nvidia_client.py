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

Everything shared with the other free, OpenAI-compatible service lives in
:mod:`thesis.llm.openai_compatible`.

The API key is read from ``NVIDIA_API_KEY``. It belongs in ``.env``, which is
gitignored.
"""

from __future__ import annotations

from typing import ClassVar

from thesis.llm.base import Provider
from thesis.llm.openai_compatible import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_RETRIES,
    REASONING_EFFORTS,
    RETRY_BASE_SECONDS,
    RETRY_STATUS,
    FreeTierUnavailableError,
    OpenAICompatibleClient,
    split_reasoning_effort,
)

__all__ = [
    "API_KEY_ENV",
    "BASE_URL",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_RETRIES",
    "MIN_SECONDS_BETWEEN_CALLS",
    "NIM_MODEL_PREFIX",
    "REASONING_EFFORTS",
    "RETRY_BASE_SECONDS",
    "RETRY_STATUS",
    "NvidiaClient",
    "NvidiaUnavailableError",
    "is_nim_model",
    "split_reasoning_effort",
]

BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY_ENV = "NVIDIA_API_KEY"
NIM_MODEL_PREFIX = "nim/"
# 40 requests per minute is one call every 1.5 seconds.
MIN_SECONDS_BETWEEN_CALLS = 1.5


class NvidiaUnavailableError(FreeTierUnavailableError):
    """Raised when the NVIDIA API is unreachable or refuses a request."""


def is_nim_model(model: str) -> bool:
    """Whether a model id came from the free NVIDIA catalog."""
    return model.startswith(NIM_MODEL_PREFIX)


class NvidiaClient(OpenAICompatibleClient):
    """The shared client body, pointed at NVIDIA's catalog."""

    # Not a ClassVar: the client protocol expects an instance variable.
    provider: Provider = "nvidia"
    base_url: ClassVar[str] = BASE_URL
    api_key_env: ClassVar[str] = API_KEY_ENV
    model_prefix: ClassVar[str] = NIM_MODEL_PREFIX
    service_name: ClassVar[str] = "NVIDIA"
    min_seconds_between_calls: ClassVar[float] = MIN_SECONDS_BETWEEN_CALLS
    error: ClassVar[type[FreeTierUnavailableError]] = NvidiaUnavailableError
