"""A client for Groq's free tier (console.groq.com).

Groq serves both gpt-oss models, including the 120B one that NVIDIA's catalog
does not carry and this laptop cannot run. The free tier costs nothing, so
every response is priced at zero. Response model ids get a ``groq/`` prefix,
which keeps free rows easy to find and stops the cost ledger from pricing
them as paid calls.

**The daily caps decide how a long run is shaped.** The free tier allows 30
requests per minute, 1,000 requests per day and 200,000 tokens per day. The
client spaces its calls for the per-minute limit. The daily caps it cannot
dodge: once one is reached the API answers 429 until the next day. At about
1,800 tokens per reply that is roughly 110 replies a day, so the 183-pair run
spans two days. Every finished reply is cached, so continuing the next day
repeats nothing.

Two things differ from NVIDIA:

- **The schema is sent in strict mode.** Groq then constrains decoding to the
  schema, so the reply is guaranteed to match it. On NVIDIA, gpt-oss
  sometimes ran past the JSON until the token limit (PROGRESS_nvidia.md
  section 11), and only ``@low`` avoided it.
- **The reasoning is left out.** gpt-oss returns its thinking in a separate
  field, which this project never reads.

The API key is read from ``GROQ_API_KEY``. It belongs in ``.env``, which is
gitignored.
"""

from __future__ import annotations

from typing import Any, ClassVar

from thesis.llm.base import Provider
from thesis.llm.openai_compatible import (
    FreeTierUnavailableError,
    OpenAICompatibleClient,
)

BASE_URL = "https://api.groq.com/openai/v1"
API_KEY_ENV = "GROQ_API_KEY"
GROQ_MODEL_PREFIX = "groq/"
# 30 requests per minute is one call every 2 seconds.
MIN_SECONDS_BETWEEN_CALLS = 2.0


class GroqUnavailableError(FreeTierUnavailableError):
    """Raised when the Groq API is unreachable or refuses a request."""


def is_groq_model(model: str) -> bool:
    """Whether a model id came from Groq's free tier."""
    return model.startswith(GROQ_MODEL_PREFIX)


class GroqClient(OpenAICompatibleClient):
    """The shared client body, pointed at Groq."""

    # Not a ClassVar: the client protocol expects an instance variable.
    provider: Provider = "groq"
    base_url: ClassVar[str] = BASE_URL
    api_key_env: ClassVar[str] = API_KEY_ENV
    model_prefix: ClassVar[str] = GROQ_MODEL_PREFIX
    service_name: ClassVar[str] = "Groq"
    min_seconds_between_calls: ClassVar[float] = MIN_SECONDS_BETWEEN_CALLS
    error: ClassVar[type[FreeTierUnavailableError]] = GroqUnavailableError
    # The model's own reasoning is never read here, so it is not requested.
    extra_payload: ClassVar[dict[str, Any]] = {"include_reasoning": False}

    def _response_format(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Strict mode: Groq constrains decoding, so the reply matches the schema.

        It requires every property to be required and additional properties to
        be refused, which the reply schema already does.
        """
        return {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }
