"""Groq client tests.

No network or API key is needed: the HTTP boundary is stubbed. The tests
check the two things that differ from the NVIDIA client, strict schema mode
and the dropped reasoning, plus that free output is never priced.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from thesis.llm.base import CompletionRequest, Message
from thesis.llm.groq_client import (
    GROQ_MODEL_PREFIX,
    MIN_SECONDS_BETWEEN_CALLS,
    GroqClient,
    GroqUnavailableError,
    is_groq_model,
)
from thesis.sim.schemas import RESPONSE_SCHEMA

MODEL = "openai/gpt-oss-120b"


def _request(**overrides: Any) -> CompletionRequest:
    base: dict[str, Any] = {
        "model": MODEL,
        "messages": [Message(role="user", content="Reply to this.")],
        "max_tokens": 512,
        "system": "You are a Director.",
    }
    base.update(overrides)
    return CompletionRequest(**base)


def _client_with(handler: Any, sleeps: list[float] | None = None) -> GroqClient:
    record = sleeps if sleeps is not None else []
    client = GroqClient(api_key="test-key", min_interval=0.0, sleep=record.append)
    client._client = httpx.Client(
        base_url="http://test",
        headers={"Authorization": "Bearer test-key"},
        transport=httpx.MockTransport(handler),
    )
    return client


def _completion(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-groq-1",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1600, "completion_tokens": 140},
    }


def _capture(captured: dict[str, Any]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        captured["auth"] = request.headers["Authorization"]
        captured["path"] = request.url.path
        return httpx.Response(200, json=_completion("{}"))

    return handler


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GroqUnavailableError, match="GROQ_API_KEY"):
        GroqClient()


def test_groq_models_are_identifiable() -> None:
    assert is_groq_model("groq/openai/gpt-oss-120b")
    assert not is_groq_model("nim/openai/gpt-oss-20b")
    assert not is_groq_model("openai/gpt-oss-120b")


def test_the_schema_is_sent_in_strict_mode() -> None:
    """Strict mode is the reason to prefer Groq: it guarantees the shape."""
    captured: dict[str, Any] = {}
    _client_with(_capture(captured)).complete(_request(output_schema=RESPONSE_SCHEMA))

    schema_format = captured["response_format"]["json_schema"]
    assert schema_format["strict"] is True
    assert schema_format["schema"] == RESPONSE_SCHEMA


def test_the_model_reasoning_is_not_requested() -> None:
    captured: dict[str, Any] = {}
    _client_with(_capture(captured)).complete(_request())

    assert captured["include_reasoning"] is False
    assert captured["path"] == "/chat/completions"
    assert captured["auth"] == "Bearer test-key"


def test_a_reasoning_effort_suffix_is_sent_as_a_setting() -> None:
    captured: dict[str, Any] = {}
    response = _client_with(_capture(captured)).complete(_request(model=f"{MODEL}@low"))

    assert captured["model"] == MODEL
    assert captured["reasoning_effort"] == "low"
    assert response.model == f"{GROQ_MODEL_PREFIX}{MODEL}@low"


def test_response_is_marked_and_parsed() -> None:
    body = {
        "subject": "Re: volumes",
        "body": "Numbers attached.",
        "decision": "accept",
        "confidence": "high",
        "reasoning_brief": "Routine.",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(json.dumps(body)))

    response = _client_with(handler).complete(_request(output_schema=RESPONSE_SCHEMA))

    assert response.model == f"{GROQ_MODEL_PREFIX}{MODEL}"
    assert response.parsed is not None
    assert response.parsed["decision"] == "accept"
    assert response.usage.input_tokens == 1600
    assert response.usage.output_tokens == 140


def test_the_daily_cap_is_reported_after_the_retries() -> None:
    """A 429 is retried; when it does not clear, the run must stop loudly."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limit reached"}})

    with pytest.raises(GroqUnavailableError, match="after"):
        _client_with(handler).complete(_request())


def test_calls_are_spaced_for_the_token_per_minute_cap() -> None:
    """8,000 tokens a minute at about 1,930 tokens per reply is 4 a minute."""
    assert MIN_SECONDS_BETWEEN_CALLS == 15.0
    sleeps: list[float] = []
    client = _client_with(_capture({}), sleeps)
    client.min_interval = MIN_SECONDS_BETWEEN_CALLS
    client.complete(_request())
    client.complete(_request())

    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= MIN_SECONDS_BETWEEN_CALLS


def test_groq_output_is_not_billable() -> None:
    from thesis.sim.run import _is_billable

    assert _is_billable(f"{GROQ_MODEL_PREFIX}{MODEL}", from_cache=False) is False
    assert _is_billable("claude-opus-5", from_cache=False) is True
