"""NVIDIA catalog client tests.

No network or API key is needed: the HTTP boundary is stubbed. The tests
check the request shape, the response mapping, the retry rule, and that
free-tier output is never priced.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from thesis.llm.base import CompletionRequest, Message
from thesis.llm.nvidia_client import (
    MAX_RETRIES,
    NIM_MODEL_PREFIX,
    NvidiaClient,
    NvidiaUnavailableError,
    is_nim_model,
    split_reasoning_effort,
)
from thesis.sim.schemas import RESPONSE_SCHEMA

MODEL = "google/gemma-4-31b-it"


def _request(**overrides: Any) -> CompletionRequest:
    base: dict[str, Any] = {
        "model": MODEL,
        "messages": [Message(role="user", content="Reply to this.")],
        "max_tokens": 512,
        "system": "You are a Director.",
    }
    base.update(overrides)
    return CompletionRequest(**base)


def _client_with(handler: Any, sleeps: list[float] | None = None) -> NvidiaClient:
    record = sleeps if sleeps is not None else []
    client = NvidiaClient(api_key="test-key", min_interval=0.0, sleep=record.append)
    client._client = httpx.Client(
        base_url="http://test",
        headers={"Authorization": "Bearer test-key"},
        transport=httpx.MockTransport(handler),
    )
    return client


def _completion(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 90},
    }


def _ok_response() -> Any:
    body = {
        "subject": "Re: volumes",
        "body": "Numbers attached.",
        "decision": "accept",
        "confidence": "high",
        "reasoning_brief": "Routine.",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(json.dumps(body)))

    return handler


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(NvidiaUnavailableError, match="NVIDIA_API_KEY"):
        NvidiaClient()


def test_nim_models_are_identifiable() -> None:
    assert is_nim_model("nim/google/gemma-4-31b-it")
    assert not is_nim_model("google/gemma-4-31b-it")
    assert not is_nim_model("claude-opus-5")


def test_response_is_marked_and_parsed() -> None:
    response = _client_with(_ok_response()).complete(_request(output_schema=RESPONSE_SCHEMA))
    assert response.model == f"{NIM_MODEL_PREFIX}{MODEL}"
    assert response.parsed is not None
    assert response.parsed["decision"] == "accept"
    assert response.usage.input_tokens == 1200
    assert response.usage.output_tokens == 90
    assert response.stop_reason == "stop"
    assert response.request_id == "chatcmpl-1"


def test_request_shape() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        captured["auth"] = request.headers["Authorization"]
        captured["path"] = request.url.path
        return httpx.Response(200, json=_completion("{}"))

    _client_with(handler).complete(_request(output_schema=RESPONSE_SCHEMA))
    assert captured["path"] == "/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert captured["model"] == MODEL
    assert captured["max_tokens"] == 512
    assert captured["messages"][0]["role"] == "system"
    assert captured["response_format"]["json_schema"]["schema"] == RESPONSE_SCHEMA


def test_no_schema_means_no_response_format() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_completion("Hello."))

    response = _client_with(handler).complete(_request())
    assert "response_format" not in captured
    assert response.parsed is None
    assert response.text == "Hello."


def test_non_json_output_is_recorded_not_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion("Sure! Here you go."))

    response = _client_with(handler).complete(_request(output_schema=RESPONSE_SCHEMA))
    assert response.parsed is None
    assert response.text == "Sure! Here you go."


def test_rate_limit_is_retried() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, json={"detail": "Too Many Requests"})
        return httpx.Response(200, json=_completion("{}"))

    _client_with(handler, sleeps).complete(_request())
    assert len(calls) == 3
    assert sleeps == [2.0, 4.0]


def test_retries_stop_after_the_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(NvidiaUnavailableError, match="after"):
        _client_with(handler).complete(_request())


def test_client_error_is_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, text="model not found")

    with pytest.raises(NvidiaUnavailableError, match="404"):
        _client_with(handler).complete(_request())
    assert len(calls) == 1


def test_calls_are_spaced_to_the_rate_limit() -> None:
    sleeps: list[float] = []
    client = _client_with(_ok_response(), sleeps)
    client.min_interval = 1.5
    client.complete(_request())
    client.complete(_request())
    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= 1.5


def test_batch_is_refused_loudly() -> None:
    client = _client_with(_ok_response())
    with pytest.raises(NotImplementedError):
        client.submit_batch([])
    with pytest.raises(NotImplementedError):
        client.fetch_batch("x")


def test_nim_output_is_not_billable() -> None:
    from thesis.sim.run import _is_billable

    assert _is_billable(f"{NIM_MODEL_PREFIX}{MODEL}", from_cache=False) is False
    assert _is_billable("claude-opus-5", from_cache=False) is True


def test_retry_count_is_bounded() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429)

    with pytest.raises(NvidiaUnavailableError):
        _client_with(handler).complete(_request())
    assert len(calls) == MAX_RETRIES + 1


def test_reasoning_effort_suffix_is_sent_as_a_setting() -> None:
    """The API gets the plain model name; the saved reply keeps the full one."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_completion("{}"))

    response = _client_with(handler).complete(_request(model="openai/gpt-oss-20b@low"))
    assert captured["model"] == "openai/gpt-oss-20b"
    assert captured["reasoning_effort"] == "low"
    assert response.model == f"{NIM_MODEL_PREFIX}openai/gpt-oss-20b@low"


def test_a_model_without_suffix_sends_no_reasoning_effort() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_completion("{}"))

    _client_with(handler).complete(_request())
    assert captured["model"] == MODEL
    assert "reasoning_effort" not in captured
    assert split_reasoning_effort(MODEL) == (MODEL, None)


def test_an_unknown_reasoning_effort_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown reasoning effort"):
        split_reasoning_effort("openai/gpt-oss-20b@fast")
