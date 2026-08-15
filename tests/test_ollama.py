"""Local-model client tests.

No running Ollama server is required: the HTTP boundary is stubbed, because
what needs testing is the request shape, the response mapping, and above all
that local output is never treated as billable or as a real result.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from thesis.llm.base import CompletionRequest, Message
from thesis.llm.ollama_client import (
    LOCAL_MODEL_PREFIX,
    OllamaClient,
    OllamaUnavailableError,
    is_local_model,
)
from thesis.sim.schemas import RESPONSE_SCHEMA


def _request(**overrides: Any) -> CompletionRequest:
    base: dict[str, Any] = {
        "model": "llama3.2:3b",
        "messages": [Message(role="user", content="Reply to this.")],
        "max_tokens": 512,
        "system": "You are a Director.",
    }
    base.update(overrides)
    return CompletionRequest(**base)


def _client_with(handler: Any) -> OllamaClient:
    client = OllamaClient("llama3.2:3b")
    client._client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return client


def _ok_response(payload: dict[str, Any] | None = None) -> Any:
    body = payload or {
        "subject": "Re: volumes",
        "body": "Numbers attached.",
        "decision": "accept",
        "confidence": "high",
        "reasoning_brief": "Routine.",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(body)},
                "prompt_eval_count": 1200,
                "eval_count": 90,
                "done_reason": "stop",
            },
        )

    return handler


def test_local_models_are_identifiable() -> None:
    assert is_local_model("local/llama3.2:3b")
    assert not is_local_model("claude-opus-5")


def test_response_is_marked_as_local() -> None:
    """Local output must be distinguishable from a pinned provider model."""
    response = _client_with(_ok_response()).complete(_request())
    assert response.model.startswith(LOCAL_MODEL_PREFIX)
    assert is_local_model(response.model)


def test_response_is_parsed_and_usage_mapped() -> None:
    response = _client_with(_ok_response()).complete(_request())
    assert response.parsed is not None
    assert response.parsed["decision"] == "accept"
    assert response.usage.input_tokens == 1200
    assert response.usage.output_tokens == 90


def test_schema_is_sent_to_the_native_format_field() -> None:
    """The native API takes a JSON schema directly; that is what keeps a small
    model's output in the expected shape."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    _client_with(handler).complete(_request(output_schema=RESPONSE_SCHEMA))
    assert captured["format"] == RESPONSE_SCHEMA
    assert captured["stream"] is False


def test_system_prompt_is_sent_as_its_own_turn() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    _client_with(handler).complete(_request())
    roles = [m["role"] for m in captured["messages"]]
    assert roles[0] == "system"


def test_non_json_output_is_recorded_not_raised() -> None:
    """Small models drift from a schema; one bad reply must not abort a run."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "Sure! Here you go."}})

    response = _client_with(handler).complete(_request())
    assert response.parsed is None
    assert response.text == "Sure! Here you go."


def test_unreachable_server_raises_a_useful_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaUnavailableError, match="server running"):
        _client_with(handler).complete(_request())


def test_is_available_never_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert _client_with(handler).is_available() is False


def test_capabilities_report_no_prompt_caching() -> None:
    """Reporting a small threshold would let callers expect savings that
    cannot occur locally."""
    caps = _client_with(_ok_response()).capabilities("llama3.2:3b")
    assert caps.min_cacheable_prompt_tokens > 10**6
    assert caps.supports_batch is False


def test_batch_is_refused_loudly() -> None:
    """Silently looping would look like batching while providing none of it."""
    client = _client_with(_ok_response())
    with pytest.raises(NotImplementedError):
        client.submit_batch([])
    with pytest.raises(NotImplementedError):
        client.fetch_batch("x")


def test_local_output_is_not_billable() -> None:
    from thesis.sim.run import _is_billable

    assert _is_billable("local/llama3.2:3b", from_cache=False) is False
    assert _is_billable("claude-opus-5", from_cache=False) is True
