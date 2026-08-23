"""Web-demo tests.

Uses Flask's test client, so no server, browser, or Ollama instance is
needed. Module state (personas, scenarios, the backend client) is populated
directly rather than through _init_data(), so tests don't depend on the real
corpus or a running local model -- the same isolation principle as the
terminal demo's tests.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import thesis.sim.webdemo as webdemo_module
from tests.conftest import make_persona
from thesis.llm.base import CompletionResponse, Provider, Usage
from thesis.sim.scenario import build_scenarios


class _FixedClient:
    provider: Provider = "anthropic"
    model = "llama3.2:3b"

    def __init__(self, response: CompletionResponse) -> None:
        self._response = response

    def capabilities(self, model: str) -> Any:
        raise NotImplementedError

    def count_tokens(self, request: Any) -> int:
        raise NotImplementedError

    def complete(self, request: Any) -> CompletionResponse:
        return self._response

    def submit_batch(self, requests: Any) -> str:
        raise NotImplementedError

    def fetch_batch(self, batch_id: str) -> Any:
        raise NotImplementedError


def _valid_payload() -> dict[str, str]:
    return {
        "subject": "Re: volumes",
        "body": "Numbers attached.",
        "decision": "accept",
        "confidence": "high",
        "reasoning_brief": "Routine and within my remit.",
    }


def _valid_response(model: str = "local/llama3.2:3b") -> CompletionResponse:
    return CompletionResponse(text="{}", usage=Usage(), model=model, parsed=_valid_payload())


@pytest.fixture(autouse=True)
def _seeded_state(monkeypatch: Any) -> None:
    """Populate module state directly, bypassing _init_data()'s corpus/snapshot lookup."""
    persona = make_persona("r3_trading")
    monkeypatch.setattr(webdemo_module, "_PERSONAS", [persona])
    monkeypatch.setattr(
        webdemo_module,
        "_SCENARIOS_BY_KEY",
        {(s.task_type, s.direction, s.stakes, s.style): s for s in build_scenarios()},
    )


@pytest.fixture
def client() -> Any:
    webdemo_module.app.testing = True
    return webdemo_module.app.test_client()


def test_index_page_loads_and_lists_the_seeded_persona(client: Any) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Director" in resp.data
    assert b"Trading" in resp.data


def test_generate_with_builtin_scenario(client: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        webdemo_module, "_client_for", lambda backend: _FixedClient(_valid_response())
    )
    resp = client.post(
        "/api/generate",
        json={
            "backend": "local",
            "persona_id": "r3_trading",
            "mode": "builtin",
            "task_type": "approve_or_decline",
            "direction": "down",
            "stakes": "high",
            "style": "neutral",
        },
    )
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["decision"] == "accept"
    assert data["is_real"] is False
    assert "NOT thesis data" not in data["model"]  # the raw model id, label is a frontend concern


def test_generate_with_custom_scenario(client: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        webdemo_module, "_client_for", lambda backend: _FixedClient(_valid_response())
    )
    resp = client.post(
        "/api/generate",
        json={
            "backend": "local",
            "persona_id": "r3_trading",
            "mode": "custom",
            "direction": "up",
            "situation": "Budget review",
            "incoming": "Can we push our 1pm to tomorrow?",
        },
    )
    assert resp.status_code == 200
    assert json.loads(resp.data)["subject"] == "Re: volumes"


def test_generate_rejects_unknown_persona(client: Any) -> None:
    resp = client.post(
        "/api/generate",
        json={"backend": "local", "persona_id": "no-such-persona", "mode": "builtin"},
    )
    assert resp.status_code == 400


def test_generate_rejects_custom_scenario_missing_incoming_message(client: Any) -> None:
    resp = client.post(
        "/api/generate",
        json={
            "backend": "local",
            "persona_id": "r3_trading",
            "mode": "custom",
            "direction": "up",
            "incoming": "",
        },
    )
    assert resp.status_code == 400


def test_generate_rejects_unknown_builtin_combination(client: Any) -> None:
    resp = client.post(
        "/api/generate",
        json={
            "backend": "local",
            "persona_id": "r3_trading",
            "mode": "builtin",
            "task_type": "not_a_real_task",
            "direction": "up",
            "stakes": "routine",
        },
    )
    assert resp.status_code == 400


def test_generate_surfaces_ollama_unavailable_as_503(client: Any, monkeypatch: Any) -> None:
    from thesis.llm.ollama_client import OllamaUnavailableError

    def _raise(backend: str) -> Any:
        raise OllamaUnavailableError("no server")

    monkeypatch.setattr(webdemo_module, "_client_for", _raise)
    resp = client.post(
        "/api/generate",
        json={
            "backend": "local",
            "persona_id": "r3_trading",
            "mode": "builtin",
            "task_type": "approve_or_decline",
            "direction": "down",
            "stakes": "high",
            "style": "neutral",
        },
    )
    assert resp.status_code == 503


def test_generate_surfaces_invalid_response_as_502(client: Any, monkeypatch: Any) -> None:
    """A schema drift must be shown to the user, not returned as a fake 200."""
    bad = CompletionResponse(
        text="{}",
        usage=Usage(),
        model="local/llama3.2:3b",
        parsed={
            "subject": "s",
            "body": "b",
            "decision": "maybe",
            "confidence": "high",
            "reasoning_brief": "r",
        },
    )
    monkeypatch.setattr(webdemo_module, "_client_for", lambda backend: _FixedClient(bad))
    resp = client.post(
        "/api/generate",
        json={
            "backend": "local",
            "persona_id": "r3_trading",
            "mode": "builtin",
            "task_type": "approve_or_decline",
            "direction": "down",
            "stakes": "high",
            "style": "neutral",
        },
    )
    assert resp.status_code == 502


def test_generate_marks_a_real_provider_model_as_real(client: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        webdemo_module,
        "_client_for",
        lambda backend: _FixedClient(_valid_response(model="claude-opus-5")),
    )
    resp = client.post(
        "/api/generate",
        json={
            "backend": "local",
            "persona_id": "r3_trading",
            "mode": "builtin",
            "task_type": "approve_or_decline",
            "direction": "down",
            "stakes": "high",
            "style": "neutral",
        },
    )
    assert json.loads(resp.data)["is_real"] is True
