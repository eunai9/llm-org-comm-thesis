"""Demo-script tests.

The demo is meant to run live in front of a supervisor, so what matters most
is that it never crashes visibly: an unreachable Ollama server, or a malformed
model response, should be shown as a clear message rather than a traceback.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.prompt import Prompt

from tests.conftest import make_persona
from thesis.llm.base import CompletionResponse, Provider, Usage
from thesis.sim.demo import _custom_scenario, _render
from thesis.sim.scenario import build_scenarios


def _captured_console() -> Console:
    """A real Console recording to a string, so Rich renderables (Panel,
    Table) are actually rendered rather than just str()'d into a repr."""
    return Console(record=True, width=100)


class _FixedClient:
    """Satisfies LLMClient fully -- the demo only calls complete(), but the
    type is what the function signature declares, so the double should too."""

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


def _valid_response(model: str = "local/llama3.2:3b") -> CompletionResponse:
    payload = {
        "subject": "Re: volumes",
        "body": "Numbers attached.",
        "decision": "accept",
        "confidence": "high",
        "reasoning_brief": "Routine and within my remit.",
    }
    return CompletionResponse(text="{}", usage=Usage(), model=model, parsed=payload)


def test_render_shows_a_free_mode_disclaimer(monkeypatch: Any) -> None:
    """A local or stub reply must never be presentable as a real result."""
    import thesis.sim.demo as demo_module

    test_console = _captured_console()
    monkeypatch.setattr(demo_module, "console", test_console)

    _render(make_persona(), build_scenarios()[0], _FixedClient(_valid_response()))

    output = test_console.export_text()
    assert "NOT thesis data" in output
    assert "llama3.2:3b" in output


def test_render_labels_a_real_model_differently(monkeypatch: Any) -> None:
    """A response from a pinned provider model must not be flagged as fake."""
    import thesis.sim.demo as demo_module

    test_console = _captured_console()
    monkeypatch.setattr(demo_module, "console", test_console)

    _render(
        make_persona(),
        build_scenarios()[0],
        _FixedClient(_valid_response(model="claude-opus-5")),
    )

    output = test_console.export_text()
    assert "NOT thesis data" not in output


def test_render_handles_malformed_output_without_raising(monkeypatch: Any) -> None:
    """A live demo must survive a bad response, not crash mid-presentation."""
    import thesis.sim.demo as demo_module

    monkeypatch.setattr(demo_module, "console", _captured_console())

    bad = CompletionResponse(text="not json", usage=Usage(), model="local/llama3.2:3b", parsed=None)
    _render(make_persona(), build_scenarios()[0], _FixedClient(bad))  # must not raise

    bad_schema = CompletionResponse(
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
    _render(make_persona(), build_scenarios()[0], _FixedClient(bad_schema))  # must not raise


def test_custom_scenario_carries_typed_input(monkeypatch: Any) -> None:
    import thesis.sim.demo as demo_module

    monkeypatch.setattr(demo_module, "_choose", lambda *a, **k: "up")
    monkeypatch.setattr(Prompt, "ask", lambda *a, **k: "typed answer")

    scenario = _custom_scenario()
    assert scenario.direction == "up"
    assert scenario.situation == "typed answer"
    assert scenario.incoming_message == "typed answer"


def test_falls_back_to_snapshot_when_corpus_is_absent(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A supervisor without the ~270MB processed corpus must still get real,
    corpus-derived personas -- not a crash reaching for files that don't exist."""
    import thesis.sim.demo as demo_module

    monkeypatch.setattr(demo_module, "_corpus_is_processed", lambda: False)
    monkeypatch.setattr(demo_module, "console", _captured_console())

    personas = demo_module._load_personas()
    assert len(personas) == 10
