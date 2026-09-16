"""Tests for the decide-first prompt variant.

The default prompt must stay byte for byte the same, because every cached
reply is keyed on its exact text. The decide-first variant must differ only in
field order and in the passages that describe that order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from thesis.config import load_config
from thesis.llm.base import (
    Capabilities,
    CompletionRequest,
    CompletionResponse,
    Provider,
    Usage,
)
from thesis.llm.cache import ResponseCache, cache_key
from thesis.llm.cost import CostLedger
from thesis.sim import prompt
from thesis.sim.grid import GridCell
from thesis.sim.persona import Persona, PersonaStyle
from thesis.sim.run import RunManifest, build_request, run_grid
from thesis.sim.scenario import build_scenarios
from thesis.sim.schemas import (
    DECIDE_FIRST_RESPONSE_SCHEMA,
    RESPONSE_SCHEMA,
    response_schema,
    validate_response,
)

DECIDE_FIRST_ORDER = ["reasoning_brief", "decision", "confidence", "subject", "body"]
DEFAULT_ORDER = ["subject", "body", "decision", "confidence", "reasoning_brief"]

# sha256 of the default prompt passages and schema, taken from the commit
# before the variant was added. A change here changes every cache key.
PINNED_TEXT = {
    "TASK_FRAMING": "8f57f1bfc612271e1b24b10fb5b60152b39e94d5476e968253ab0f94ad2fff59",
    "ORGANIZATION_CONTEXT": "08308478ef457dfe071c1b19e0796d8dde862e696ed6730a2b4527ef608c4a3d",
    "HIERARCHY_CONTEXT": "cbc8f46d801d46f5c7dd60a90bc143f4b99110d77d267e398f376f7f277f1e22",
    "DECISION_TAXONOMY": "1062f44ac4d5346bc2bc50660f0f16c065b196088dcf83e05fbe296439f44b42",
    "OUTPUT_INSTRUCTION": "a905f4ccc5259b5d1b5c3238245233f2aefe19747dd961b9b05841f917e4e815",
}
PINNED_SCHEMA = "6f7c9e1e117730f7cdc988eaadb6c681d4b30d4770897d1c4d79aea66d8d33fa"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cell() -> GridCell:
    persona = Persona(
        persona_id="r3_trading",
        seniority_rank=3,
        rank_label="Director",
        department="Trading",
        style=PersonaStyle(
            mean_tokens=50.0,
            mean_recipients=2.0,
            imperative_ratio=0.15,
            hedge_rate=0.03,
            deference_rate=0.005,
            question_ratio=0.09,
        ),
        n_people=25,
        n_messages=4236,
        derivation="cell",
    )
    return GridCell(
        cell_id="sim_local__r3_trading__s1",
        persona=persona,
        scenario=build_scenarios()[0],
        replicate=1,
        model="llama3.2:3b",
        role_label="sim_local",
    )


def _payload() -> dict[str, str]:
    return {
        "subject": "Re: volumes",
        "body": "Numbers attached.",
        "decision": "accept",
        "confidence": "high",
        "reasoning_brief": "I will send the numbers.",
    }


# ------------------------------------------------------- default is unchanged


def test_default_prompt_text_is_pinned() -> None:
    for name, digest in PINNED_TEXT.items():
        assert _sha(getattr(prompt, name)) == digest, name


def test_default_schema_is_pinned() -> None:
    """Hashed without sorting keys, so the field order is pinned too."""
    assert _sha(json.dumps(RESPONSE_SCHEMA)) == PINNED_SCHEMA
    assert list(RESPONSE_SCHEMA["properties"]) == DEFAULT_ORDER
    assert RESPONSE_SCHEMA["required"] == DEFAULT_ORDER


def test_default_variant_builds_the_same_prompt_as_no_variant() -> None:
    cell = _cell()
    plain = prompt.assemble(cell.persona, cell.scenario)
    named = prompt.assemble(cell.persona, cell.scenario, variant="default")
    assert plain == named
    assert prompt.DECISION_TAXONOMY.strip() in plain.stable_prefix
    assert prompt.OUTPUT_INSTRUCTION.strip() in plain.variable_suffix


def test_default_request_uses_the_default_schema() -> None:
    assert build_request(_cell(), []).output_schema is RESPONSE_SCHEMA
    assert response_schema() is RESPONSE_SCHEMA


# ------------------------------------------------------- decide-first variant


def test_decide_first_schema_puts_the_decision_before_the_email() -> None:
    assert list(DECIDE_FIRST_RESPONSE_SCHEMA["properties"]) == DECIDE_FIRST_ORDER
    assert DECIDE_FIRST_RESPONSE_SCHEMA["required"] == DECIDE_FIRST_ORDER
    assert DECIDE_FIRST_RESPONSE_SCHEMA["additionalProperties"] is False


def test_decide_first_schema_keeps_names_and_enums() -> None:
    default, decide = RESPONSE_SCHEMA["properties"], DECIDE_FIRST_RESPONSE_SCHEMA["properties"]
    assert set(default) == set(decide)
    for name in ("decision", "confidence"):
        assert default[name]["enum"] == decide[name]["enum"]


def test_decide_first_descriptions() -> None:
    props = DECIDE_FIRST_RESPONSE_SCHEMA["properties"]
    assert props["reasoning_brief"]["description"] == (
        "Before writing: one sentence on what you will do about this message, "
        "from your role's point of view."
    )
    assert props["decision"]["description"] == (
        "The stance you are taking on what was asked. Your email will carry it out. "
        "Use 'none' when the message asks for nothing decidable."
    )
    assert props["confidence"]["description"] == "How firmly you commit to that decision."
    assert (
        props["subject"]["description"] == RESPONSE_SCHEMA["properties"]["subject"]["description"]
    )
    assert props["body"]["description"] == (
        "The body of the reply, as it would be sent. It does what you decided above."
    )


def test_decide_first_text_replaces_only_the_order_passages() -> None:
    taxonomy = " ".join(prompt.DECISION_TAXONOMY_DECIDE_FIRST.split())
    instruction = " ".join(prompt.OUTPUT_INSTRUCTION_DECIDE_FIRST.split())
    assert (
        "Before you write, choose one of five stances on whatever was asked. Then write the "
        "reply that carries it out. Do not default to agreeing:" in taxonomy
    )
    assert "Every reply you write" not in taxonomy
    assert "Fill the fields in order." in instruction
    assert "separate bookkeeping" not in instruction
    assert "The `reasoning_brief` field is your plan, as one complete sentence." in instruction
    # Section 11 found bodies opening with "decline.". The rule stays.
    assert 'Do not open with the decision word. An email that begins "Decline."' in instruction


def test_decide_first_leaves_the_rest_of_the_prompt_unchanged() -> None:
    cell = _cell()
    default = prompt.assemble(cell.persona, cell.scenario)
    decide = prompt.assemble(cell.persona, cell.scenario, variant="decide_first")
    swapped_prefix = decide.stable_prefix.replace(
        prompt.DECISION_TAXONOMY_DECIDE_FIRST.strip(), prompt.DECISION_TAXONOMY.strip()
    )
    swapped_suffix = decide.variable_suffix.replace(
        prompt.OUTPUT_INSTRUCTION_DECIDE_FIRST.strip(), prompt.OUTPUT_INSTRUCTION.strip()
    )
    assert swapped_prefix == default.stable_prefix
    assert swapped_suffix == default.variable_suffix
    assert decide.cache_group == default.cache_group


def test_variant_reaches_the_request() -> None:
    request = build_request(_cell(), [], "decide_first")
    assert request.output_schema is DECIDE_FIRST_RESPONSE_SCHEMA
    assert "Before you write, choose one of five stances" in (request.system or "")
    assert "Fill the fields in order." in request.messages[0].content


def test_variants_get_different_cache_keys() -> None:
    """The key sorts dict keys, so field order alone would not split it.

    The required list and the changed text do, and this checks they are there.
    """
    cell = _cell()
    default_key = cache_key(build_request(cell, []), "ollama")
    assert default_key == cache_key(build_request(cell, [], "default"), "ollama")
    assert default_key != cache_key(build_request(cell, [], "decide_first"), "ollama")


def test_validation_accepts_both_field_orders() -> None:
    payload = _payload()
    reordered = {name: payload[name] for name in DECIDE_FIRST_ORDER}
    assert validate_response(payload) == payload
    assert validate_response(reordered) == reordered


# ---------------------------------------------------------- threading through


class _RecordingClient:
    provider: Provider = "ollama"

    def __init__(self) -> None:
        self.calls: list[CompletionRequest] = []

    def capabilities(self, model: str) -> Capabilities:
        return Capabilities(
            supports_sampling_params=True,
            min_cacheable_prompt_tokens=10**9,
            thinking_on_by_default=False,
            supports_batch=False,
        )

    def count_tokens(self, request: CompletionRequest) -> int:
        return 1

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return CompletionResponse(
            text=json.dumps(_payload()),
            usage=Usage(input_tokens=10, output_tokens=5),
            model="local/llama3.2:3b",
            parsed=_payload(),
        )

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        raise NotImplementedError

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        raise NotImplementedError


def test_run_grid_sends_the_variant_to_the_client(tmp_path: Path) -> None:
    client = _RecordingClient()
    rows = run_grid(
        [_cell()],
        client,
        {},
        load_config(),
        run_id="t",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        manifest=RunManifest(
            run_id="t",
            started_at="2026-09-14T00:00:00+00:00",
            git_commit="x",
            git_dirty=False,
            config_hash="x",
            models=["llama3.2:3b"],
        ),
        prompt_variant="decide_first",
    )
    assert len(rows) == 1
    assert client.calls[0].output_schema is DECIDE_FIRST_RESPONSE_SCHEMA
