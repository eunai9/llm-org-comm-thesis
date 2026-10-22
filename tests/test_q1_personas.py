from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from thesis.analysis.q1_personas import REPLY_OUTCOMES, analyse_persona_grid, generate_persona_grid
from thesis.llm.base import Capabilities, CompletionRequest, CompletionResponse, Provider, Usage
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.sim.persona_variants import persona_grid_prompt_hash


class FakeClient:
    """Answers every cell with one valid reply and never touches the network."""

    provider: Provider = "ollama"

    def __init__(self) -> None:
        self.calls: list[CompletionRequest] = []

    def capabilities(self, model: str) -> Capabilities:
        return Capabilities(
            supports_sampling_params=True,
            min_cacheable_prompt_tokens=10**9,
            thinking_on_by_default=False,
        )

    def count_tokens(self, request: CompletionRequest) -> int:
        return 100

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return CompletionResponse(
            text="{}",
            usage=Usage(input_tokens=200, output_tokens=20),
            model=f"local/{request.model}",
            stop_reason="end_turn",
            parsed={
                "subject": "Re: update",
                "body": "Send me the numbers by Friday.",
                "decision": "accept",
                "confidence": "medium",
                "reasoning_brief": "Routine.",
            },
        )

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        raise NotImplementedError

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        raise NotImplementedError


def test_generate_persona_grid_adds_variant_columns(tmp_path: Path) -> None:
    client = FakeClient()
    frame = generate_persona_grid(
        client,
        model="llama3.2:3b",
        design="pilot",
        stores={},
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        limit=12,
    )
    assert len(frame) == 12
    for col in ("base_persona_id", "tenure", "origin", "receiver_rank", "prompt_text_hash"):
        assert col in frame.columns
    assert (frame["prompt_text_hash"] == persona_grid_prompt_hash()).all()
    assert (frame["receiver_rank"] - frame["seniority_rank"]).isin([-1, 0, 1]).all()
    prompt = repr(client.calls[0])
    assert "You are writing to a" in prompt or "You are writing to an" in prompt
    assert "years" in prompt


_BODIES = {
    "down": ["Send the report today. Call me after.", "Could you send the report?"],
    "lateral": ["Let's talk about it this week.", "I will send it when it is ready."],
    "up": ["Could you look at this when you have time?", "I think we might need more time."],
}


def _synthetic_grid(drop: int = 0) -> pd.DataFrame:
    rows = []
    for p in range(8):
        tenure, origin = (2, 12)[p % 2], ("internal", "external")[(p // 2) % 2]
        pid = f"r3_trading_t{tenure}_{origin}_{p}"
        for direction, bodies in _BODIES.items():
            for i, body in enumerate(bodies * 3):
                rows.append(
                    {
                        "cell_id": f"{pid}__{direction}__{i}",
                        "persona_id": pid,
                        "base_persona_id": "r3_trading",
                        "tenure": tenure,
                        "origin": origin,
                        "direction": direction,
                        "scenario_id": f"approve_or_decline__{direction}__high__neutral",
                        "stakes": "high",
                        "task_type": "approve_or_decline",
                        "replicate": 1,
                        "body": body,
                    }
                )
    frame = pd.DataFrame(rows)
    return frame.iloc[drop:].reset_index(drop=True)


def test_analysis_finds_more_orders_writing_down() -> None:
    result = analyse_persona_grid(_synthetic_grid(), {("could", "you"): 1.0})
    assert result["n_personas"] == 8
    assert result["primary"]["down"]["coefficient"] > 0
    # No synthetic body thanks anyone, so gratitude never varies and is skipped.
    assert set(result["reply_level"]) == set(REPLY_OUTCOMES) - {"gratitude"}
    assert {"tenure", "origin"} <= set(result["traits"])


def test_analysis_runs_with_missing_rows() -> None:
    result = analyse_persona_grid(_synthetic_grid(drop=5), {})
    assert result["n_replies"] == len(_synthetic_grid()) - 5
