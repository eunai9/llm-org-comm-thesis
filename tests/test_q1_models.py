"""Tests for the Q1 model comparison: argument parsing, the real-email row,
the interval arithmetic, the table, and the figure. No model is called and no
grid is refit; the fit itself is covered by tests/test_q1.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from thesis.analysis.plots import plot_effect_intervals
from thesis.analysis.q1 import generate_q1_grid, run_q1_analysis
from thesis.analysis.q1_models import (
    PRIMARY,
    Z_95,
    format_table,
    load_grid,
    parse_grid_arg,
    plot_models_vs_real,
    real_row,
    summarize_model,
)
from thesis.llm.base import Capabilities, CompletionRequest, CompletionResponse, Provider, Usage
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.sim.persona import Persona, PersonaStyle

REAL_MANIFEST = {"simulator_vs_real": {PRIMARY: {"real": 0.253, "real_p": 0.0004}}}


def test_parse_grid_arg_splits_label_and_path() -> None:
    label, path = parse_grid_arg("gpt-oss-20b=data/interim/grid.parquet")
    assert label == "gpt-oss-20b"
    assert path == Path("data/interim/grid.parquet")


def test_parse_grid_arg_keeps_an_equals_sign_in_the_path() -> None:
    label, path = parse_grid_arg("a=b=c")
    assert (label, path) == ("a", Path("b=c"))


@pytest.mark.parametrize("bad", ["no-separator", "=path-only", "label-only="])
def test_parse_grid_arg_rejects_a_missing_part(bad: str) -> None:
    with pytest.raises(ValueError, match="LABEL=PATH"):
        parse_grid_arg(bad)


def test_real_row_interval_is_centered_on_the_coefficient() -> None:
    row = real_row(REAL_MANIFEST)
    assert row["coefficient"] == pytest.approx(0.253)
    assert row["ci_high"] - row["coefficient"] == pytest.approx(Z_95 * row["se"], abs=1e-3)
    assert row["coefficient"] - row["ci_low"] == pytest.approx(Z_95 * row["se"], abs=1e-3)


def _model(coefficient: float, p: float) -> dict[str, object]:
    return {
        "n_replies": 238,
        "n_sentences": 719,
        "primary": {
            "coefficient": coefficient,
            "p": p,
            "ci_low": coefficient - 0.25,
            "ci_high": coefficient + 0.25,
        },
        "vs_real": {PRIMARY: {"difference_p": 0.4}},
    }


def test_format_table_has_one_row_per_model_plus_real() -> None:
    manifest = {
        "real": real_row(REAL_MANIFEST),
        "models": {"model a": _model(0.14, 0.296), "model b": _model(0.44, 0.0)},
    }
    lines = format_table(manifest).splitlines()
    assert len(lines) == 4
    assert lines[1].startswith("real email")
    assert lines[2].startswith("model a")
    assert "+0.140" in lines[2]


def test_plot_effect_intervals_writes_a_file(tmp_path: Path) -> None:
    out = plot_effect_intervals(
        ["real", "model"], [0.25, 0.1], [0.15, -0.1], [0.35, 0.3], tmp_path / "f.png", title="t"
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_effect_intervals_rejects_unequal_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_effect_intervals(["a", "b"], [0.1], [0.0], [0.2], tmp_path / "f.png", title="t")


def test_plot_models_vs_real_puts_real_email_first(tmp_path: Path) -> None:
    real = real_row(REAL_MANIFEST)
    out = plot_models_vs_real(real, {"model a": _model(0.14, 0.296)}, tmp_path / "m.png")
    assert out.exists()


def test_load_grid_counts_cache_and_generation_from_the_from_cache_column(
    tmp_path: Path,
) -> None:
    """load_grid used to hard-code n_from_cache=len(frame), n_generated=0,
    regardless of the grid's real split -- the from_cache column a saved
    grid already carries (thesis.sim.run._result_row) was sitting unread.
    """
    frame = pd.DataFrame(
        {
            "model": ["m", "m", "m"],
            "scenario_id": ["s1", "s2", "s3"],
            "from_cache": [True, False, False],
        }
    )
    path = tmp_path / "grid.parquet"
    frame.to_parquet(path)

    grid = load_grid(path)

    assert grid.n_from_cache == 1
    assert grid.n_generated == 2


REAL_MANIFEST_FULL = {
    "simulator_vs_real": {
        f"{outcome}:{level}": {"real": 0.2, "real_p": 0.01}
        for outcome in ("imperative_ratio", "is_imperative", "hedge_rate")
        for level in ("down", "up")
    }
}


class _FakeClient:
    """Direction-dependent body, no network -- the same pattern
    tests/test_q1.py's FakeClient uses, duplicated here rather than
    imported because it is test-only setup local to this file's scope."""

    provider: Provider = "ollama"

    def capabilities(self, model: str) -> Capabilities:
        return Capabilities(
            supports_sampling_params=True,
            min_cacheable_prompt_tokens=10**9,
            thinking_on_by_default=False,
        )

    def count_tokens(self, request: CompletionRequest) -> int:
        return 100

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        content = request.messages[0].content
        if "senior to you" in content:
            body = "Could you take a look at this when you have a moment?"
        elif "reports into" in content:
            body = "Send me the updated numbers by end of day."
        else:
            body = "Let's sync on this sometime this week."
        payload = {
            "subject": "Re: update",
            "body": body,
            "decision": "accept",
            "confidence": "medium",
            "reasoning_brief": "Routine, within my remit.",
        }
        return CompletionResponse(
            text="{}",
            usage=Usage(input_tokens=200, output_tokens=20),
            model=f"local/{request.model}",
            stop_reason="end_turn",
            parsed=payload,
        )

    def submit_batch(self, requests: object) -> str:
        raise NotImplementedError

    def fetch_batch(self, batch_id: str) -> None:
        raise NotImplementedError


def _small_result(tmp_path: Path):  # type: ignore[no-untyped-def]
    personas = [
        Persona(
            persona_id=f"p{i}",
            seniority_rank=i + 1,
            rank_label=f"Rank {i + 1}",
            department="Trading",
            style=PersonaStyle(
                mean_tokens=40.0 + 10.0 * (i + 1),
                mean_recipients=2.0,
                imperative_ratio=0.10 + 0.02 * (i + 1),
                hedge_rate=0.03,
                deference_rate=0.005,
                question_ratio=0.09,
            ),
            n_people=10,
            n_messages=100,
            derivation="cell",
        )
        for i in range(3)
    ]
    grid = generate_q1_grid(
        _FakeClient(),
        model="llama3.2:3b",
        personas=personas,
        stores={},
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
    )
    return run_q1_analysis(grid)


def test_summarize_model_reports_a_persona_clustered_p_alongside_the_vb_one(
    tmp_path: Path,
) -> None:
    """The VB fit's p-value understates uncertainty on a large effect
    (PROGRESS_llms.md, Q1 section). summarize_model must report the
    persona-clustered cross-check next to it, not swap it in silently, and
    the interval must be centered on the coefficient it belongs to."""
    result = _small_result(tmp_path)

    row = summarize_model(result, REAL_MANIFEST_FULL)

    vb_coefficient, vb_p = result.sentence_model.contrast("down")
    clustered_coefficient, clustered_p = result.sentence_model_persona_fe.contrast("down")
    primary = row["primary"]
    assert primary["p"] == pytest.approx(clustered_p, abs=1e-4)
    assert primary["p_vb"] == pytest.approx(vb_p, abs=1e-4)
    assert primary["coefficient"] == pytest.approx(clustered_coefficient, abs=1e-4)
    assert primary["coefficient_vb"] == pytest.approx(vb_coefficient, abs=1e-4)
    assert primary["ci_low"] < primary["coefficient"] < primary["ci_high"]
