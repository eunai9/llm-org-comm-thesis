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
    compare_with_length_matched_real,
    format_table,
    load_grid,
    parse_grid_arg,
    plot_models_vs_real,
    real_row,
    summarize_model,
)
from thesis.analysis.q1_real import fit_sentence_model
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


def _small_result_two_draws(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Same personas as :func:`_small_result`, but two draws per cell --
    what exercises the multi-draw pooling path in ``run_q1_analysis``."""
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
        n_replicates=2,
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


def test_summarize_model_uses_the_pooled_persona_fe_fit_for_a_multi_draw_grid(
    tmp_path: Path,
) -> None:
    """``summarize_model`` built ``primary.coefficient``/``p`` from
    ``sentence_model_persona_fe``, which ``run_q1_analysis`` always fits on
    draw 1 alone -- while ``coefficient_vb``/``vs_real`` in the same row
    came from ``grid_contrasts``, which already prefers the pooled fit for a
    multi-draw grid. So one reported row mixed a draw-1-only fit with pooled
    fits, the same bug ``grid_contrasts`` itself had before it was fixed
    (see test_q1.py's ``test_grid_contrasts_uses_the_pooled_fit_for_a_multi_draw_grid``).
    Found checking the Oct 4 full-design comparison against the Sep 23
    PROGRESS_llms.md entry, where the two disagreed (PROGRESS_llms.md, Oct 4)."""
    result = _small_result_two_draws(tmp_path)
    assert result.sentence_model_persona_fe_pooled is not None

    row = summarize_model(result, REAL_MANIFEST_FULL)

    pooled_coefficient, pooled_p = result.sentence_model_persona_fe_pooled.contrast("down")
    draw1_coefficient, draw1_p = result.sentence_model_persona_fe.contrast("down")
    primary = row["primary"]
    assert primary["coefficient"] == pytest.approx(pooled_coefficient, abs=1e-4)
    assert primary["p"] == pytest.approx(pooled_p, abs=1e-4)
    assert (pooled_coefficient, pooled_p) != (draw1_coefficient, draw1_p)


# ------------------------------------------------- length-matched real email


def _real_sentences_fixture() -> pd.DataFrame:
    """8 senders (half rank 2, half rank 4), each writing one "down" and one
    "lateral" message of 4 sentences. Sentence 0 carries the real direction
    signal (imperative when writing down, not when lateral); sentences 1-3
    are imperative regardless of direction -- padding that a full-length fit
    sees but a length-matched fit (cut to 1 sentence) does not. This makes
    the two fits give a different "down" coefficient by construction, not by
    chance, so a test against this fixture is deterministic."""
    rows = []
    for i in range(8):
        sender_id = i
        rank = 2 if i % 2 == 0 else 4
        for direction, first_sentence_imperative in (("down", 1), ("lateral", 0), ("up", 0)):
            message_uid = f"m{i}_{direction}"
            for sentence_index in range(4):
                rows.append(
                    {
                        "message_uid": message_uid,
                        "sentence_index": sentence_index,
                        "is_imperative": first_sentence_imperative if sentence_index == 0 else 1,
                        "direction": direction,
                        "sender_id": sender_id,
                        "sender_rank_level": rank,
                        "is_reply": True,
                    }
                )
    return pd.DataFrame(rows)


def test_compare_with_length_matched_real_uses_the_models_own_mean_length(
    tmp_path: Path,
) -> None:
    """``max_sentences`` must come from this grid's own sentences-per-reply,
    rounded to the nearest whole sentence, not a fixed number -- a 3-draw
    grid's ``sentence_features`` holds 3 draws' worth of sentences against
    1 draw's worth of replies, so the ratio must use both as given, not
    assume a single-draw grid's sentence/reply ratio of 1:1."""
    result = _small_result(tmp_path)
    expected = max(1, round(len(result.sentence_features) / len(result.reply_features)))

    comparison = compare_with_length_matched_real(result, _real_sentences_fixture())

    assert comparison["max_sentences"] == expected


def test_summarize_model_includes_the_length_matched_comparison_when_given_sentences(
    tmp_path: Path,
) -> None:
    """``summarize_model`` must wire the new comparison in when a caller
    supplies real-email sentences, using the exact same numbers
    ``compare_with_length_matched_real`` produces on its own -- not a
    second, independently-computed copy."""
    result = _small_result(tmp_path)
    real_sentences = _real_sentences_fixture()

    row = summarize_model(result, REAL_MANIFEST_FULL, real_sentences=real_sentences)

    expected = compare_with_length_matched_real(result, real_sentences)
    # The fixture's perfect separation (100% vs 0%) leaves statsmodels'
    # optimizer in a near-flat likelihood region, where two independent
    # fits of the same data can land a few units in the last place apart --
    # not a bug, so this compares with a tolerance, not exact equality.
    assert row["vs_real_length_matched"]["max_sentences"] == expected["max_sentences"]
    for key in ("is_imperative:down", "is_imperative:up"):
        for field in ("grid", "real", "difference"):
            assert row["vs_real_length_matched"][key][field] == pytest.approx(
                expected[key][field], abs=1e-2
            )


def test_summarize_model_omits_the_length_matched_comparison_without_sentences(
    tmp_path: Path,
) -> None:
    """Existing callers that do not have real-email sentences on hand must
    keep working exactly as before -- no key, not an error."""
    result = _small_result(tmp_path)

    row = summarize_model(result, REAL_MANIFEST_FULL)

    assert "vs_real_length_matched" not in row


def test_compare_with_length_matched_real_differs_from_the_full_length_comparison(
    tmp_path: Path,
) -> None:
    """The whole point of length-matching: real email's "down" coefficient
    cut to a short model's own reply length must not be the same number as
    real email's full-length "down" coefficient, because the fixture's
    padding sentences (1-3) are imperative regardless of direction and only
    a full-length fit sees them."""
    result = _small_result(tmp_path)
    real_sentences = _real_sentences_fixture()

    matched = compare_with_length_matched_real(result, real_sentences)
    full_length = fit_sentence_model(real_sentences, control_rank=True, per_email=True)

    assert matched["max_sentences"] < 4
    assert matched["is_imperative:down"]["real"] != round(full_length.contrast("down")[0], 4)
