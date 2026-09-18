"""judge_swap.py tests: design reconstruction and glue, not LLM calls.

Every mixed-model / interaction-model statistical behavior is already
covered by tests/test_hierarchy.py; judge.run.score_items and its cache/
validation behavior by tests/test_judge.py. What is new here: which
scenarios and cells the reconstructed 60-cell-per-generator design
contains, how two generator halves combine into 120 replies, how two
judges' scores combine into 240 rows with an overall-score column added,
the saturated-2x2 arithmetic the "old" half of the comparison table is
built from, and the comparison-table glue itself. No test calls a real
LLM: fake clients (the same pattern tests/test_q1.py and tests/test_run.py
use) stand in for Ollama.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from thesis.analysis import judge_swap
from thesis.analysis.judge_swap import (
    DEFAULT_GENERATORS,
    HISTORICAL_CELL_MEANS,
    HISTORICAL_INTERACTION_P_OVERALL,
    JUDGE_SWAP_GRID_PATH,
    JUDGE_SWAP_SCORES_PATH,
    JUDGE_SWAP_TWO_TONE_GRID_PATH,
    JUDGE_SWAP_TWO_TONE_SCORES_PATH,
    JudgeSwapComparison,
    build_judge_items,
    build_judge_swap_cells,
    build_judge_swap_scenarios,
    build_two_tone_manifest,
    combine_generator_grids,
    combine_judge_scores,
    compare_judge_swap,
    fit_subset,
    format_comparison_table,
    generate_judge_swap_grid,
    replies_for_power,
    run_two_tone_analysis,
    saturated_2x2_effects,
    score_judge_swap_replies,
    subset_to_neutral,
)
from thesis.judge.rubric import RUBRIC_ITEMS
from thesis.llm.base import Capabilities, CompletionRequest, CompletionResponse, Provider, Usage
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.sim.persona import Persona, PersonaStyle
from thesis.sim.prompt import prompt_text_hash
from thesis.sim.scenario import DIRECTIONS


def _persona(persona_id: str, *, seniority_rank: int = 3) -> Persona:
    """A distinct-looking persona per id/rank, mirroring test_q1.py's helper:
    two personas with identical style stats would render byte-identical
    prompts and collapse onto one cache entry."""
    return Persona(
        persona_id=persona_id,
        seniority_rank=seniority_rank,
        rank_label=f"Rank {seniority_rank}",
        department="Trading",
        style=PersonaStyle(
            mean_tokens=40.0 + 10.0 * seniority_rank,
            mean_recipients=2.0,
            imperative_ratio=0.10 + 0.02 * seniority_rank,
            hedge_rate=0.03,
            deference_rate=0.005,
            question_ratio=0.09,
        ),
        n_people=10,
        n_messages=100,
        derivation="cell",
    )


# ------------------------------------------------------------------ design


def test_build_judge_swap_scenarios_has_6_scenarios() -> None:
    assert len(build_judge_swap_scenarios()) == 6


def test_build_judge_swap_scenarios_is_neutral_tone_only() -> None:
    for scenario in build_judge_swap_scenarios():
        assert scenario.tone == "neutral"


def test_build_judge_swap_scenarios_covers_every_direction_per_task_type() -> None:
    scenarios = build_judge_swap_scenarios()
    task_types = {s.task_type for s in scenarios}
    assert task_types == {"approve_or_decline", "report_problem"}
    for task_type in task_types:
        subset = [s for s in scenarios if s.task_type == task_type]
        assert {s.direction for s in subset} == set(DIRECTIONS)
        assert len(subset) == len(DIRECTIONS)


def test_build_judge_swap_cells_reproduces_60_cells_per_generator() -> None:
    """The design this module exists to reconstruct: 10 personas x 6
    scenarios x 1 replicate = 60 cells per generator model -- matching
    section 23's "10 personas x 3 directions x 2 task types" count."""
    personas = [_persona(f"p{i}") for i in range(10)]
    cells = build_judge_swap_cells(personas, "llama3.2:3b", "gen_llama")
    assert len(cells) == 60
    assert len({c.persona.persona_id for c in cells}) == 10
    assert len({c.scenario.scenario_id for c in cells}) == 6
    assert {c.replicate for c in cells} == {1}


def test_build_judge_swap_cells_respects_n_replicates() -> None:
    cells = build_judge_swap_cells([_persona("p0")], "llama3.2:3b", "gen_llama", n_replicates=2)
    assert len(cells) == 6 * 2
    assert {c.replicate for c in cells} == {1, 2}


def test_build_judge_swap_cells_are_cache_ordered() -> None:
    personas = [_persona(f"p{i}") for i in range(3)]
    cells = build_judge_swap_cells(personas, "llama3.2:3b", "gen_llama")
    seen: set[str] = set()
    last_group = None
    for cell in cells:
        if cell.cache_group != last_group:
            assert cell.cache_group not in seen, "cache group is not contiguous"
            seen.add(cell.cache_group)
            last_group = cell.cache_group


def test_two_generators_produce_distinct_cell_ids() -> None:
    """Different role labels per generator model keep cell_id unique across
    the two halves this module concatenates -- the exact bug
    combine_generator_grids guards against at runtime."""
    personas = [_persona("p0")]
    llama_cells = build_judge_swap_cells(personas, "llama3.2:3b", "gen_llama3_2_3b")
    qwen_cells = build_judge_swap_cells(personas, "qwen2.5:3b", "gen_qwen2_5_3b")
    llama_ids = {c.cell_id for c in llama_cells}
    qwen_ids = {c.cell_id for c in qwen_cells}
    assert llama_ids.isdisjoint(qwen_ids)


# --------------------------------------------------------------- generation


class FakeClient:
    """A client that writes a persona-and-model-dependent body and never
    touches the network -- mirrors tests/test_q1.py's FakeClient."""

    provider: Provider = "ollama"

    def __init__(self, model: str) -> None:
        self.model = model
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
        payload = {
            "subject": "Re: update",
            "body": f"Reply from {self.model} about this.",
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

    def submit_batch(self, requests: Sequence[tuple[str, CompletionRequest]]) -> str:
        raise NotImplementedError

    def fetch_batch(self, batch_id: str) -> dict[str, CompletionResponse] | None:
        raise NotImplementedError


def test_generate_judge_swap_grid_produces_one_row_per_cell(tmp_path: Path) -> None:
    personas = [_persona(f"p{i}", seniority_rank=i + 1) for i in range(3)]
    grid = generate_judge_swap_grid(
        FakeClient("llama3.2:3b"),
        model="llama3.2:3b",
        personas=personas,
        stores={},
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
    )
    assert grid.n_cells == 3 * 6
    assert len(grid.frame) == grid.n_cells
    assert grid.n_generated == grid.n_cells
    assert grid.n_from_cache == 0
    assert set(grid.frame["model"]) == {"llama3.2:3b"}


def test_generate_judge_swap_grid_second_call_is_served_from_cache(tmp_path: Path) -> None:
    personas = [_persona("p0")]
    cache = ResponseCache(tmp_path / "cache")
    ledger = CostLedger(tmp_path / "ledger.csv")

    first = generate_judge_swap_grid(
        FakeClient("llama3.2:3b"),
        model="llama3.2:3b",
        personas=personas,
        stores={},
        cache=cache,
        ledger=ledger,
    )
    assert first.n_generated == 6

    second_client = FakeClient("llama3.2:3b")
    second = generate_judge_swap_grid(
        second_client,
        model="llama3.2:3b",
        personas=personas,
        stores={},
        cache=cache,
        ledger=ledger,
    )
    assert second.n_from_cache == 6
    assert second.n_generated == 0
    assert second_client.calls == []


def test_combine_generator_grids_concatenates_both_halves(tmp_path: Path) -> None:
    personas = [_persona("p0")]
    cache = ResponseCache(tmp_path / "cache")
    ledger = CostLedger(tmp_path / "ledger.csv")

    llama = generate_judge_swap_grid(
        FakeClient("llama3.2:3b"),
        model="llama3.2:3b",
        role_label="gen_llama",
        personas=personas,
        stores={},
        cache=cache,
        ledger=ledger,
    )
    qwen = generate_judge_swap_grid(
        FakeClient("qwen2.5:3b"),
        model="qwen2.5:3b",
        role_label="gen_qwen",
        personas=personas,
        stores={},
        cache=cache,
        ledger=ledger,
    )
    combined = combine_generator_grids([llama, qwen])
    assert len(combined) == 12
    assert set(combined["model"]) == {"llama3.2:3b", "qwen2.5:3b"}


def test_combine_generator_grids_rejects_duplicate_cell_ids(tmp_path: Path) -> None:
    personas = [_persona("p0")]
    cache = ResponseCache(tmp_path / "cache")
    ledger = CostLedger(tmp_path / "ledger.csv")

    # Same role_label for both halves -- the exact misconfiguration cell_id
    # uniqueness depends on avoiding.
    first = generate_judge_swap_grid(
        FakeClient("llama3.2:3b"),
        model="llama3.2:3b",
        role_label="gen_shared",
        personas=personas,
        stores={},
        cache=cache,
        ledger=ledger,
    )
    second = generate_judge_swap_grid(
        FakeClient("qwen2.5:3b"),
        model="qwen2.5:3b",
        role_label="gen_shared",
        personas=personas,
        stores={},
        cache=ResponseCache(tmp_path / "cache2"),
        ledger=ledger,
    )
    with pytest.raises(ValueError, match="duplicate cell_id"):
        combine_generator_grids([first, second])


# ------------------------------------------------------------------ judging


def test_build_judge_items_uses_cell_id_and_body() -> None:
    frame = pd.DataFrame(
        [
            {"cell_id": "c1", "body": "hello there", "persona_id": "p0", "model": "llama3.2:3b"},
            {"cell_id": "c2", "body": "goodbye now", "persona_id": "p1", "model": "qwen2.5:3b"},
        ]
    )
    items = build_judge_items(frame)
    assert [i.item_id for i in items] == ["c1", "c2"]
    assert [i.text for i in items] == ["hello there", "goodbye now"]
    assert all(i.is_generated for i in items)
    assert all(i.context is None for i in items)


class ScriptedJudgeClient:
    """Returns a fixed rubric score for every item -- enough to test the
    score-flattening glue, not the judging statistics themselves."""

    provider: Provider = "ollama"

    def __init__(self, score: int) -> None:
        self.score = score

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = {item.key: {"evidence": "because", "score": self.score} for item in RUBRIC_ITEMS}
        return CompletionResponse(
            text="{}",
            usage=Usage(input_tokens=100, output_tokens=50),
            model="local/llama3.2:3b",
            parsed=payload,
        )


def _replies_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"cell_id": "c1", "body": "hello there", "persona_id": "p0", "model": "llama3.2:3b"},
            {"cell_id": "c2", "body": "goodbye now", "persona_id": "p1", "model": "qwen2.5:3b"},
        ]
    )


def test_score_judge_swap_replies_adds_generator_judge_and_overall_score(tmp_path: Path) -> None:
    scores = score_judge_swap_replies(
        _replies_frame(),
        ScriptedJudgeClient(4),
        judge_model="llama3.2:3b",
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
    )
    assert scores.n_scored == 2
    assert set(scores.frame["judge"]) == {"llama3.2:3b"}
    assert scores.frame.set_index("item_id")["generator"].to_dict() == {
        "c1": "llama3.2:3b",
        "c2": "qwen2.5:3b",
    }
    assert (scores.frame["score_overall"] == 4.0).all()
    for item in RUBRIC_ITEMS:
        assert f"score_{item.key}" in scores.frame.columns


def test_combine_judge_scores_concatenates_both_judges(tmp_path: Path) -> None:
    replies = _replies_frame()
    cache = ResponseCache(tmp_path / "cache")
    ledger = CostLedger(tmp_path / "ledger.csv")
    llama_scores = score_judge_swap_replies(
        replies, ScriptedJudgeClient(3), judge_model="llama3.2:3b", cache=cache, ledger=ledger
    )
    qwen_scores = score_judge_swap_replies(
        replies, ScriptedJudgeClient(5), judge_model="qwen2.5:3b", cache=cache, ledger=ledger
    )
    combined = combine_judge_scores([llama_scores, qwen_scores])
    assert len(combined) == 4
    assert set(combined["judge"]) == {"llama3.2:3b", "qwen2.5:3b"}


# -------------------------------------------------------------- comparison


def test_saturated_2x2_effects_recovers_known_interaction() -> None:
    """The section-23 arithmetic this function exists to redo: reference
    generator/judge = qwen, alt = llama, over the published cell means."""
    generator_effect, judge_effect, interaction = saturated_2x2_effects(
        HISTORICAL_CELL_MEANS,
        generator_alt="llama3.2:3b",
        generator_ref="qwen2.5:3b",
        judge_alt="llama3.2:3b",
        judge_ref="qwen2.5:3b",
    )
    assert generator_effect == pytest.approx(-1.022, abs=1e-6)
    assert judge_effect == pytest.approx(0.631, abs=1e-6)
    assert interaction == pytest.approx(0.419, abs=1e-6)


def test_saturated_2x2_effects_reconstructs_exact_cell_means() -> None:
    """Round-trip check: baseline + both main effects + interaction must
    reproduce the fourth (alt, alt) cell mean exactly."""
    baseline = HISTORICAL_CELL_MEANS[("qwen2.5:3b", "qwen2.5:3b")]
    generator_effect, judge_effect, interaction = saturated_2x2_effects(
        HISTORICAL_CELL_MEANS,
        generator_alt="llama3.2:3b",
        generator_ref="qwen2.5:3b",
        judge_alt="llama3.2:3b",
        judge_ref="qwen2.5:3b",
    )
    reconstructed = baseline + generator_effect + judge_effect + interaction
    assert reconstructed == pytest.approx(HISTORICAL_CELL_MEANS[("llama3.2:3b", "llama3.2:3b")])


class _FakeInteractionModel:
    """A stand-in for InteractionModelResult exposing only what
    compare_judge_swap actually calls."""

    def __init__(
        self,
        main_effects: dict[tuple[str, str], tuple[float, float]],
        interactions: dict[tuple[str, str], tuple[float, float]],
    ) -> None:
        self._main_effects = main_effects
        self._interactions = interactions

    def main_effect(self, factor: str, level: str) -> tuple[float, float]:
        return self._main_effects[(factor, level)]

    def interaction(self, level1: str, level2: str) -> tuple[float, float]:
        return self._interactions[(level1, level2)]


def test_compare_judge_swap_builds_four_rows() -> None:
    overall = _FakeInteractionModel(
        main_effects={
            ("generator", "llama3.2:3b"): (-0.9, 0.02),
            ("judge", "llama3.2:3b"): (0.5, 0.03),
        },
        interactions={("llama3.2:3b", "llama3.2:3b"): (0.30, 0.09)},
    )
    plausibility = _FakeInteractionModel(
        main_effects={},
        interactions={("llama3.2:3b", "llama3.2:3b"): (0.10, 0.40)},
    )
    comparisons = compare_judge_swap(
        overall,
        plausibility,
        generator_alt="llama3.2:3b",
        generator_ref="qwen2.5:3b",
        judge_alt="llama3.2:3b",
        judge_ref="qwen2.5:3b",
    )
    assert len(comparisons) == 4
    interaction_row = next(c for c in comparisons if "overall rubric" in c.label)
    assert interaction_row.new_value == 0.30
    assert interaction_row.new_p_value == 0.09
    assert interaction_row.old_p_value == HISTORICAL_INTERACTION_P_OVERALL

    plausibility_row = next(c for c in comparisons if "plausibility" in c.label)
    assert plausibility_row.old_value != plausibility_row.old_value  # nan
    assert plausibility_row.new_value == 0.10


def test_format_comparison_table_includes_every_label_and_handles_nan() -> None:
    comparisons = [
        JudgeSwapComparison("a quantity", 1.0, 0.05, 1.1, 0.04),
        JudgeSwapComparison("plausibility only", float("nan"), 0.20, 0.15, 0.30),
    ]
    table = format_comparison_table(comparisons)
    assert "a quantity" in table
    assert "plausibility only" in table
    assert "n/a" in table
    assert "1.100" in table


def test_default_generators_are_keys_in_historical_cell_means() -> None:
    """The CLI default pair must actually match the published historical
    record it is compared against."""
    a, b = DEFAULT_GENERATORS
    assert (a, a) in HISTORICAL_CELL_MEANS
    assert (a, b) in HISTORICAL_CELL_MEANS
    assert (b, a) in HISTORICAL_CELL_MEANS
    assert (b, b) in HISTORICAL_CELL_MEANS


# ------------------------------------------------------- the two-tone design


def test_two_tone_design_gives_120_cells_per_generator() -> None:
    """The design PROGRESS.md section 52 runs: 10 personas x 12 scenarios x
    1 replicate, for each of the two generator models."""
    personas = [_persona(f"p{i}") for i in range(10)]
    cells = build_judge_swap_cells(personas, "llama3.2:3b", "gen_llama", design="two_tone")

    assert len(cells) == 120
    assert len({c.scenario.scenario_id for c in cells}) == 12
    assert len({c.persona.persona_id for c in cells}) == 10
    assert {c.replicate for c in cells} == {1}


def test_the_default_design_is_still_the_60_cell_pilot() -> None:
    """Adding the flag must not change what an existing caller gets."""
    personas = [_persona(f"p{i}") for i in range(10)]
    assert len(build_judge_swap_cells(personas, "llama3.2:3b", "gen_llama")) == 60


def test_the_pilot_scenarios_are_a_subset_of_the_two_tone_scenarios() -> None:
    """This is what lets one two-tone run also report section 41's design."""
    pilot = {s.scenario_id for s in build_judge_swap_scenarios("pilot")}
    two_tone = {s.scenario_id for s in build_judge_swap_scenarios("two_tone")}

    assert len(pilot) == 6
    assert len(two_tone) == 12
    assert pilot < two_tone


def test_the_two_tone_design_adds_only_the_assertive_tone() -> None:
    scenarios = build_judge_swap_scenarios("two_tone")

    assert {s.tone for s in scenarios} == {"neutral", "assertive"}
    assert {s.task_type for s in scenarios} == {"approve_or_decline", "report_problem"}


def test_the_two_tone_files_do_not_overwrite_the_pilot_files() -> None:
    """Section 41's data stays where it is."""
    assert JUDGE_SWAP_TWO_TONE_GRID_PATH != JUDGE_SWAP_GRID_PATH
    assert JUDGE_SWAP_TWO_TONE_SCORES_PATH != JUDGE_SWAP_SCORES_PATH


def _two_tone_reply_frame(tones: Sequence[str] = ("neutral", "assertive")) -> pd.DataFrame:
    """A reply frame shaped like a real run: 10 personas x 3 directions x
    both generator models, for each tone."""
    rows = []
    for persona in range(10):
        for tone in tones:
            for direction in DIRECTIONS:
                for generator in DEFAULT_GENERATORS:
                    scenario_id = f"approve_or_decline__{direction}__high__{tone}"
                    slug = generator.replace(":", "_").replace(".", "_")
                    rows.append(
                        {
                            "cell_id": f"gen_{slug}__p{persona}__{scenario_id}__r1",
                            "scenario_id": scenario_id,
                            "persona_id": f"p{persona}",
                            "model": generator,
                            "body": "a reply",
                        }
                    )
    return pd.DataFrame(rows)


def _two_tone_score_frame(
    tones: Sequence[str] = ("neutral", "assertive"), *, self_preference: float = 0.5
) -> pd.DataFrame:
    """Scores for that frame, with a known self-preference built in: the
    llama judge adds ``self_preference`` to llama-written replies only."""
    rng = np.random.default_rng(0)
    llama, qwen = DEFAULT_GENERATORS
    rows = []
    for reply in _two_tone_reply_frame(tones).itertuples():
        for judge in DEFAULT_GENERATORS:
            score = (
                3.0
                + (0.4 if reply.model == qwen else 0.0)
                + (0.3 if judge == llama else 0.0)
                + (self_preference if reply.model == llama and judge == llama else 0.0)
                + float(rng.normal(0.0, 0.3))
            )
            rows.append(
                {
                    "item_id": reply.cell_id,
                    "generator": reply.model,
                    "judge": judge,
                    "persona_id": reply.persona_id,
                    "score_overall": score,
                    "score_corpus_plausibility": score,
                }
            )
    return pd.DataFrame(rows)


def test_subset_to_neutral_keeps_only_the_neutral_rows() -> None:
    """Works on a reply frame, which names the scenario, and on a score
    frame, which carries the scenario inside the item id."""
    replies = _two_tone_reply_frame()
    scores = _two_tone_score_frame()

    assert len(replies) == 120
    assert len(subset_to_neutral(replies)) == 60
    assert subset_to_neutral(replies)["scenario_id"].str.endswith("neutral").all()
    assert len(scores) == 240
    assert len(subset_to_neutral(scores)) == 120


def test_fit_subset_recovers_the_injected_self_preference_with_a_standard_error() -> None:
    """Self-preference is the interaction term. It is built into this data
    at +0.5, so the fit has to find it, and say how precisely."""
    llama, qwen = DEFAULT_GENERATORS
    fit = fit_subset(
        _two_tone_reply_frame(),
        _two_tone_score_frame(self_preference=0.5),
        label="full",
        generator_alt=llama,
        generator_ref=qwen,
        judge_alt=llama,
        judge_ref=qwen,
    )
    effect = fit.effect("self_preference_overall")

    assert effect.coefficient == pytest.approx(0.5, abs=0.2)
    assert effect.std_error > 0
    assert [e.key for e in fit.effects] == [
        "generator_quality",
        "judge_generosity",
        "self_preference_overall",
        "self_preference_plausibility",
    ]


def test_run_two_tone_analysis_fits_the_neutral_half_separately() -> None:
    """The neutral half is section 41's design. Fitting both separates more
    data from a wider set of situations."""
    llama, qwen = DEFAULT_GENERATORS
    result = run_two_tone_analysis(
        _two_tone_reply_frame(),
        _two_tone_score_frame(),
        generator_alt=llama,
        generator_ref=qwen,
        judge_alt=llama,
        judge_ref=qwen,
    )

    assert result.full.n_replies == 120
    assert result.full.n_scores == 240
    assert result.neutral.n_replies == 60
    assert result.neutral.n_scores == 120
    full_se = result.full.effect("self_preference_overall").std_error
    neutral_se = result.neutral.effect("self_preference_overall").std_error
    assert full_se < neutral_se


def test_replies_for_power_grows_with_the_square_of_the_precision_gap() -> None:
    """Precision improves with the square root of the sample size, so twice
    the noise needs four times the replies."""
    at_target = replies_for_power(0.28, 0.28 / 2.8016, 240)
    twice_as_noisy = replies_for_power(0.28, 2 * 0.28 / 2.8016, 240)

    assert at_target == pytest.approx(240, rel=1e-3)
    assert twice_as_noisy == pytest.approx(960, rel=1e-3)


def test_build_two_tone_manifest_records_the_prompt_hash_and_both_subsets() -> None:
    """A result has to say which prompt produced it. Section 41's replies
    came from a prompt that had changed since, and no file recorded it."""
    llama, qwen = DEFAULT_GENERATORS
    result = run_two_tone_analysis(
        _two_tone_reply_frame(),
        _two_tone_score_frame(),
        generator_alt=llama,
        generator_ref=qwen,
        judge_alt=llama,
        judge_ref=qwen,
    )
    manifest = build_two_tone_manifest(
        result,
        generators=DEFAULT_GENERATORS,
        judges=DEFAULT_GENERATORS,
        n_from_cache=120,
        n_generated=120,
    )

    assert manifest["run"]["prompt_text_hash"] == prompt_text_hash()
    assert manifest["run"]["design"] == "two_tone"
    assert set(manifest["full"]["effects"]) == {
        "generator_quality",
        "judge_generosity",
        "self_preference_overall",
        "self_preference_plausibility",
    }
    assert manifest["neutral_only"]["n_replies"] == 60
    assert manifest["power"]["replies_needed"] > 0


def test_generate_judge_swap_grid_writes_the_prompt_hash_into_its_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The grid's own run manifest records the prompt too, the way q1.py and
    pairs.py have since section 51."""
    seen: list[Any] = []

    def _spy(cells: Any, client: Any, stores: Any, config: Any, **kwargs: Any) -> list[Any]:
        seen.append(kwargs["manifest"])
        return []

    monkeypatch.setattr(judge_swap, "run_grid", _spy)
    generate_judge_swap_grid(
        FakeClient("llama3.2:3b"),
        model="llama3.2:3b",
        personas=[_persona("p0")],
        stores={},
        cache=ResponseCache(tmp_path / "cache"),
        ledger=CostLedger(tmp_path / "ledger.csv"),
        design="two_tone",
    )

    assert seen[0].design["prompt_text_hash"] == prompt_text_hash()
    assert seen[0].design["design"] == "two_tone"
    assert seen[0].design["n_scenarios"] == 12
