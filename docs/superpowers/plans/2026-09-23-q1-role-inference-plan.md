# Q1 Layer 1: Blind Role Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the blind role-inference judge (Layer 1 of the Q1 redesign): score whether a reply's writer can be identified as senior, peer or junior to its recipient, on generated replies from four models and on real Enron email, with the validation the spec requires.

**Architecture:** A new `analysis/role_inference.py` module scores single replies (absolute form, three-way, real email included) and same-scenario reply pairs (paired form, binary, simulator only), following the request/cache/ledger pattern `judge/discrimination.py` already established for a non-rubric judge task. A new `analysis/blinding.py` strips identity markers before any reply reaches the judge. Three free-tier models are widened from a 24-scenario pilot to the full 144-scenario grid first, so the model comparison covers four models instead of one.

**Tech Stack:** Python, pandas, spaCy (already loaded via `parse_replies`), scikit-learn (`balanced_accuracy_score`, `confusion_matrix`), statsmodels (`proportion_confint`), the project's existing LLM client/cache/cost machinery (`thesis.llm.*`), Ollama for the judge model.

**Spec:** [docs/superpowers/specs/2026-09-23-q1-authority-judge-design.md](../specs/2026-09-23-q1-authority-judge-design.md) — this plan implements section 3 ("Layer 1: is authority in the text at all?"), section 4 (model widening), and validation checks 1, 2 and 4 of section 5. Layer 2 (the indicator panel, section 3's second half) and validation check 3's actual human coding session are follow-on work; this plan builds the coding tool but the coding itself is manual, off-plan work.

## Global Constraints

- New outcomes are added; nothing in `q1.py`, `hierarchy.py` or the existing grids is replaced. Sections 39, 51 and 54 of `PROGRESS.md` stay comparable.
- No regeneration of the Llama grid, no prompt change, no cache invalidation.
- The judge must never see a role label, a direction-framing sentence, a persona name, or any text from `thesis.sim.scenario._DIRECTION_FRAMING`. Verified by a static test on every constructed item, not assumed.
- The judge model must not be one of the four generating models (`llama3.2:3b`, `deepseek-ai/deepseek-v4-flash-0731`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`). Use `qwen2.5:3b` via Ollama, the project's existing free stand-in for "a different model family" (`judge_blindness.py`'s `DEFAULT_JUDGE_MODEL`).
- The cross-model comparison uses `replicate == 1` from every grid, including Llama's. Llama's three draws stay in use only for the reliability check, never for a precision advantage over the single-draw models.
- Free-tier and local models only. No paid API calls, per the project's standing decision.
- Every commit passes `black .`, `ruff check .`, `mypy .`, and the full test suite, in that order, before the commit is made.
- Push after each task's commit, so results are visible on GitHub as they land.

## Review Focus

- **A reply with zero sentences** (spaCy parses it to nothing, or the body is empty after cleaning): `render_item_block` must still produce a valid prompt, not raise. `parse_replies` already handles this upstream for `is_imperative`; role_inference's item builders must not assume `len(body) > 0`.
- **A judge response that fails schema validation** (missing field, an enum value outside `DIRECTIONS`): must be recorded as invalid and skipped, the same way `score_items` and `run_discrimination` already handle it, not raised and not silently dropped from the count.
- **A reply that legitimately contains the word "senior" or "junior" as ordinary business language** (a job title mentioned in the task itself, e.g. "the Senior Analyst report"): blinding strips known title strings from the *lexicon* (mid-text mentions), so this is caught by the same mechanism as a leaked role label, not a special case — a test must confirm ordinary title mentions are stripped, not just planted signatures.
- **A paired item where both replies are byte-identical** (a model gave up and produced the same one-liner regardless of direction): the paired judge is forced to answer A or B anyway. This must be counted as a real 50/50 trial, not filtered out — filtering it would inflate accuracy by removing the trials a non-differentiating model should lose.
- **Fewer than three distinct judge outputs on a repeated item** during the self-consistency check (the judge model is deterministic on some inputs and returns the identical answer three times): `decision_stability_n` must not divide by zero or crash on a degenerate all-agree case — this is exactly what `test_grid_draw_reliability_is_perfect_when_draws_match` already covers for draws, and the same code path is being reused here, but the reuse itself needs a test showing it accepts a judge-agreement input shaped like a reliability input rather than a draw input.

---

## File Structure

New files:

- `src/thesis/analysis/blinding.py` — `strip_identity(text)`. Standalone, no dependency on judge or LLM code, so it can be tested with plain strings.
- `src/thesis/analysis/role_inference.py` — schemas, request builders, result dataclasses, the two scoring loops (absolute, paired), item builders, metrics, the positive-control and self-consistency checks, and `main()`.
- `src/thesis/analysis/role_coding.py` — the human-coding page and stratified sample for validation check 3. Separate from `blind_review.py`: that module compares two models' replies to the same email for failure-mode coding; this one shows one reply and asks who it was written to. Different data shape, different question, so a new small module rather than bending an existing one.
- `tests/test_blinding.py`, `tests/test_role_inference.py`, `tests/test_role_coding.py`.

Modified:

- `src/thesis/analysis/draw_stability.py` — `_decision_stability_n` becomes public (`decision_stability_n`), since `role_inference.py` needs it from outside the module for the self-consistency check. No behavior change.
- `tests/test_draw_stability.py` — import updated to the public name.

Deliberately not touched: `data/features.py`, `hierarchy.py`, `q1.py`, `q1_real.py`, `judge/rubric.py`, `judge/prompt.py`, `judge/run.py`. The role-inference schema is a three-way categorical choice, not the fixed six-item 1–5 rubric those modules implement, so it gets its own request builder rather than a forced fit — the same design choice `judge/discrimination.py` already made for its own, differently-shaped question.

---

### Task 1: Commit the untracked human-coding infrastructure

**Status: done, commit `50bd3aa` (2026-09-25 session).**

`blind_review.py` and its test exist only on this laptop (handoff item 5). This plan's own human-coding module (Task 12) will need the same git hygiene, so fix the existing gap first.

**Files:**
- Modify (git add only, no code changes): `src/thesis/analysis/blind_review.py`, `tests/test_blind_review.py`

**Interfaces:** none — this task changes no code.

- [ ] **Step 1: Verify the files are untracked and unchanged from what the handoff describes**

Run: `cd ~/projects/thesis && git status --short src/thesis/analysis/blind_review.py tests/test_blind_review.py`
Expected: both listed with `??` (untracked).

- [ ] **Step 2: Run the existing test suite for this file**

Run: `python -m pytest tests/test_blind_review.py -v`
Expected: all tests pass. If any fail, stop and report — do not commit failing code as a "hygiene" commit.

- [ ] **Step 3: Run the quality gates on the two files**

Run: `black --check src/thesis/analysis/blind_review.py tests/test_blind_review.py && ruff check src/thesis/analysis/blind_review.py tests/test_blind_review.py && mypy src/thesis/analysis/blind_review.py tests/test_blind_review.py`
Expected: all three pass with no changes needed. If `black` or `ruff` want changes, apply them (`black src/thesis/analysis/blind_review.py tests/test_blind_review.py`) and re-run the test suite.

- [ ] **Step 4: Commit and push**

```bash
git add src/thesis/analysis/blind_review.py tests/test_blind_review.py
git commit -m "Commit blind_review.py and its test, untracked since Sep 13

Existed only on this laptop. No code change."
git push
```

---

### Task 2: Widen Q1 generation to three more models

**Status: not started.** Deliberately skipped for now (2026-09-26 session,
user decision): Tasks 3-5 don't touch grid data at all, and validation
checks 1-2 (spec section 5) only need "small samples" — the existing
238-240 row pilot grids for deepseek/gpt-oss-20b/gpt-oss-120b cover that.
Still needed eventually for the real cross-model headline comparison
(Task 11's `main()` hardcodes `_full.parquet` paths) and for Tasks 6+
once they read real grid data instead of synthetic test fixtures.

No new code. `python -m thesis.analysis.q1 --nvidia MODEL --design full` already exists; only the pilot grids (24 scenarios) have been run for these three models. This generates the full 144-scenario grid for each, matching Llama's design.

**Files:** none created or modified. Output goes to `data/interim/`, which is entirely gitignored (`data/*`), so nothing here is committed.

**Interfaces:**
- Produces: three parquet files later tasks read directly by path:
  - `data/interim/q1_direction_grid_deepseek_full.parquet`
  - `data/interim/q1_direction_grid_gpt_oss_20b_full.parquet`
  - `data/interim/q1_direction_grid_gpt_oss_120b_full.parquet`

- [ ] **Step 1: Generate the DeepSeek full grid**

Run: `cd ~/projects/thesis && source .venv/bin/activate && python -m thesis.analysis.q1 --nvidia deepseek-ai/deepseek-v4-flash-0731 --design full --out data/interim/q1_direction_grid_deepseek_full.parquet`

Expected: takes about 36 minutes at 40 requests/minute; ends with a log line `wrote 1440 rows to ...`.

- [ ] **Step 2: Generate the gpt-oss-20b full grid**

Run: `python -m thesis.analysis.q1 --nvidia openai/gpt-oss-20b@low --design full --out data/interim/q1_direction_grid_gpt_oss_20b_full.parquet`

The `@low` reasoning-effort suffix is required — without it, `PROGRESS_nvidia.md` records gpt-oss-20b running past the JSON output until it hits the 2,048-token limit.

Expected: `wrote 1440 rows to ...`.

- [ ] **Step 3: Generate the gpt-oss-120b full grid**

Run: `python -m thesis.analysis.q1 --nvidia openai/gpt-oss-120b@low --design full --out data/interim/q1_direction_grid_gpt_oss_120b_full.parquet`

Expected: `wrote 1440 rows to ...`.

- [ ] **Step 4: Verify each grid's shape and model column**

Run:
```bash
python -c "
import pandas as pd
for path, expected_model in [
    ('data/interim/q1_direction_grid_deepseek_full.parquet', 'deepseek-ai/deepseek-v4-flash-0731'),
    ('data/interim/q1_direction_grid_gpt_oss_20b_full.parquet', 'openai/gpt-oss-20b@low'),
    ('data/interim/q1_direction_grid_gpt_oss_120b_full.parquet', 'openai/gpt-oss-120b@low'),
]:
    f = pd.read_parquet(path)
    assert len(f) == 1440, f'{path}: expected 1440 rows, got {len(f)}'
    assert f['scenario_id'].nunique() == 144, f'{path}: expected 144 scenarios'
    models = f['model'].unique()
    assert list(models) == [expected_model], f'{path}: unexpected model column {models}'
    print(path, 'OK', len(f), 'rows')
"
```
Expected: three `OK` lines, no assertion errors.

- [ ] **Step 5: No commit** — `data/interim/` is gitignored. Move to Task 3.

---

### Task 3: Make the Fleiss-kappa reliability function public

**Status: done, commit `84f5de9` (2026-09-26 session).**

`decision_stability_n` (currently `_decision_stability_n`) computes agreement across any number of raters given a list of `pd.Series`, delegating to Cohen's kappa at exactly two and Fleiss' kappa above that. Task 10's self-consistency check needs it from `role_inference.py`, outside `draw_stability`'s own module. This is a rename with no behavior change.

**Files:**
- Modify: `src/thesis/analysis/draw_stability.py:197` (the function definition), `:415` (its one call site inside `grid_draw_reliability`)
- Modify: `tests/test_draw_stability.py:17-18` (the import)

**Interfaces:**
- Produces: `decision_stability_n(draws: Sequence[pd.Series]) -> DecisionStability`, importable as `from thesis.analysis.draw_stability import decision_stability_n`.

- [ ] **Step 1: Confirm the current test names and imports before renaming**

Run: `grep -n "_decision_stability_n" src/thesis/analysis/draw_stability.py tests/test_draw_stability.py`
Expected: one definition, one call site in `draw_stability.py`; one import and three call sites in the test file (matching the earlier exploration of this file).

- [ ] **Step 2: Rename in the source module**

In `src/thesis/analysis/draw_stability.py`, change:
```python
def _decision_stability_n(draws: Sequence[pd.Series]) -> DecisionStability:
```
to:
```python
def decision_stability_n(draws: Sequence[pd.Series]) -> DecisionStability:
    """Public because :mod:`thesis.analysis.role_inference` reuses this for
    judge self-consistency (agreement across repeated judge calls on the
    same item), not only for draw-to-draw agreement. The statistic itself
    doesn't care what produced the repeated observations."""
```
and update its one call site inside `grid_draw_reliability` from `_decision_stability_n(_draws("decision"))` to `decision_stability_n(_draws("decision"))`.

- [ ] **Step 3: Update the test file's import and call sites**

In `tests/test_draw_stability.py`, change the import from `_decision_stability_n` to `decision_stability_n`, and update its three call sites (`test_decision_stability_n_delegates_to_cohens_kappa_at_two_draws`, `test_decision_stability_n_is_one_for_full_agreement_across_three_draws`, `test_decision_stability_n_matches_fleiss_kappa_by_hand`) to call `decision_stability_n(...)` instead of `_decision_stability_n(...)`. The test function names themselves stay as they are — the underscore in the name is describing the function under test, and renaming the test functions too would only churn the diff with no benefit.

- [ ] **Step 4: Run the affected tests**

Run: `python -m pytest tests/test_draw_stability.py -v`
Expected: all pass, same count as before the rename.

- [ ] **Step 5: Quality gates and commit**

```bash
black src/thesis/analysis/draw_stability.py tests/test_draw_stability.py
ruff check src/thesis/analysis/draw_stability.py tests/test_draw_stability.py
mypy src/thesis/analysis/draw_stability.py tests/test_draw_stability.py
python -m pytest -q
git add src/thesis/analysis/draw_stability.py tests/test_draw_stability.py
git commit -m "draw_stability: make decision_stability_n public

role_inference.py (Q1 redesign, next commits) needs the Fleiss-kappa
generalization for judge self-consistency, not only draw agreement.
No behavior change."
git push
```

---

### Task 4: `blinding.py` — strip identity markers from a reply

**Status: done, commit `432a413` (2026-09-26 session).**

Strips two structural leaks (a greeting naming the recipient, a sign-off block naming the writer) and one lexical leak (a job title mentioned mid-body), using the project's own 36-title roster as the stripping lexicon rather than a hand-built list.

**Files:**
- Create: `src/thesis/analysis/blinding.py`
- Test: `tests/test_blinding.py`

**Interfaces:**
- Consumes: `thesis.data.roles.load_title_rank_table() -> dict[str, tuple[int, str]]` (keys are the title strings, e.g. `"VP Trading"`, `"Managing Director"`) — already read this way in `q1_real.py`.
- Produces: `strip_identity(text: str, *, titles: Iterable[str] | None = None) -> str`. `titles` defaults to the roster's own title strings; a caller may pass a smaller set for a test. Later tasks call `strip_identity(body)` with no `titles` argument.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_blinding.py
from __future__ import annotations

from thesis.analysis.blinding import strip_identity

TEST_TITLES = ("Vice President", "Managing Director", "Director")


def test_removes_a_signoff_block() -> None:
    text = "Sure, I'll get that over today.\n\nBest,\nJohn Smith"
    result = strip_identity(text, titles=TEST_TITLES)
    assert "John Smith" not in result
    assert "Sure, I'll get that over today." in result


def test_removes_a_signoff_with_a_title() -> None:
    text = "Approved.\n\nRegards,\nSarah Lee, Vice President of Trading"
    result = strip_identity(text, titles=TEST_TITLES)
    assert "Sarah Lee" not in result
    assert "Vice President" not in result


def test_removes_a_greeting_naming_the_recipient() -> None:
    text = "Hi Sarah,\n\nCan you send the figures over today?"
    result = strip_identity(text, titles=TEST_TITLES)
    assert "Sarah" not in result
    assert "Can you send the figures over today?" in result


def test_removes_a_title_mentioned_mid_body() -> None:
    text = "As the Managing Director I've decided to approve this."
    result = strip_identity(text, titles=TEST_TITLES)
    assert "Managing Director" not in result
    assert "decided to approve this" in result


def test_does_not_touch_ordinary_text_with_no_markers() -> None:
    text = "I need the trading team to confirm they can accommodate the new date."
    assert strip_identity(text, titles=TEST_TITLES) == text


def test_uses_the_real_title_roster_by_default() -> None:
    """No titles argument: falls back to the project's own 36-title roster,
    so a real Enron signature ("VP Trading", "Mng Dir Trading", ...) is
    caught without the caller enumerating titles by hand."""
    text = "Sounds good.\n\nBest,\nTom - VP Trading"
    result = strip_identity(text)
    assert "VP Trading" not in result
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_blinding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'thesis.analysis.blinding'`.

- [ ] **Step 3: Implement**

```python
# src/thesis/analysis/blinding.py
"""Strip identity markers from a reply before it reaches the role-inference
judge: a greeting naming the recipient, a sign-off block naming the writer,
and a job title mentioned anywhere in the body.

Generated replies rarely carry any of these -- the median reply is one
sentence and has no room for a sign-off. Real Enron email is the reason this
exists: a signature block naming a Vice President is exactly the kind of
leak that would let a judge guess direction from provenance rather than from
how the message is actually written, which is the whole point of the blind
task in :mod:`thesis.analysis.role_inference`.

The title lexicon is the project's own 36-entry roster
(:func:`thesis.data.roles.load_title_rank_table`), not a hand-built list --
these are the titles that can actually appear in this corpus's signatures.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from thesis.data.roles import load_title_rank_table

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|dear)\s+[A-Z][a-zA-Z'-]*[,:]?\s*$", re.IGNORECASE | re.MULTILINE
)
_SIGNOFF_START_RE = re.compile(
    r"^\s*(best|regards|sincerely|thanks|thank you|cheers)[,.]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _default_titles() -> tuple[str, ...]:
    return tuple(load_title_rank_table().keys())


def _strip_signoff(text: str) -> str:
    """Everything from the first sign-off line onward, dropped."""
    match = _SIGNOFF_START_RE.search(text)
    return text[: match.start()].rstrip() if match else text


def _strip_greeting(text: str) -> str:
    return _GREETING_RE.sub("", text)


def _strip_titles(text: str, titles: Iterable[str]) -> str:
    """Replace each title string, longest first so "Director" doesn't eat
    part of "Managing Director" before the longer match gets a chance."""
    result = text
    for title in sorted(titles, key=len, reverse=True):
        pattern = re.compile(re.escape(title), re.IGNORECASE)
        result = pattern.sub("", result)
    return result


def strip_identity(text: str, *, titles: Iterable[str] | None = None) -> str:
    """Remove greeting, sign-off, and job-title mentions from a reply body.

    ``titles`` defaults to the project's real 36-title roster. Order of
    operations matters: the sign-off is stripped whole (removing a trailing
    name along with any title next to it) before the mid-body title pass
    runs on what's left, so a title inside a stripped sign-off is never
    double-processed and a title still standing in the body (e.g. someone
    referring to their own role mid-sentence) is still caught.
    """
    resolved_titles = tuple(titles) if titles is not None else _default_titles()
    text = _strip_signoff(text)
    text = _strip_greeting(text)
    text = _strip_titles(text, resolved_titles)
    return re.sub(r"[ \t]+", " ", text).strip()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_blinding.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Quality gates and commit**

```bash
black src/thesis/analysis/blinding.py tests/test_blinding.py
ruff check src/thesis/analysis/blinding.py tests/test_blinding.py
mypy src/thesis/analysis/blinding.py tests/test_blinding.py
python -m pytest -q
git add src/thesis/analysis/blinding.py tests/test_blinding.py
git commit -m "Q1 redesign: blinding.py strips identity before the role judge

Greeting, sign-off and mid-body job-title mentions, using the project's
own 36-title roster. Needed for real Enron email, which can carry all
three; generated replies rarely do, but the same pass runs on both.

Part of the Layer-1 blind role-inference judge in
docs/superpowers/specs/2026-09-23-q1-authority-judge-design.md."
git push
```

---

### Task 5: `role_inference.py` — absolute-form schema, request and scorer

**Status: done, commit `e81ba37` (2026-09-26 session).** Caught one real
bug: the plan's own test fixture used a bare model name
(`"qwen2.5:3b"`) for the scripted response, so the `is_local_model`
billing guard was never actually exercised and `cost_usd()` raised on
an unpriced model. Fixed to `"local/qwen2.5:3b"`, matching the
`"local/"` prefix the real `OllamaClient.complete()` always adds
(`ollama_client.py:184`) and the convention `test_judge.py`'s
`test_local_model_scores_are_not_billed` already established.

The three-way judge: one reply, no label, guess `up` / `lateral` / `down`. Follows `judge/discrimination.py`'s shape exactly (own schema, own validator, own request builder, own scoring loop), because the schema here — one categorical field — doesn't fit the fixed six-item 1–5 rubric `judge/run.py` implements.

**Files:**
- Create: `src/thesis/analysis/role_inference.py`
- Test: `tests/test_role_inference.py`

**Interfaces:**
- Consumes: `thesis.judge.prompt.JudgeItem`, `thesis.judge.prompt.render_item_block` (unmodified, reused as-is); `thesis.llm.base.{CompletionRequest, CompletionResponse, Message, Provider}`; `thesis.llm.cache.{ResponseCache, cache_key}`; `thesis.llm.cost.{CostLedger, LedgerEntry, cost_usd}`; `thesis.llm.ollama_client.is_local_model`; `thesis.llm.stub_client.is_stub_model`; `thesis.sim.scenario.DIRECTIONS` (`("up", "lateral", "down")`).
- Produces (for Task 6 onward, in this same file): `RoleInferenceResult`, `RoleInferenceSummary`, `run_role_inference_absolute(items, client, *, model, cache, ledger, run_id) -> tuple[list[RoleInferenceResult], RoleInferenceSummary]`, `ABSOLUTE_SCHEMA: dict[str, Any]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_role_inference.py
from __future__ import annotations

from dataclasses import dataclass

from thesis.judge.prompt import JudgeItem
from thesis.llm.base import CompletionRequest, CompletionResponse, Provider, Usage
from thesis.llm.cache import ResponseCache
from thesis.llm.cost import CostLedger
from thesis.analysis.role_inference import (
    ABSOLUTE_SCHEMA,
    RoleInferenceResult,
    build_absolute_request,
    run_role_inference_absolute,
    validate_absolute_response,
    InvalidRoleInferenceResponseError,
)


@dataclass
class _ScriptedClient:
    responses: list[CompletionResponse]
    provider: Provider = "ollama"

    def __post_init__(self) -> None:
        self.calls: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return self.responses.pop(0)


def _item(item_id: str = "i1", text: str = "Sure, I'll take care of it.") -> JudgeItem:
    return JudgeItem(item_id=item_id, text=text, is_generated=True, source_id="msg_1")


def _response(payload: dict[str, object] | None, model: str = "qwen2.5:3b") -> CompletionResponse:
    return CompletionResponse(
        text="{}", usage=Usage(input_tokens=80, output_tokens=20), model=model, parsed=payload
    )


def test_schema_restricts_the_direction_field_to_the_three_values() -> None:
    assert set(ABSOLUTE_SCHEMA["properties"]["inferred_direction"]["enum"]) == {
        "up",
        "lateral",
        "down",
    }


def test_validate_accepts_a_well_formed_response() -> None:
    payload = {"evidence": "no directive language", "inferred_direction": "lateral"}
    assert validate_absolute_response(payload) == payload


def test_validate_rejects_a_direction_outside_the_enum() -> None:
    payload = {"evidence": "x", "inferred_direction": "sideways"}
    try:
        validate_absolute_response(payload)
        raise AssertionError("expected InvalidRoleInferenceResponseError")
    except InvalidRoleInferenceResponseError:
        pass


def test_request_never_contains_the_word_direction_labels() -> None:
    """The rendered prompt must be the reply and the task framing only --
    never a direction word, which would defeat blinding by suggesting the
    answer set outside the schema's own enum. The enum values are allowed
    to appear in the schema (a structural field, never shown as prose in
    the message text) but not in the system framing's free text."""
    request = build_absolute_request(_item(), "qwen2.5:3b")
    assert request.system is not None
    for leak_word in ("you are writing to", "reports into", "senior to you"):
        assert leak_word not in request.system.lower()


def test_run_scores_one_item_end_to_end() -> None:
    client = _ScriptedClient(
        [_response({"evidence": "no directive language", "inferred_direction": "lateral"})]
    )
    cache = ResponseCache(":memory:")
    ledger = CostLedger(":memory:")
    results, summary = run_role_inference_absolute(
        [_item()], client, model="qwen2.5:3b", cache=cache, ledger=ledger, run_id="test-run"
    )
    assert summary.n_scored == 1
    assert summary.n_invalid == 0
    assert results[0].inferred_direction == "lateral"
    assert results[0].item_id == "i1"


def test_invalid_response_is_counted_not_raised() -> None:
    client = _ScriptedClient([_response({"evidence": "x", "inferred_direction": "sideways"})])
    cache = ResponseCache(":memory:")
    ledger = CostLedger(":memory:")
    results, summary = run_role_inference_absolute(
        [_item()], client, model="qwen2.5:3b", cache=cache, ledger=ledger, run_id="test-run"
    )
    assert results == []
    assert summary.n_invalid == 1
    assert summary.n_scored == 0


def test_result_carries_the_true_label_the_judge_never_saw() -> None:
    """is_generated and the caller's own true direction are tracked on the
    result object, never rendered into the prompt -- the same split
    JudgeItem/DiscriminationResult already use."""
    client = _ScriptedClient(
        [_response({"evidence": "no directive language", "inferred_direction": "down"})]
    )
    cache = ResponseCache(":memory:")
    ledger = CostLedger(":memory:")
    results, _ = run_role_inference_absolute(
        [_item()], client, model="qwen2.5:3b", cache=cache, ledger=ledger, run_id="test-run"
    )
    assert results[0].is_generated is True
```

Check whether `ResponseCache`/`CostLedger` accept `":memory:"` before relying on it — if the constructor signature only accepts a filesystem `Path` (as `judge_blindness.py`'s `ResponseCache(CACHE_DIR)` suggests), use `tmp_path / "cache"` and `tmp_path / "ledger.csv"` (pytest's built-in `tmp_path` fixture) instead, matching whatever `tests/test_judge.py` actually does for these two objects — check that file's fixture setup for the exact pattern before writing this step, since it exercises the identical cache/ledger construction this test needs.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_role_inference.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'thesis.analysis.role_inference'`.

- [ ] **Step 3: Implement the absolute-form scorer**

```python
# src/thesis/analysis/role_inference.py
"""Q1 Layer 1: blind role inference.

Give a judge a reply body with no role label, no direction-framing
sentence, and no persona -- ask who it was written to. This presupposes no
channel (no lexicon, no rule), so it answers a different question than
imperative_ratio or hedge_rate: not "does this contain an order", but "is
authority recoverable from the text at all". Real Enron email gives the
ceiling; each generating model is measured against it.

Two forms:

- **Absolute**: one reply, three-way choice (up / lateral / down). Chance
  is 33%. Runs on generated replies from every model and on real Enron
  email, so it produces the headline model-vs-real comparison.
- **Paired**: two replies to the identical scenario (same task_type,
  stakes and tone -- the scenario grid guarantees this, see
  :mod:`thesis.sim.scenario`), one written up and one down. Binary choice:
  which was written to the more senior recipient? Chance is 50%. More
  sensitive because it cancels scenario content exactly rather than only on
  average, and it only makes sense on the simulator, where the same
  scenario can be regenerated at every direction -- real email has no such
  pairing.

Schema and request-building follow :mod:`thesis.judge.discrimination`'s
shape, not :mod:`thesis.judge.run`'s: this is a categorical choice, not the
fixed six-item 1-5 rubric ``judge/run.py`` implements, so it earns its own
request builder rather than a forced fit onto that rubric's schema.

Run with ``python -m thesis.analysis.role_inference``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Protocol

from thesis.judge.prompt import JudgeItem, render_item_block
from thesis.llm.base import CompletionRequest, CompletionResponse, Message, Provider
from thesis.llm.cache import ResponseCache, cache_key
from thesis.llm.cost import CostLedger, LedgerEntry, cost_usd
from thesis.llm.ollama_client import is_local_model
from thesis.llm.stub_client import is_stub_model
from thesis.logging_setup import get_logger
from thesis.sim.scenario import DIRECTIONS

log = get_logger(__name__)

DEFAULT_JUDGE_MODEL = "qwen2.5:3b"

_ABSOLUTE_TASK_FRAMING = (
    "You will be shown a single reply to a workplace email. The reply may "
    "have been written by someone replying to a more senior colleague, a "
    "peer, or a more junior colleague.\n\n"
    "Judge only from how the reply itself is written -- word choice, how "
    "directly it makes requests or gives instructions, how much it "
    "explains or hedges -- which of the three is most likely.\n\n"
    "Give a short piece of evidence from the text, then your answer."
)

ABSOLUTE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "evidence": {
            "type": "string",
            "description": "A short, specific quote or paraphrase from the reply that supports your answer.",
        },
        "inferred_direction": {
            "type": "string",
            "enum": list(DIRECTIONS),
            "description": (
                "'up' if the reply looks written to someone more senior than the "
                "writer, 'down' if to someone more junior, 'lateral' if to a peer."
            ),
        },
    },
    "required": ["evidence", "inferred_direction"],
    "additionalProperties": False,
}


class InvalidRoleInferenceResponseError(ValueError):
    """Raised when a role-inference response does not satisfy its shape."""


def validate_absolute_response(payload: dict[str, Any]) -> dict[str, Any]:
    if "inferred_direction" not in payload or "evidence" not in payload:
        msg = "role-inference response missing 'inferred_direction' or 'evidence'"
        raise InvalidRoleInferenceResponseError(msg)
    if payload["inferred_direction"] not in DIRECTIONS:
        msg = f"inferred_direction {payload['inferred_direction']!r} not in {DIRECTIONS}"
        raise InvalidRoleInferenceResponseError(msg)
    return payload


def build_absolute_request(item: JudgeItem, model: str, *, replicate: int = 1) -> CompletionRequest:
    """One absolute-form call. ``replicate`` becomes the cache-key draw
    index (see ``CompletionRequest.variant``'s docstring) -- the
    self-consistency check (Task 10) scores the same item three times and
    needs each draw in its own cache entry, not the first draw served
    three times."""
    return CompletionRequest(
        model=model,
        messages=[Message(role="user", content=render_item_block(item))],
        max_tokens=512,
        system=_ABSOLUTE_TASK_FRAMING,
        output_schema=ABSOLUTE_SCHEMA,
        cache_system=True,
        variant=replicate,
        metadata={"item_id": item.item_id, "task": "role_inference_absolute"},
    )


@dataclass(frozen=True, slots=True)
class RoleInferenceResult:
    """One absolute-form item's judged direction, alongside its true label.

    ``true_direction`` is the caller's own record of the assigned direction
    -- never rendered into the prompt, exactly as ``is_generated`` is
    tracked on ``DiscriminationResult`` without ever reaching the judge.
    """

    item_id: str
    source_id: str
    is_generated: bool
    true_direction: str
    model: str
    inferred_direction: str
    evidence: str
    from_cache: bool


@dataclass
class RoleInferenceSummary:
    n_requested: int = 0
    n_scored: int = 0
    n_invalid: int = 0
    n_from_cache: int = 0
    total_cost_usd: float = 0.0


class _CompletionClient(Protocol):
    provider: Provider

    def complete(self, request: CompletionRequest) -> CompletionResponse: ...


def run_role_inference_absolute(
    items: list[JudgeItem],
    client: _CompletionClient,
    *,
    model: str,
    cache: ResponseCache,
    ledger: CostLedger,
    run_id: str,
    true_directions: dict[str, str] | None = None,
    replicate: int = 1,
) -> tuple[list[RoleInferenceResult], RoleInferenceSummary]:
    """Score every absolute-form item, cache-first, mirroring
    :func:`thesis.judge.discrimination.run_discrimination`'s loop shape.

    ``true_directions`` maps ``item_id`` to the assigned direction the
    caller already knows and the judge never sees; defaults to an empty
    map (``true_direction`` then reads ``"unknown"``) for callers, such as
    the positive-control check, that don't need it tracked.
    """
    true_directions = true_directions or {}
    summary = RoleInferenceSummary(n_requested=len(items))
    results: list[RoleInferenceResult] = []

    for item in items:
        request = build_absolute_request(item, model, replicate=replicate)
        key = cache_key(request, client.provider)

        response = cache.get(key)
        if response is None:
            response = client.complete(request)
            cache.put(key, request, response, client.provider)
        else:
            summary.n_from_cache += 1

        if response.parsed is None:
            log.warning("item %s: no parseable structured output", item.item_id)
            summary.n_invalid += 1
            continue

        try:
            payload = validate_absolute_response(response.parsed)
        except InvalidRoleInferenceResponseError as exc:
            log.warning("item %s: failed validation: %s", item.item_id, exc)
            summary.n_invalid += 1
            continue

        results.append(
            RoleInferenceResult(
                item_id=item.item_id,
                source_id=item.source_id,
                is_generated=item.is_generated,
                true_direction=true_directions.get(item.item_id, "unknown"),
                model=response.model,
                inferred_direction=payload["inferred_direction"],
                evidence=payload["evidence"],
                from_cache=response.from_cache,
            )
        )
        summary.n_scored += 1

        billable = (
            not response.from_cache
            and not is_stub_model(response.model)
            and not is_local_model(response.model)
        )
        cost = cost_usd(response.model, response.usage) if billable else 0.0
        summary.total_cost_usd += cost
        ledger.record(
            LedgerEntry(
                run_id=run_id,
                provider=client.provider,
                model=response.model,
                call_kind="role_inference_absolute",
                usage=response.usage,
                from_cache=response.from_cache or not billable,
            )
        )

    return results, summary
```

- [ ] **Step 4: Fix the cache/ledger test fixtures against the real constructor signatures**

Before running the tests, check `tests/test_judge.py`'s exact `ResponseCache`/`CostLedger` construction (grep `ResponseCache(` and `CostLedger(` in that file) and adjust Step 1's test bodies to match — replace any `":memory:"` argument with whatever that file actually uses (most likely a `tmp_path`-derived path, given both classes are described elsewhere as file-backed).

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/test_role_inference.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 6: Quality gates and commit**

```bash
black src/thesis/analysis/role_inference.py tests/test_role_inference.py
ruff check src/thesis/analysis/role_inference.py tests/test_role_inference.py
mypy src/thesis/analysis/role_inference.py tests/test_role_inference.py
python -m pytest -q
git add src/thesis/analysis/role_inference.py tests/test_role_inference.py
git commit -m "Q1 redesign: role_inference.py absolute-form judge

One reply, no label, three-way guess: up / lateral / down. Own schema
and request builder, following judge/discrimination.py's shape rather
than judge/run.py's fixed six-item rubric, since this is a categorical
choice, not a 1-5 score.

Part of Layer 1 in
docs/superpowers/specs/2026-09-23-q1-authority-judge-design.md."
git push
```

---

### Task 6: `role_inference.py` — paired-form schema, request and scorer

Two replies to the identical scenario, one written up and one written down. Binary: which was written to the more senior recipient? Chance is 50%. Simulator only.

**Files:**
- Modify: `src/thesis/analysis/role_inference.py` (add to the same file)
- Test: `tests/test_role_inference.py` (add to the same file)

**Interfaces:**
- Consumes: nothing new beyond Task 5's imports.
- Produces: `PAIRED_SCHEMA`, `PairedItem` (dataclass: `pair_id: str`, `text_a: str`, `text_b: str`, `senior_slot: str` — which of "A"/"B" is the true senior-recipient reply, never rendered), `render_paired_block(item: PairedItem) -> str`, `build_paired_request(item, model, *, replicate=1) -> CompletionRequest`, `validate_paired_response`, `PairedResult`, `run_role_inference_paired(items, client, *, model, cache, ledger, run_id, replicate=1) -> tuple[list[PairedResult], RoleInferenceSummary]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_role_inference.py`:

```python
from thesis.analysis.role_inference import (
    PAIRED_SCHEMA,
    PairedItem,
    build_paired_request,
    render_paired_block,
    run_role_inference_paired,
    validate_paired_response,
)


def _paired_item(pair_id: str = "p1") -> PairedItem:
    return PairedItem(
        pair_id=pair_id,
        text_a="I need this by Friday.",
        text_b="Whenever you get a chance, could you take a look?",
        senior_slot="A",
    )


def test_paired_schema_restricts_the_answer_to_a_or_b() -> None:
    assert set(PAIRED_SCHEMA["properties"]["answer"]["enum"]) == {"A", "B"}


def test_paired_block_shows_both_replies_labeled_and_hides_the_answer() -> None:
    block = render_paired_block(_paired_item())
    assert "I need this by Friday." in block
    assert "Whenever you get a chance" in block
    assert "A" in block and "B" in block
    assert "senior_slot" not in block.lower()


def test_paired_validate_rejects_an_answer_outside_a_or_b() -> None:
    try:
        validate_paired_response({"evidence": "x", "answer": "C"})
        raise AssertionError("expected InvalidRoleInferenceResponseError")
    except InvalidRoleInferenceResponseError:
        pass


def test_paired_run_scores_one_pair_and_checks_it_against_the_true_slot() -> None:
    client = _ScriptedClient([_response({"evidence": "more direct", "answer": "A"})])
    cache = ResponseCache(":memory:")  # see Task 5 Step 4 note on the real fixture
    ledger = CostLedger(":memory:")
    results, summary = run_role_inference_paired(
        [_paired_item()], client, model="qwen2.5:3b", cache=cache, ledger=ledger, run_id="test-run"
    )
    assert summary.n_scored == 1
    assert results[0].judged_correctly is True  # answered A, true senior slot is A
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_role_inference.py -v -k paired`
Expected: FAIL — the new names don't exist yet.

- [ ] **Step 3: Implement the paired-form scorer**

Append to `src/thesis/analysis/role_inference.py`:

```python
_PAIRED_TASK_FRAMING = (
    "You will be shown two replies, A and B, both written by the same kind "
    "of person answering the identical email. One reply was written to "
    "someone more senior than the writer; the other was written to someone "
    "more junior.\n\n"
    "Judge only from how each reply is written -- word choice, directness, "
    "how much it explains or hedges -- which one, A or B, was written to "
    "the MORE SENIOR recipient.\n\n"
    "Give a short piece of evidence, then your answer."
)

PAIRED_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "evidence": {
            "type": "string",
            "description": "A short, specific comparison between A and B that supports your answer.",
        },
        "answer": {
            "type": "string",
            "enum": ["A", "B"],
            "description": "Which reply, A or B, was written to the more senior recipient.",
        },
    },
    "required": ["evidence", "answer"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class PairedItem:
    """Two replies to the identical scenario, one written up and one down.

    ``senior_slot`` ("A" or "B") records which text is the true
    senior-recipient reply -- read only by the calling code that scores the
    judge's answer, never rendered into :func:`render_paired_block`.
    """

    pair_id: str
    text_a: str
    text_b: str
    senior_slot: str


def render_paired_block(item: PairedItem) -> str:
    return (
        "## Reply A\n\n"
        f"{item.text_a}\n\n"
        "## Reply B\n\n"
        f"{item.text_b}"
    )


def build_paired_request(item: PairedItem, model: str, *, replicate: int = 1) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=[Message(role="user", content=render_paired_block(item))],
        max_tokens=512,
        system=_PAIRED_TASK_FRAMING,
        output_schema=PAIRED_SCHEMA,
        cache_system=True,
        variant=replicate,
        metadata={"pair_id": item.pair_id, "task": "role_inference_paired"},
    )


def validate_paired_response(payload: dict[str, Any]) -> dict[str, Any]:
    if "answer" not in payload or "evidence" not in payload:
        msg = "paired role-inference response missing 'answer' or 'evidence'"
        raise InvalidRoleInferenceResponseError(msg)
    if payload["answer"] not in ("A", "B"):
        msg = f"answer {payload['answer']!r} not in ('A', 'B')"
        raise InvalidRoleInferenceResponseError(msg)
    return payload


@dataclass(frozen=True, slots=True)
class PairedResult:
    pair_id: str
    model: str
    answer: str
    judged_correctly: bool
    evidence: str
    from_cache: bool


def run_role_inference_paired(
    items: list[PairedItem],
    client: _CompletionClient,
    *,
    model: str,
    cache: ResponseCache,
    ledger: CostLedger,
    run_id: str,
    replicate: int = 1,
) -> tuple[list[PairedResult], RoleInferenceSummary]:
    """Score every paired item, cache-first. Structurally identical to
    :func:`run_role_inference_absolute`'s loop; kept as a separate function
    rather than parameterized over both shapes, since the two result types
    and schemas differ enough that a shared loop would need its own branch
    per shape anyway -- see the module docstring's rationale for two forms."""
    summary = RoleInferenceSummary(n_requested=len(items))
    results: list[PairedResult] = []

    for item in items:
        request = build_paired_request(item, model, replicate=replicate)
        key = cache_key(request, client.provider)

        response = cache.get(key)
        if response is None:
            response = client.complete(request)
            cache.put(key, request, response, client.provider)
        else:
            summary.n_from_cache += 1

        if response.parsed is None:
            log.warning("pair %s: no parseable structured output", item.pair_id)
            summary.n_invalid += 1
            continue

        try:
            payload = validate_paired_response(response.parsed)
        except InvalidRoleInferenceResponseError as exc:
            log.warning("pair %s: failed validation: %s", item.pair_id, exc)
            summary.n_invalid += 1
            continue

        results.append(
            PairedResult(
                pair_id=item.pair_id,
                model=response.model,
                answer=payload["answer"],
                judged_correctly=payload["answer"] == item.senior_slot,
                evidence=payload["evidence"],
                from_cache=response.from_cache,
            )
        )
        summary.n_scored += 1

        billable = (
            not response.from_cache
            and not is_stub_model(response.model)
            and not is_local_model(response.model)
        )
        cost = cost_usd(response.model, response.usage) if billable else 0.0
        summary.total_cost_usd += cost
        ledger.record(
            LedgerEntry(
                run_id=run_id,
                provider=client.provider,
                model=response.model,
                call_kind="role_inference_paired",
                usage=response.usage,
                from_cache=response.from_cache or not billable,
            )
        )

    return results, summary
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_role_inference.py -v`
Expected: all tests from Task 5 and Task 6 PASS (11 total).

- [ ] **Step 5: Quality gates and commit**

```bash
black src/thesis/analysis/role_inference.py tests/test_role_inference.py
ruff check src/thesis/analysis/role_inference.py tests/test_role_inference.py
mypy src/thesis/analysis/role_inference.py tests/test_role_inference.py
python -m pytest -q
git add src/thesis/analysis/role_inference.py tests/test_role_inference.py
git commit -m "Q1 redesign: role_inference.py paired-form judge

Two replies to the identical scenario, one written up and one down --
binary, which was written to the more senior recipient. Cancels
scenario content exactly, since the scenario grid already guarantees
identical situation and incoming-message text across directions."
git push
```

---

### Task 7: Item builders, with the no-leak blinding test as the acceptance bar

Turn a generated grid and the cached real-email frame into `JudgeItem`s (absolute form) and a generated grid into `PairedItem`s (paired form), running every reply body through `strip_identity` first. The no-leak test here is the actual verification of the blinding constraint stated in Global Constraints — not a separate task, because it has to run against exactly the text these builders produce.

**Files:**
- Modify: `src/thesis/analysis/role_inference.py`
- Test: `tests/test_role_inference.py`

**Interfaces:**
- Consumes: `thesis.analysis.blinding.strip_identity`; a grid frame shaped like `data/interim/q1_direction_grid_full_3draws.parquet` (columns: `cell_id`, `direction`, `body`, `persona_id`, `scenario_id`, `task_type`, `stakes`, `replicate`, `model`); the real-email frame shaped like `data/interim/q1_real_emails.parquet` loaded via `thesis.analysis.q1_real.load_bodies` (columns: `message_uid`, `body_clean`) joined to `direction` from the emails table.
- Produces: `build_absolute_items_from_grid(frame: pd.DataFrame, *, replicate: int = 1) -> tuple[list[JudgeItem], dict[str, str]]` (items plus the `item_id -> true_direction` map `run_role_inference_absolute` takes), `build_absolute_items_from_real_email(bodies: pd.DataFrame, directions: pd.DataFrame) -> tuple[list[JudgeItem], dict[str, str]]`, `build_paired_items_from_grid(frame: pd.DataFrame, *, replicate: int = 1) -> list[PairedItem]`.

- [ ] **Step 1: Write the failing tests, including the no-leak assertion**

Add to `tests/test_role_inference.py`:

```python
import pandas as pd

from thesis.analysis.role_inference import (
    build_absolute_items_from_grid,
    build_absolute_items_from_real_email,
    build_paired_items_from_grid,
)
from thesis.sim.scenario import _DIRECTION_FRAMING


def _tiny_grid() -> pd.DataFrame:
    rows = []
    for direction in ("up", "lateral", "down"):
        rows.append(
            {
                "cell_id": f"cell_{direction}",
                "persona_id": "persona_1",
                "scenario_id": f"approve_or_decline__{direction}__high__neutral",
                "task_type": "approve_or_decline",
                "direction": direction,
                "stakes": "high",
                "replicate": 1,
                "model": "llama3.2:3b",
                "body": "Best,\nJohn Smith, Vice President\n\nApproved, go ahead.",
            }
        )
    return pd.DataFrame(rows)


def test_build_absolute_items_from_grid_returns_one_item_per_row() -> None:
    items, true_directions = build_absolute_items_from_grid(_tiny_grid())
    assert len(items) == 3
    assert {true_directions[i.item_id] for i in items} == {"up", "lateral", "down"}


def test_build_absolute_items_from_grid_strips_identity() -> None:
    items, _ = build_absolute_items_from_grid(_tiny_grid())
    for item in items:
        assert "John Smith" not in item.text
        assert "Vice President" not in item.text
    assert "Approved, go ahead." in items[0].text


def test_no_direction_framing_language_reaches_the_rendered_prompt() -> None:
    """The blinding constraint itself: none of the exact framing sentences
    thesis.sim.scenario uses to tell the *generating* model which
    direction it is writing in may appear anywhere in the text the judge
    is shown. This is the actual verification of the Global Constraints
    blinding rule, not a restatement of it."""
    items, _ = build_absolute_items_from_grid(_tiny_grid())
    for item in items:
        rendered = render_item_block(item).lower()
        for framing_sentence in _DIRECTION_FRAMING.values():
            assert framing_sentence.lower() not in rendered


def test_build_absolute_items_from_real_email() -> None:
    bodies = pd.DataFrame(
        {"message_uid": ["m1", "m2"], "body_clean": ["Approved.", "Please review by Friday."]}
    )
    directions = pd.DataFrame({"message_uid": ["m1", "m2"], "direction": ["lateral", "down"]})
    items, true_directions = build_absolute_items_from_real_email(bodies, directions)
    assert len(items) == 2
    assert all(not item.is_generated for item in items)
    assert true_directions[items[0].item_id] in ("lateral", "down")


def test_build_paired_items_from_grid_pairs_up_with_down_same_scenario() -> None:
    pairs = build_paired_items_from_grid(_tiny_grid())
    assert len(pairs) == 1  # one (task_type, stakes, tone) triple in the fixture
    assert pairs[0].senior_slot in ("A", "B")
```

Note: `render_item_block` needs importing into the test file alongside the existing `thesis.judge.prompt` import from Task 5.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_role_inference.py -v -k build_`
Expected: FAIL — the builder functions don't exist yet.

- [ ] **Step 3: Implement the item builders**

Append to `src/thesis/analysis/role_inference.py`:

```python
import numpy as np
import pandas as pd

from thesis.analysis.blinding import strip_identity


def build_absolute_items_from_grid(
    frame: pd.DataFrame, *, replicate: int = 1
) -> tuple[list[JudgeItem], dict[str, str]]:
    """One absolute-form item per reply at the given draw, identity-stripped.

    ``replicate`` defaults to 1 to honor the draw-balance rule (Global
    Constraints): every model contributes one draw to the absolute-form
    comparison, including Llama, which has three.
    """
    subset = frame[frame["replicate"] == replicate]
    items = [
        JudgeItem(
            item_id=str(row.cell_id),
            text=strip_identity(str(row.body)),
            is_generated=True,
            source_id=str(row.cell_id),
        )
        for row in subset.itertuples(index=False)
    ]
    true_directions = {str(row.cell_id): str(row.direction) for row in subset.itertuples(index=False)}
    return items, true_directions


def build_absolute_items_from_real_email(
    bodies: pd.DataFrame, directions: pd.DataFrame
) -> tuple[list[JudgeItem], dict[str, str]]:
    """One absolute-form item per real email. ``bodies`` has
    ``message_uid``/``body_clean`` (:func:`thesis.analysis.q1_real.load_bodies`'s
    shape); ``directions`` has ``message_uid``/``direction``
    (:data:`thesis.analysis.q1_real`'s cached emails table)."""
    merged = bodies.merge(directions[["message_uid", "direction"]], on="message_uid", how="inner")
    items = [
        JudgeItem(
            item_id=str(row.message_uid),
            text=strip_identity(str(row.body_clean)),
            is_generated=False,
            source_id=str(row.message_uid),
        )
        for row in merged.itertuples(index=False)
    ]
    true_directions = {str(row.message_uid): str(row.direction) for row in merged.itertuples(index=False)}
    return items, true_directions


def build_paired_items_from_grid(frame: pd.DataFrame, *, replicate: int = 1, seed: int = 20260923) -> list[PairedItem]:
    """One paired item per (task_type, stakes, tone) triple that has both
    an 'up' and a 'down' reply at the given draw -- 'lateral' is not part
    of this comparison, since the paired question is specifically about
    the two directions the writing-down effect concerns.

    A/B slot assignment is randomized per pair (seeded, for reproducible
    manifests) so the judge cannot learn "A is always senior" from
    position alone.
    """
    subset = frame[frame["replicate"] == replicate].assign(
        triple=lambda d: d["scenario_id"].str.split("__").str[0]
        + "__"
        + d["stakes"]
        + "__"
        + d["scenario_id"].str.split("__").str[-1]
    )
    rng = np.random.default_rng(seed)
    pairs: list[PairedItem] = []
    for triple, group in subset.groupby("triple"):
        by_direction = dict(zip(group["direction"], group["body"], strict=False))
        if "up" not in by_direction or "down" not in by_direction:
            continue
        up_text = strip_identity(str(by_direction["up"]))
        down_text = strip_identity(str(by_direction["down"]))
        up_is_a = bool(rng.integers(0, 2))
        text_a, text_b = (up_text, down_text) if up_is_a else (down_text, up_text)
        pairs.append(
            PairedItem(
                pair_id=str(triple),
                text_a=text_a,
                text_b=text_b,
                senior_slot="A" if up_is_a else "B",
            )
        )
    return pairs
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_role_inference.py -v`
Expected: all tests PASS (16 total across Tasks 5–7).

- [ ] **Step 5: Quality gates and commit**

```bash
black src/thesis/analysis/role_inference.py tests/test_role_inference.py
ruff check src/thesis/analysis/role_inference.py tests/test_role_inference.py
mypy src/thesis/analysis/role_inference.py tests/test_role_inference.py
python -m pytest -q
git add src/thesis/analysis/role_inference.py tests/test_role_inference.py
git commit -m "Q1 redesign: role_inference item builders, blinding verified

build_absolute_items_from_grid/_real_email, build_paired_items_from_grid.
Every reply body runs through strip_identity(). A dedicated test renders
each built item and asserts none of sim.scenario's direction-framing
sentences reach the text the judge is shown -- the actual check on the
blinding constraint, not just a restatement of it."
git push
```

---

### Task 8: Metrics — accuracy, Wilson interval, Cohen's kappa, confusion matrix

**Files:**
- Modify: `src/thesis/analysis/role_inference.py`
- Test: `tests/test_role_inference.py`

**Interfaces:**
- Consumes: `sklearn.metrics.{balanced_accuracy_score, confusion_matrix, cohen_kappa_score}`, `statsmodels.stats.proportion.proportion_confint`, `thesis.analysis.plots.plot_effect_intervals`.
- Produces: `AbsoluteMetrics` (dataclass: `n`, `accuracy`, `ci_low`, `ci_high`, `kappa`, `confusion: dict[str, dict[str, int]]`), `summarize_absolute(results: list[RoleInferenceResult]) -> AbsoluteMetrics`, `plot_accuracy_vs_real(metrics_by_label: dict[str, AbsoluteMetrics], path: Path) -> Path` (real email's label must be first in `metrics_by_label` so it draws in the reference color, matching `plot_effect_intervals`'s own documented convention).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_role_inference.py`:

```python
from thesis.analysis.role_inference import AbsoluteMetrics, summarize_absolute


def _result(true_direction: str, inferred_direction: str) -> RoleInferenceResult:
    return RoleInferenceResult(
        item_id="i",
        source_id="i",
        is_generated=True,
        true_direction=true_direction,
        model="qwen2.5:3b",
        inferred_direction=inferred_direction,
        evidence="x",
        from_cache=False,
    )


def test_summarize_absolute_perfect_agreement() -> None:
    results = [_result(d, d) for d in ("up", "lateral", "down") for _ in range(10)]
    metrics = summarize_absolute(results)
    assert metrics.accuracy == 1.0
    assert metrics.kappa == 1.0
    assert metrics.n == 30


def test_summarize_absolute_chance_agreement_has_kappa_near_zero() -> None:
    import itertools

    labels = ["up", "lateral", "down"]
    # Every true label paired with every inferred label equally often --
    # by construction, no better than chance.
    results = [
        _result(true, inferred)
        for true, inferred in itertools.product(labels, labels)
        for _ in range(10)
    ]
    metrics = summarize_absolute(results)
    assert abs(metrics.kappa) < 0.05


def test_summarize_absolute_confusion_matrix_shape() -> None:
    results = [_result("up", "up"), _result("up", "lateral"), _result("down", "down")]
    metrics = summarize_absolute(results)
    assert metrics.confusion["up"]["up"] == 1
    assert metrics.confusion["up"]["lateral"] == 1
    assert metrics.confusion["down"]["down"] == 1


def test_summarize_absolute_confidence_interval_widens_with_fewer_items() -> None:
    small = summarize_absolute([_result("up", "up")] * 5 + [_result("up", "down")] * 5)
    large = summarize_absolute([_result("up", "up")] * 50 + [_result("up", "down")] * 50)
    assert (small.ci_high - small.ci_low) > (large.ci_high - large.ci_low)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_role_inference.py -v -k summarize`
Expected: FAIL — `summarize_absolute` doesn't exist yet.

- [ ] **Step 3: Implement**

Append to `src/thesis/analysis/role_inference.py`:

```python
from pathlib import Path

from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from statsmodels.stats.proportion import proportion_confint

from thesis.analysis.plots import plot_effect_intervals


@dataclass(frozen=True, slots=True)
class AbsoluteMetrics:
    """Accuracy, its 95% Wilson interval, Cohen's kappa against chance, and
    the full 3x3 confusion matrix -- the matrix matters on its own: a model
    that separates 'down' cleanly while confusing 'up' with 'lateral' is a
    different finding than uniform chance-level guessing, and a bare
    accuracy number would report both the same way."""

    n: int
    accuracy: float
    ci_low: float
    ci_high: float
    kappa: float
    confusion: dict[str, dict[str, int]]


def summarize_absolute(results: list[RoleInferenceResult]) -> AbsoluteMetrics:
    true = [r.true_direction for r in results]
    predicted = [r.inferred_direction for r in results]
    n_correct = sum(1 for t, p in zip(true, predicted, strict=True) if t == p)
    ci_low, ci_high = proportion_confint(n_correct, len(results), method="wilson")
    matrix = confusion_matrix(true, predicted, labels=list(DIRECTIONS))
    confusion = {
        actual: {predicted_label: int(matrix[i, j]) for j, predicted_label in enumerate(DIRECTIONS)}
        for i, actual in enumerate(DIRECTIONS)
    }
    return AbsoluteMetrics(
        n=len(results),
        accuracy=round(float(balanced_accuracy_score(true, predicted)), 4),
        ci_low=round(float(ci_low), 4),
        ci_high=round(float(ci_high), 4),
        kappa=round(float(cohen_kappa_score(true, predicted, labels=list(DIRECTIONS))), 4),
        confusion=confusion,
    )


def plot_accuracy_vs_real(metrics_by_label: dict[str, AbsoluteMetrics], path: Path) -> Path:
    """One row per label (model or "real email"), an accuracy point with
    its 95% interval, against the chance line at 1/3. The first key in
    ``metrics_by_label`` is drawn in the reference color -- pass real
    email first, the same convention :func:`plot_effect_intervals` already
    documents for the Q1 grid-vs-real comparison."""
    labels = list(metrics_by_label)
    return plot_effect_intervals(
        labels,
        [metrics_by_label[label].accuracy for label in labels],
        [metrics_by_label[label].ci_low for label in labels],
        [metrics_by_label[label].ci_high for label in labels],
        path,
        title="Can a blind judge tell who a reply was written to?",
        subtitle="Balanced accuracy, 3-way choice, chance = 0.33 (dashed line to add manually if useful)",
        x_label="balanced accuracy",
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_role_inference.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Quality gates and commit**

```bash
black src/thesis/analysis/role_inference.py tests/test_role_inference.py
ruff check src/thesis/analysis/role_inference.py tests/test_role_inference.py
mypy src/thesis/analysis/role_inference.py tests/test_role_inference.py
python -m pytest -q
git add src/thesis/analysis/role_inference.py tests/test_role_inference.py
git commit -m "Q1 redesign: role_inference metrics and the accuracy figure

Balanced accuracy, a Wilson 95% interval, Cohen's kappa against chance,
and the full 3x3 confusion matrix -- reported separately because a model
that separates 'down' cleanly while confusing 'up'/'lateral' is a
different finding than uniform chance guessing."
git push
```

---

### Task 9: Positive control — can the harness detect a signal when one exists?

Standing in for the spec's originally proposed "shuffle the labels" check: shuffling labels *after* scoring measures nothing, because the judge's blind answer never depended on the label in the first place — the label only enters when computing accuracy, so a shuffle only re-tests that `summarize_absolute` correctly computes chance-level accuracy on random pairings, which is already covered by Task 8's chance-agreement test. The check that actually matters is the opposite direction: deliberately hand the judge the direction-framing sentence the generating model saw (never done in the real run), and confirm accuracy jumps close to 1.0. If it doesn't, the schema, the parsing, or the scoring loop is broken, and every blind number upstream is untrustworthy for a reason that has nothing to do with the model's real ability.

**Files:**
- Modify: `src/thesis/analysis/role_inference.py`
- Test: `tests/test_role_inference.py`

**Interfaces:**
- Consumes: `thesis.sim.scenario._DIRECTION_FRAMING`.
- Produces: `build_positive_control_items(frame: pd.DataFrame, *, replicate: int = 1) -> tuple[list[JudgeItem], dict[str, str]]` — identical to `build_absolute_items_from_grid` except the direction-framing sentence is prepended to the (still identity-stripped) body rather than omitted.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_role_inference.py`:

```python
from thesis.analysis.role_inference import build_positive_control_items


def test_positive_control_items_contain_the_framing_sentence() -> None:
    """The inverse of test_no_direction_framing_language_reaches_the_rendered_prompt:
    here the framing sentence must be present, since this is the sanity
    check that the harness can detect the label when it is actually there."""
    items, true_directions = build_positive_control_items(_tiny_grid())
    for item in items:
        true_direction = true_directions[item.item_id]
        assert _DIRECTION_FRAMING[true_direction].lower() in item.text.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_role_inference.py -v -k positive_control`
Expected: FAIL — `build_positive_control_items` doesn't exist yet.

- [ ] **Step 3: Implement**

Append to `src/thesis/analysis/role_inference.py`:

```python
def build_positive_control_items(
    frame: pd.DataFrame, *, replicate: int = 1
) -> tuple[list[JudgeItem], dict[str, str]]:
    """Same as :func:`build_absolute_items_from_grid`, except the true
    direction-framing sentence is prepended to the reply text. Used only to
    verify the scoring harness can detect a signal when the label is
    actually present -- never part of the blind run itself. If accuracy
    here is not close to 1.0, the schema, parsing or scoring loop is
    broken, independent of whether the model shows the real effect.
    """
    subset = frame[frame["replicate"] == replicate]
    items = [
        JudgeItem(
            item_id=str(row.cell_id),
            text=f"{_DIRECTION_FRAMING[row.direction]} {strip_identity(str(row.body))}",
            is_generated=True,
            source_id=str(row.cell_id),
        )
        for row in subset.itertuples(index=False)
    ]
    true_directions = {str(row.cell_id): str(row.direction) for row in subset.itertuples(index=False)}
    return items, true_directions
```

Import `_DIRECTION_FRAMING` from `thesis.sim.scenario` alongside the existing `DIRECTIONS` import at the top of the module.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_role_inference.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Quality gates and commit**

```bash
black src/thesis/analysis/role_inference.py tests/test_role_inference.py
ruff check src/thesis/analysis/role_inference.py tests/test_role_inference.py
mypy src/thesis/analysis/role_inference.py tests/test_role_inference.py
python -m pytest -q
git add src/thesis/analysis/role_inference.py tests/test_role_inference.py
git commit -m "Q1 redesign: role_inference positive-control harness check

Replaces the spec's originally proposed label-shuffle check, which
would only re-verify that summarize_absolute computes chance accuracy
on random pairings -- already covered by the chance-agreement test in
the metrics task. This checks the opposite and more useful direction:
with the true direction-framing sentence deliberately included,
accuracy must jump close to 1.0, or the harness itself (schema,
parsing, scoring) is broken independent of any model's real ability.

Noted as a deviation from docs/superpowers/specs/2026-09-23-q1-authority-judge-design.md
section 5 check 2, for the same reason given here."
git push
```

---

### Task 10: Judge self-consistency (validation check 1)

Score a 300-item subsample three times and report Fleiss' kappa via the now-public `decision_stability_n`. Stop condition: kappa below about 0.4 means the judge is not usable, and this is found on 900 calls rather than the full run's several thousand.

**Files:**
- Modify: `src/thesis/analysis/role_inference.py`
- Test: `tests/test_role_inference.py`

**Interfaces:**
- Consumes: `thesis.analysis.draw_stability.decision_stability_n`, `thesis.analysis.draw_stability.DecisionStability`.
- Produces: `judge_self_consistency(items: list[JudgeItem], client, *, model, cache, ledger, run_id) -> DecisionStability` — scores the same items three times (`replicate=1,2,3`) and reports agreement across the three passes.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_role_inference.py`:

```python
from thesis.analysis.role_inference import judge_self_consistency


def test_judge_self_consistency_is_perfect_when_every_pass_agrees() -> None:
    item = _item()
    payload = {"evidence": "no directive language", "inferred_direction": "lateral"}
    client = _ScriptedClient([_response(payload), _response(payload), _response(payload)])
    cache = ResponseCache(":memory:")  # see Task 5 Step 4 note on the real fixture
    ledger = CostLedger(":memory:")
    result = judge_self_consistency(
        [item], client, model="qwen2.5:3b", cache=cache, ledger=ledger, run_id="test-run"
    )
    assert result.kappa == 1.0
    assert result.share_agree == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_role_inference.py -v -k self_consistency`
Expected: FAIL — `judge_self_consistency` doesn't exist yet.

- [ ] **Step 3: Implement**

Append to `src/thesis/analysis/role_inference.py`:

```python
import pandas as pd

from thesis.analysis.draw_stability import DecisionStability, decision_stability_n


def judge_self_consistency(
    items: list[JudgeItem],
    client: _CompletionClient,
    *,
    model: str,
    cache: ResponseCache,
    ledger: CostLedger,
    run_id: str,
    n_passes: int = 3,
) -> DecisionStability:
    """Score the same items ``n_passes`` times (each an independent draw,
    via ``replicate``) and report agreement with :func:`decision_stability_n`
    -- the same Fleiss-kappa generalization :mod:`draw_stability` already
    uses for draw-to-draw agreement, reused here because 'how much do
    repeated observations of the same item agree' is the identical
    question whether the repeated observations are generation draws or
    judge passes.

    Stop condition (checked by the caller, not enforced here): a kappa
    below about 0.4 means the judge is not reliable enough to trust for
    the full run.
    """
    passes: list[pd.Series] = []
    for replicate in range(1, n_passes + 1):
        results, _ = run_role_inference_absolute(
            items,
            client,
            model=model,
            cache=cache,
            ledger=ledger,
            run_id=f"{run_id}-pass{replicate}",
            replicate=replicate,
        )
        by_item = {r.item_id: r.inferred_direction for r in results}
        passes.append(pd.Series([by_item[item.item_id] for item in items], name=f"pass_{replicate}"))
    return decision_stability_n(passes)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_role_inference.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Quality gates and commit**

```bash
black src/thesis/analysis/role_inference.py tests/test_role_inference.py
ruff check src/thesis/analysis/role_inference.py tests/test_role_inference.py
mypy src/thesis/analysis/role_inference.py tests/test_role_inference.py
python -m pytest -q
git add src/thesis/analysis/role_inference.py tests/test_role_inference.py
git commit -m "Q1 redesign: role_inference judge self-consistency check

Scores the same items three times and reports Fleiss' kappa via the
now-public decision_stability_n. Validation check 1 of
docs/superpowers/specs/2026-09-23-q1-authority-judge-design.md section 5:
run on a 300-item subsample before the full judge run, stop if kappa
is below about 0.4."
git push
```

---

### Task 11: `main()` — the full run, manifest and figure

Wires every prior task together: load the four grids plus real email, run the absolute-form judge on all of them (draw 1 only), run the paired-form judge on the simulator, run the self-consistency and positive-control checks first as a gate, write the manifest and figure.

**Files:**
- Modify: `src/thesis/analysis/role_inference.py`
- Test: `tests/test_role_inference.py`

**Interfaces:**
- Consumes: `thesis.analysis.q1_real.load_bodies`, `thesis.paths.{INTERIM_DIR, CACHE_DIR, COST_LEDGER, MANIFESTS_DIR, DOCS_FIGURES_DIR, ensure_dirs}`, `thesis.llm.ollama_client.{OllamaClient, OllamaUnavailableError}`.
- Produces: `GENERATING_GRIDS: dict[str, Path]`, `main() -> None`, entry point `python -m thesis.analysis.role_inference`.

- [ ] **Step 1: Write a failing integration-shaped test for the manifest builder**

Add to `tests/test_role_inference.py`:

```python
from thesis.analysis.role_inference import build_manifest


def test_build_manifest_orders_real_email_first() -> None:
    """plot_accuracy_vs_real's docstring convention: the first key drawn
    is the reference row. The manifest's own ordering must put real email
    first so a caller building the figure from it gets that for free."""
    metrics_by_label = {
        "llama3.2:3b": AbsoluteMetrics(10, 0.4, 0.2, 0.6, 0.1, {}),
        "real email": AbsoluteMetrics(20, 0.6, 0.4, 0.8, 0.3, {}),
    }
    manifest = build_manifest(metrics_by_label, self_consistency_kappa=0.5, positive_control_accuracy=0.95)
    assert list(manifest["accuracy_by_label"]) == ["real email", "llama3.2:3b"]
    assert manifest["self_consistency_kappa"] == 0.5
    assert manifest["positive_control_accuracy"] == 0.95
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_role_inference.py -v -k build_manifest`
Expected: FAIL — `build_manifest` doesn't exist yet.

- [ ] **Step 3: Implement `build_manifest` and `main()`**

Append to `src/thesis/analysis/role_inference.py`:

```python
import json

from thesis.analysis.q1_real import load_bodies
from thesis.llm.ollama_client import OllamaClient, OllamaUnavailableError
from thesis.paths import CACHE_DIR, COST_LEDGER, DOCS_FIGURES_DIR, INTERIM_DIR, MANIFESTS_DIR, ensure_dirs

MANIFEST_PATH: Path = MANIFESTS_DIR / "role_inference.json"
FIGURE_PATH: Path = DOCS_FIGURES_DIR / "role_inference_accuracy.png"

GENERATING_GRIDS: Final[dict[str, Path]] = {
    "llama3.2:3b": INTERIM_DIR / "q1_direction_grid_full_3draws.parquet",
    "deepseek-v4-flash": INTERIM_DIR / "q1_direction_grid_deepseek_full.parquet",
    "gpt-oss-20b": INTERIM_DIR / "q1_direction_grid_gpt_oss_20b_full.parquet",
    "gpt-oss-120b": INTERIM_DIR / "q1_direction_grid_gpt_oss_120b_full.parquet",
}


def build_manifest(
    metrics_by_label: dict[str, AbsoluteMetrics],
    *,
    self_consistency_kappa: float,
    positive_control_accuracy: float,
) -> dict[str, Any]:
    """Real email first, so a caller drawing the figure from this manifest
    gets :func:`plot_accuracy_vs_real`'s reference-row convention for free.
    """
    ordered = {"real email": metrics_by_label["real email"]} | {
        label: metrics for label, metrics in metrics_by_label.items() if label != "real email"
    }
    return {
        "accuracy_by_label": {
            label: {
                "n": m.n,
                "accuracy": m.accuracy,
                "ci_low": m.ci_low,
                "ci_high": m.ci_high,
                "kappa": m.kappa,
                "confusion": m.confusion,
            }
            for label, m in ordered.items()
        },
        "self_consistency_kappa": self_consistency_kappa,
        "positive_control_accuracy": positive_control_accuracy,
    }


def main() -> None:
    import argparse

    from thesis.analysis.q1 import parse_replies
    from thesis.llm.cache import ResponseCache
    from thesis.llm.cost import CostLedger
    from thesis.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Skip the self-consistency and positive-control checks. For re-runs "
        "once the gate has already passed once; never for a first run.",
    )
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    client = OllamaClient(args.judge_model)
    if not client.is_available():
        msg = f"no Ollama server reachable; start it with 'ollama serve' and pull {args.judge_model}"
        raise OllamaUnavailableError(msg)

    cache = ResponseCache(CACHE_DIR)
    ledger = CostLedger(COST_LEDGER)
    run_id = "role_inference"

    llama_frame = pd.read_parquet(GENERATING_GRIDS["llama3.2:3b"])

    if not args.skip_gate:
        gate_items, _ = build_absolute_items_from_grid(llama_frame.sample(n=300, random_state=1))
        consistency = judge_self_consistency(
            gate_items, client, model=args.judge_model, cache=cache, ledger=ledger, run_id=run_id
        )
        log.info("self-consistency kappa: %.3f (stop below ~0.4)", consistency.kappa)
        if consistency.kappa < 0.4:
            log.error("self-consistency below 0.4; stopping before the full run")
            return

        control_items, control_truth = build_positive_control_items(
            llama_frame.sample(n=50, random_state=1)
        )
        control_results, _ = run_role_inference_absolute(
            control_items, client, model=args.judge_model, cache=cache, ledger=ledger,
            run_id=run_id, true_directions=control_truth,
        )
        control_metrics = summarize_absolute(control_results)
        log.info("positive-control accuracy: %.3f (expect close to 1.0)", control_metrics.accuracy)
        if control_metrics.accuracy < 0.8:
            log.error("positive-control accuracy below 0.8; the harness itself looks broken")
            return
    else:
        consistency = DecisionStability(n=0, n_agree=0, share_agree=0.0, share_expected=0.0, kappa=0.0, counts={})
        control_metrics = AbsoluteMetrics(0, 0.0, 0.0, 0.0, 0.0, {})

    metrics_by_label: dict[str, AbsoluteMetrics] = {}
    for label, path in GENERATING_GRIDS.items():
        frame = pd.read_parquet(path) if path != GENERATING_GRIDS["llama3.2:3b"] else llama_frame
        items, true_directions = build_absolute_items_from_grid(frame)
        results, summary = run_role_inference_absolute(
            items, client, model=args.judge_model, cache=cache, ledger=ledger,
            run_id=run_id, true_directions=true_directions,
        )
        log.info("%s: %d scored, %d invalid, %d from cache", label, summary.n_scored, summary.n_invalid, summary.n_from_cache)
        metrics_by_label[label] = summarize_absolute(results)

    real_bodies = load_bodies(pd.read_parquet(INTERIM_DIR / "q1_real_emails.parquet")["message_uid"])
    real_directions = pd.read_parquet(INTERIM_DIR / "q1_real_emails.parquet")[["message_uid", "direction"]]
    real_items, real_truth = build_absolute_items_from_real_email(real_bodies, real_directions)
    real_results, real_summary = run_role_inference_absolute(
        real_items, client, model=args.judge_model, cache=cache, ledger=ledger,
        run_id=run_id, true_directions=real_truth,
    )
    log.info("real email: %d scored, %d invalid, %d from cache", real_summary.n_scored, real_summary.n_invalid, real_summary.n_from_cache)
    metrics_by_label["real email"] = summarize_absolute(real_results)

    manifest = build_manifest(
        metrics_by_label,
        self_consistency_kappa=consistency.kappa,
        positive_control_accuracy=control_metrics.accuracy,
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    figure = plot_accuracy_vs_real(manifest_metrics := {"real email": metrics_by_label["real email"], **{k: v for k, v in metrics_by_label.items() if k != "real email"}}, FIGURE_PATH)
    log.info("wrote %s and %s", MANIFEST_PATH, figure)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_role_inference.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Quality gates and commit**

```bash
black src/thesis/analysis/role_inference.py tests/test_role_inference.py
ruff check src/thesis/analysis/role_inference.py tests/test_role_inference.py
mypy src/thesis/analysis/role_inference.py tests/test_role_inference.py
python -m pytest -q
git add src/thesis/analysis/role_inference.py tests/test_role_inference.py
git commit -m "Q1 redesign: role_inference.py main() wires the full Layer-1 run

Self-consistency and positive-control checks run as a gate before the
full absolute-form pass over four models and real email, plus the
paired-form pass on the simulator. Writes
outputs/manifests/role_inference.json and
docs/figures/role_inference_accuracy.png.

python -m thesis.analysis.role_inference"
git push
```

---

### Task 12: `role_coding.py` — human-coding harness for validation check 3

Builds the tool for check 3 (human agreement); the coding session itself is manual, off-plan work, run after this lands. A stratified sample of 150–200 replies across model x direction, and a self-contained HTML page asking the same blind question the judge answers: who was this written to?

**Files:**
- Create: `src/thesis/analysis/role_coding.py`
- Test: `tests/test_role_coding.py`

**Interfaces:**
- Consumes: `thesis.analysis.blinding.strip_identity`; `thesis.analysis.role_inference.GENERATING_GRIDS`.
- Produces: `draw_stratified_sample(grids: dict[str, Path], *, n: int = 180, seed: int = 20260923) -> pd.DataFrame` (columns: `item`, `model`, `true_direction`, `text`), `render_coding_page(sample: pd.DataFrame) -> str`, `main()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_role_coding.py
from __future__ import annotations

import pandas as pd

from thesis.analysis.role_coding import draw_stratified_sample, render_coding_page


def _fake_grid(direction: str, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": [f"{direction}_{i}" for i in range(n)],
            "direction": [direction] * n,
            "replicate": [1] * n,
            "body": [f"reply {i}" for i in range(n)],
        }
    )


def test_draw_stratified_sample_covers_every_direction(tmp_path) -> None:
    frame = pd.concat([_fake_grid(d, 30) for d in ("up", "lateral", "down")], ignore_index=True)
    path = tmp_path / "grid.parquet"
    frame.to_parquet(path)
    sample = draw_stratified_sample({"model_a": path}, n=9, seed=1)
    assert len(sample) == 9
    assert set(sample["true_direction"]) == {"up", "lateral", "down"}


def test_draw_stratified_sample_is_reproducible(tmp_path) -> None:
    frame = pd.concat([_fake_grid(d, 30) for d in ("up", "lateral", "down")], ignore_index=True)
    path = tmp_path / "grid.parquet"
    frame.to_parquet(path)
    first = draw_stratified_sample({"model_a": path}, n=9, seed=1)
    second = draw_stratified_sample({"model_a": path}, n=9, seed=1)
    assert first["item"].tolist() == second["item"].tolist()


def test_render_coding_page_hides_the_true_direction() -> None:
    sample = pd.DataFrame(
        {"item": [1], "model": ["model_a"], "true_direction": ["down"], "text": ["Send it now."]}
    )
    page = render_coding_page(sample)
    assert "Send it now." in page
    assert "down" not in page.lower().replace("dropdown", "").replace("markdown", "")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_role_coding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'thesis.analysis.role_coding'`.

- [ ] **Step 3: Implement**

```python
# src/thesis/analysis/role_coding.py
"""A blind human-coding page for validation check 3 of the Q1 role-inference
redesign: can a person tell, from a reply alone, who it was written to?

This is the true ceiling the judge is measured against, not a secondary
check. If a person cannot tell direction from a reply, no judge can, and
the honest finding is that the replies don't encode direction -- a result
that would not depend on any model or judge choice.

The page follows blind_review.py's self-contained-HTML pattern (no server,
codes kept in the browser, exported as CSV), but asks a different question
over different data: one reply, not two, and a three-way "who was this
written to" choice rather than a failure-mode picklist. That difference in
shape is why this is its own module rather than an extension of
blind_review.py, whose page compares two models' replies for a different
purpose (Q2 mirroring review, not Q1 direction).

Run with ``python -m thesis.analysis.role_coding``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.analysis.blinding import strip_identity
from thesis.analysis.role_inference import GENERATING_GRIDS
from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import TABLES_DIR, ensure_dirs

log = get_logger(__name__)

SAMPLE_SIZE = 180
SEED = 20260923
CHOICES: tuple[str, ...] = ("up", "lateral", "down")
CHOICE_TEXT: dict[str, str] = {
    "up": "written to someone more senior than the writer",
    "lateral": "written to a peer",
    "down": "written to someone more junior than the writer",
}


def draw_stratified_sample(
    grids: dict[str, Path], *, n: int = SAMPLE_SIZE, seed: int = SEED
) -> pd.DataFrame:
    """``n`` replies at replicate 1, stratified by model and true direction
    as evenly as ``n`` allows, identity-stripped. Frozen by the seed, the
    same reason ``review_pack.draw_sample`` freezes its own sample -- codes
    from different sessions must be pooled against the same items.
    """
    frames = []
    for model, path in grids.items():
        frame = pd.read_parquet(path)
        frame = frame[frame["replicate"] == 1].assign(model=model)
        frames.append(frame[["cell_id", "direction", "body", "model"]])
    pool = pd.concat(frames, ignore_index=True)

    per_stratum = max(1, n // (pool["model"].nunique() * len(CHOICES)))
    parts = [
        group.sample(n=min(per_stratum, len(group)), random_state=seed)
        for _, group in pool.groupby(["model", "direction"])
    ]
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True).head(n)
    return pd.DataFrame(
        {
            "item": range(1, len(sample) + 1),
            "model": sample["model"],
            "true_direction": sample["direction"],
            "text": sample["body"].map(strip_identity),
        }
    )


def render_coding_page(sample: pd.DataFrame, *, title: str = "Q1 role-inference coding") -> str:
    """A self-contained HTML page. Model and true_direction are never
    embedded in the page data -- only ``item`` and ``text`` are, so the
    rendered page cannot leak the answer key any more than the judge's own
    prompt does."""
    items = [
        {"item": int(row.item), "text": str(row.text)}
        for row in sample.itertuples(index=False)
    ]
    payload = json.dumps({"items": items, "choices": list(CHOICES), "choice_text": CHOICE_TEXT}).replace(
        "</", "<\\/"
    )
    storage_key = "role-coding-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return (
        _PAGE.replace("__TITLE__", html.escape(title))
        .replace("__STORAGE_KEY__", storage_key)
        .replace("__DATA__", payload)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default=str(TABLES_DIR))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    sample = draw_stratified_sample(GENERATING_GRIDS, n=args.n, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_path = out_dir / "role_coding_page.html"
    key_path = out_dir / "role_coding_key.csv"
    page_path.write_text(render_coding_page(sample), encoding="utf-8")
    sample[["item", "model", "true_direction"]].to_csv(key_path, index=False)
    log.info("wrote %s (%d items) and the key %s", page_path, len(sample), key_path)


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --ink:#1f2328; --muted:#59636e; --line:#d1d9e0; --bg:#f6f8fa; --card:#fff; --done:#1a7f37; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
  header { position: sticky; top: 0; z-index: 1; background: var(--card); border-bottom: 1px solid var(--line);
           padding: 10px 20px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  header h1 { font-size: 16px; margin: 0 12px 0 0; }
  main { max-width: 760px; margin: 0 auto; padding: 16px 20px 60px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; }
  .text { white-space: pre-wrap; overflow-wrap: anywhere; background: var(--bg); border-radius: 6px; padding: 10px 12px; }
  .muted { color: var(--muted); }
  .done { color: var(--done); font-weight: 600; }
  fieldset { border: 1px solid var(--line); border-radius: 6px; margin: 10px 0; padding: 6px 12px 8px; }
  .choice { display: block; padding: 4px 0; cursor: pointer; }
  button, select, input[type=text] { font: inherit; padding: 5px 10px; }
  nav { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
</style>
</head>
<body>
<header>
  <h1>Who was this written to?</h1>
  <label>Your name <input type="text" id="coder" size="14"></label>
  <span id="progress" class="muted"></span>
  <button id="export">Export CSV</button>
</header>
<main>
  <details class="card" open>
    <summary><strong>How to code</strong></summary>
    <ol>
      <li>Read the reply. It is one side of a workplace email exchange; you are not shown what it answers.</li>
      <li>Guess who it was written to: someone more senior than the writer, a peer, or someone more junior.</li>
      <li>Judge only from how it is written. There is no right answer visible anywhere on this page.</li>
      <li>Your codes are saved in this browser as you go. Press Export CSV when you finish.</li>
    </ol>
  </details>
  <nav class="card">
    <button id="prev">&larr; Previous</button>
    <select id="jump"></select>
    <button id="next">Next &rarr;</button>
  </nav>
  <section class="card" id="item"></section>
</main>
<script>
const DATA = __DATA__;
const STORAGE_KEY = "__STORAGE_KEY__";
let codes = {};
try { codes = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (e) { codes = {}; }
let current = 0;

function save() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(codes)); } catch (e) { /* private window etc. */ }
}

function render() {
  const item = DATA.items[current];
  const chosen = codes[item.item] || {};
  const choices = DATA.choices.map(c => `
    <label class="choice">
      <input type="radio" name="answer" value="${c}" ${chosen.answer === c ? "checked" : ""}>
      ${DATA.choice_text[c]}
    </label>`).join("");
  document.getElementById("item").innerHTML = `
    <p class="muted">Item ${item.item} of ${DATA.items.length}</p>
    <div class="text">${item.text.replace(/</g, "&lt;")}</div>
    <fieldset><legend>Who was this written to?</legend>${choices}</fieldset>
  `;
  document.querySelectorAll('input[name="answer"]').forEach(el => {
    el.addEventListener("change", () => {
      codes[item.item] = { answer: el.value, coder: document.getElementById("coder").value };
      save();
      updateProgress();
    });
  });
  const jump = document.getElementById("jump");
  jump.innerHTML = DATA.items.map((it, i) => `<option value="${i}" ${i === current ? "selected" : ""}>${it.item}${codes[it.item] ? " done" : ""}</option>`).join("");
  updateProgress();
}

function updateProgress() {
  const done = Object.keys(codes).length;
  document.getElementById("progress").textContent = `${done} / ${DATA.items.length} coded`;
}

document.getElementById("prev").addEventListener("click", () => { current = Math.max(0, current - 1); render(); });
document.getElementById("next").addEventListener("click", () => { current = Math.min(DATA.items.length - 1, current + 1); render(); });
document.getElementById("jump").addEventListener("change", (e) => { current = Number(e.target.value); render(); });
document.getElementById("export").addEventListener("click", () => {
  const rows = [["item", "answer", "coder"]];
  for (const item of DATA.items) {
    const c = codes[item.item] || {};
    rows.push([item.item, c.answer || "", c.coder || ""]);
  }
  const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "role_coding_codes.csv";
  a.click();
});

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_role_coding.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Quality gates and commit**

```bash
black src/thesis/analysis/role_coding.py tests/test_role_coding.py
ruff check src/thesis/analysis/role_coding.py tests/test_role_coding.py
mypy src/thesis/analysis/role_coding.py tests/test_role_coding.py
python -m pytest -q
git add src/thesis/analysis/role_coding.py tests/test_role_coding.py
git commit -m "Q1 redesign: role_coding.py, the human-coding harness for check 3

Stratified sample (model x true direction) plus a self-contained HTML
page asking the same blind question the judge answers. Building this
is in scope for this plan; the coding session itself is manual, off-plan
work that runs after this lands.

python -m thesis.analysis.role_coding"
git push
```

---

### Task 13: Run the full pipeline for real and write up the result

Everything above has been exercised only against scripted fake clients. This task runs it against the real Ollama judge model and the real grids, and records what actually happened — not what the design predicted.

**Files:**
- Modify: `PROGRESS.md`, `PROGRESS_llms.md`, `HANDOVER.md` (local, gitignored)

**Interfaces:** none — this task produces numbers, not code.

- [ ] **Step 1: Confirm the judge model is available**

Run: `ollama list | grep qwen2.5:3b || ollama pull qwen2.5:3b`

- [ ] **Step 2: Run the gated pipeline**

Run: `cd ~/projects/thesis && source .venv/bin/activate && python -m thesis.analysis.role_inference`

Watch the log lines for the self-consistency and positive-control gate results before the full run proceeds. If either gate fails, stop — do not write up a result the gate itself flagged as untrustworthy, and instead treat the failure as the finding (report the failing kappa or accuracy and stop the plan there).

- [ ] **Step 3: Read the manifest and figure**

Run: `cat outputs/manifests/role_inference.json` and open `docs/figures/role_inference_accuracy.png`.

- [ ] **Step 4: Write the PROGRESS.md entry**

Add a new dated, numbered section (following the existing convention: a table of what ran, the real numbers from the manifest — accuracy, its interval, kappa, and the confusion matrix per model against real email — and what the result says about whether authority is recoverable from the text at all). State plainly whether each model's accuracy interval overlaps real email's or falls clearly below it, since that is the headline the whole redesign was built to answer.

- [ ] **Step 5: Write the PROGRESS_llms.md entry**

Same numbers, organized by model, following that file's existing per-model comparison format.

- [ ] **Step 6: Update HANDOVER.md**

Add a short subsection under section 6 recording that Layer 1 is built and what its headline number is, and update section 4 (open work) to move "run Q1 on the larger models" to done and add Layer 2 (the indicator panel) as the next open item, referencing this plan's spec.

- [ ] **Step 7: Commit and push the write-up**

```bash
git add PROGRESS.md PROGRESS_llms.md
git commit -m "Q1 redesign: Layer 1 (blind role inference) results

[fill in: the real headline accuracy/kappa numbers from
outputs/manifests/role_inference.json, and what they say about whether
authority is recoverable from generated replies versus real email]"
git push
```

`HANDOVER.md` is gitignored; it is edited but not committed.

---

## Self-Review Notes

**Spec coverage.** Section 3's Layer 1 (both forms): Tasks 5–9. Section 4 (model widening): Task 2. Section 5 checks 1, 2 (revised, see Task 9's commit message), 4 (resolved by using `qwen2.5:3b`, none of the four generators): Tasks 9, 10, and the `DEFAULT_JUDGE_MODEL` constant in Task 5. Section 5 check 3 (human agreement): Task 12 builds the tool; the coding session is explicitly out of this plan's scope. Section 6's module list: `blinding.py` (Task 4), `role_inference.py` (Tasks 5–11) — `indicators.py` is Layer 2, a follow-on plan, not built here. Section 7's order of work: Tasks 1–13 follow it directly, with the human-agreement check building its tool (Task 12) before the final write-up rather than in the middle, since the write-up needs the automated run's numbers first.

**Deviation from the spec, stated plainly:** Task 9 replaces the spec's "shuffle the direction labels on a subsample and re-run" with a positive-control check, because the shuffle as literally described doesn't test what it claims to — see Task 9's commit message for the full reasoning. Everything else in this plan follows the spec as written.

**Not in this plan:** Layer 2 (the indicator panel — graded directive score, pronoun rate, `hierarchy.py` fits), and the reply-length diagnostic (spec section 8's open item). Both are separate, smaller pieces of work best planned once Layer 1's actual numbers are in hand, since (as the spec itself notes) a null result on Layer 1 would be a different kind of finding than Layer 2 explaining a channel that turned out not to exist.
