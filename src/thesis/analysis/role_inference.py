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
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from statsmodels.stats.proportion import proportion_confint

from thesis.analysis.blinding import strip_identity
from thesis.analysis.draw_stability import DecisionStability, decision_stability_n
from thesis.analysis.plots import plot_effect_intervals
from thesis.judge.prompt import JudgeItem, render_item_block
from thesis.llm.base import CompletionRequest, CompletionResponse, Message, Provider
from thesis.llm.cache import ResponseCache, cache_key
from thesis.llm.cost import CostLedger, LedgerEntry, cost_usd
from thesis.llm.ollama_client import is_local_model
from thesis.llm.stub_client import is_stub_model
from thesis.logging_setup import get_logger
from thesis.sim.scenario import _DIRECTION_FRAMING, DIRECTIONS

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
    return f"## Reply A\n\n{item.text_a}\n\n## Reply B\n\n{item.text_b}"


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
    true_directions = {
        str(row.cell_id): str(row.direction) for row in subset.itertuples(index=False)
    }
    return items, true_directions


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
        passes.append(
            pd.Series([by_item[item.item_id] for item in items], name=f"pass_{replicate}")
        )
    return decision_stability_n(passes)


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
    true_directions = {
        str(row.message_uid): str(row.direction) for row in merged.itertuples(index=False)
    }
    return items, true_directions


def build_paired_items_from_grid(
    frame: pd.DataFrame, *, replicate: int = 1, seed: int = 20260923
) -> list[PairedItem]:
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
    confusion: dict[str, dict[str, int]] = {
        str(actual): {
            str(predicted_label): int(matrix[i, j]) for j, predicted_label in enumerate(DIRECTIONS)
        }
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
    true_directions = {
        str(row.cell_id): str(row.direction) for row in subset.itertuples(index=False)
    }
    return items, true_directions
