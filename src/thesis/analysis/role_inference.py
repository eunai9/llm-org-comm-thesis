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
