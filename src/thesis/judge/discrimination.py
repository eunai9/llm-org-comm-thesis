"""The discrimination check: can the judge directly tell real from generated?

A genuinely separate measurement from rubric scoring, not a seventh rubric
item bolted onto the same call. Two reasons it has to stay a separate
request:

1. **Contamination risk.** If a single call both scored the six rubric
   dimensions *and* asked "is this real?", the act of guessing origin could
   leak into the rubric scores themselves -- a judge primed to hunt for
   tells of AI authorship might mark down role_consistency or
   corpus_plausibility for reasons that have nothing to do with those
   dimensions on their own terms. Keeping the requests separate keeps each
   measurement clean of the other.
2. **It is answering a different question.** Rubric scoring asks "is this
   good, on its own terms?" -- answerable without ever knowing or guessing
   where the text came from. This asks "can you tell where it came from?"
   directly. Conflating the two would muddy exactly the distinction Q2's
   fidelity claim depends on.

**What "blind" means here is different from the rubric prompt.** The rubric
prompt hides provenance because origin is irrelevant to the questions being
asked. Here the judge's entire task *is* to guess origin -- what must still
be hidden is any other item's label, and of course the item's own true
label, never revealed anywhere the judge can see it.

**The output is a single ordinal rating, not a binary label.** A 1-5
"how confident are you this is real" scale, rather than a forced real/
generated pick, is what makes an AUC computable at all downstream: ROC/AUC
needs a score that ranks confidently-generated below confidently-real, not
just a class label. Computing that AUC is calibration analysis, not this
module's job -- this module's job ends at producing the rating.
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

log = get_logger(__name__)

ORIGIN_MIN = 1
ORIGIN_MAX = 5
ORIGIN_VALUES: Final[tuple[int, ...]] = tuple(range(ORIGIN_MIN, ORIGIN_MAX + 1))

_TASK_FRAMING = (
    "You will be shown a single workplace email. It may be a real email "
    "drawn from a corporate archive, or it may have been written for a "
    "research study designed to resemble one.\n\n"
    "Your task is to judge which is more likely, based only on the writing "
    "itself -- word choice, structure, the small habits of real "
    "correspondence versus writing that is a little too explanatory or a "
    "little too polished.\n\n"
    "Give a short piece of evidence from the text, then a rating from 1 to "
    "5: 1 means you are confident it was written for a research study, 5 "
    "means you are confident it is a real archived email, and 3 means you "
    "genuinely cannot tell."
)

DISCRIMINATION_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "evidence": {
            "type": "string",
            "description": "A short, specific quote or paraphrase from the message that supports your rating.",
        },
        "likely_origin": {
            "type": "integer",
            "enum": list(ORIGIN_VALUES),
            "description": (
                "1 = confident this was written for a research study, "
                "5 = confident this is a real archived email, "
                "3 = cannot tell."
            ),
        },
    },
    "required": ["evidence", "likely_origin"],
    "additionalProperties": False,
}


class InvalidDiscriminationResponseError(ValueError):
    """Raised when a discrimination response does not satisfy its shape."""


def validate_discrimination_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Second line of defence behind the API's own schema enforcement --
    the same role validate_judge_response() plays for rubric scoring."""
    if "likely_origin" not in payload or "evidence" not in payload:
        msg = "discrimination response missing 'likely_origin' or 'evidence'"
        raise InvalidDiscriminationResponseError(msg)
    if payload["likely_origin"] not in ORIGIN_VALUES:
        msg = f"likely_origin {payload['likely_origin']!r} not in {ORIGIN_VALUES}"
        raise InvalidDiscriminationResponseError(msg)
    return payload


def build_discrimination_request(
    item: JudgeItem, model: str, *, replicate: int = 1
) -> CompletionRequest:
    """Assemble one discrimination call.

    The task framing is cached as the system prompt -- fixed across every
    item -- with the message itself as the user turn, the same split
    judge/run.py uses for rubric scoring.
    """
    return CompletionRequest(
        model=model,
        messages=[Message(role="user", content=render_item_block(item))],
        max_tokens=512,
        system=_TASK_FRAMING,
        output_schema=DISCRIMINATION_SCHEMA,
        cache_system=True,
        variant=replicate,
        metadata={"item_id": item.item_id, "task": "discrimination"},
    )


@dataclass(frozen=True, slots=True)
class DiscriminationResult:
    """One item's discrimination rating, alongside its true label.

    The true label (is_generated) is carried here for the calling code's
    analysis -- it was never visible to the judge, which only ever saw
    render_item_block(item), the same blind text the rubric prompt uses.
    """

    item_id: str
    source_id: str
    is_generated: bool
    model: str
    likely_origin: int
    evidence: str
    from_cache: bool


@dataclass
class DiscriminationSummary:
    n_requested: int = 0
    n_scored: int = 0
    n_invalid: int = 0
    n_from_cache: int = 0
    total_cost_usd: float = 0.0


class _CompletionClient(Protocol):
    """The one capability this module needs.

    Narrower than LLMClient for the same reason judge/run.py,
    memory_generation.py, and batch.py each define their own equivalent:
    stating the real dependency means a test double only implements what is
    actually called.
    """

    provider: Provider

    def complete(self, request: CompletionRequest) -> CompletionResponse: ...


def run_discrimination(
    items: list[JudgeItem],
    client: _CompletionClient,
    *,
    model: str,
    cache: ResponseCache,
    ledger: CostLedger,
    run_id: str,
) -> tuple[list[DiscriminationResult], DiscriminationSummary]:
    """Run the discrimination check over every item, cache-first.

    An invalid response is recorded and skipped rather than raised -- one
    malformed rating must not discard every other item's result, the same
    choice judge/run.py's score_items() makes.
    """
    summary = DiscriminationSummary(n_requested=len(items))
    results: list[DiscriminationResult] = []

    for item in items:
        request = build_discrimination_request(item, model)
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
            payload = validate_discrimination_response(response.parsed)
        except InvalidDiscriminationResponseError as exc:
            log.warning("item %s: failed validation: %s", item.item_id, exc)
            summary.n_invalid += 1
            continue

        results.append(
            DiscriminationResult(
                item_id=item.item_id,
                source_id=item.source_id,
                is_generated=item.is_generated,
                model=response.model,
                likely_origin=payload["likely_origin"],
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
                call_kind="discrimination",
                usage=response.usage,
                from_cache=response.from_cache or not billable,
            )
        )

    return results, summary


def results_to_rows(results: list[DiscriminationResult]) -> list[dict[str, Any]]:
    """Flatten to rows suitable for a Parquet table, mirroring
    judge.run.results_to_rows()'s shape for the rubric-scoring results."""
    return [
        {
            "item_id": r.item_id,
            "source_id": r.source_id,
            "is_generated": r.is_generated,
            "model": r.model,
            "likely_origin": r.likely_origin,
            "evidence": r.evidence,
            "from_cache": r.from_cache,
        }
        for r in results
    ]
