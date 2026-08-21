"""Score a set of items against the rubric, cache-first.

Mirrors :mod:`thesis.sim.run`'s shape deliberately -- cache lookup before any
call, a validated structured response, a per-call cost record -- because the
provenance and cost discipline built for the simulator applies identically
here: a judge call costs money too, and a scored item needs the same
"which code, which commit, which cache entry produced this" trail a
generated email does.

**What does not carry over: a persona/scenario grid.** Judging has no
factorial design of its own to expand -- the design lives in *which items get
selected for judging* (paired real/generated, stratified, etc.), which is a
sampling decision made before this module ever runs. This module's only job
is: given a list of :class:`~thesis.judge.prompt.JudgeItem`, a phrasing
variant, and a client, produce validated scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from thesis.judge.prompt import JudgeItem, Variant, render_item_block, render_rubric_block
from thesis.judge.rubric import build_judge_schema, validate_judge_response
from thesis.llm.base import CompletionRequest, CompletionResponse, Message, Provider
from thesis.llm.cache import ResponseCache, cache_key
from thesis.llm.cost import CostLedger, LedgerEntry, cost_usd
from thesis.llm.stub_client import is_stub_model
from thesis.logging_setup import get_logger

log = get_logger(__name__)


class _CompletionClient(Protocol):
    """The two capabilities scoring actually needs.

    Narrower than LLMClient for the same reason memory_generation.py and
    batch.py each define their own equivalent: stating the real dependency
    means a test double only has to implement what is actually called, and
    the type system correctly admits any client capable of a single
    completion -- including ones (like Ollama) that cannot batch at all.
    """

    provider: Provider

    def complete(self, request: CompletionRequest) -> CompletionResponse: ...


def build_judge_request(
    item: JudgeItem, variant: Variant, model: str, *, replicate: int = 1
) -> CompletionRequest:
    """Assemble one judge call.

    ``variant`` is cached as the system prompt (fixed across every item
    scored with it); the item text is the user turn. ``replicate`` becomes
    the request's ``variant`` field (an unfortunately overloaded word --
    ``CompletionRequest.variant`` is the cache-key draw index, unrelated to
    the *phrasing* variant here) so that, if a self-consistency check needs
    more than one independent score for the same item, each draw gets its own
    cache entry rather than silently reusing the first.
    """
    return CompletionRequest(
        model=model,
        messages=[Message(role="user", content=render_item_block(item))],
        max_tokens=1024,
        system=render_rubric_block(variant),
        output_schema=build_judge_schema(),
        cache_system=True,
        variant=replicate,
        metadata={"item_id": item.item_id, "rubric_variant": variant},
    )


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """One scored item: the rubric scores, plus enough provenance to analyze
    it without re-deriving anything."""

    item_id: str
    source_id: str
    is_generated: bool
    variant: Variant
    model: str
    scores: dict[str, int]
    evidence: dict[str, str]
    from_cache: bool


@dataclass
class ScoringSummary:
    """Run-level counts, for the same honest-reporting reason every other
    pass in this project reports what actually happened rather than assuming
    the request count and the result count matched."""

    n_requested: int = 0
    n_scored: int = 0
    n_invalid: int = 0
    n_from_cache: int = 0
    total_cost_usd: float = 0.0


def score_items(
    items: list[JudgeItem],
    client: _CompletionClient,
    *,
    variant: Variant,
    model: str,
    cache: ResponseCache,
    ledger: CostLedger,
    run_id: str,
) -> tuple[list[JudgeResult], ScoringSummary]:
    """Score every item, serving from cache where possible.

    Invalid responses are recorded and skipped rather than raised, the same
    choice run.py makes for the simulator: one malformed judge response must
    not discard every other score in the batch.
    """
    summary = ScoringSummary(n_requested=len(items))
    results: list[JudgeResult] = []

    for item in items:
        request = build_judge_request(item, variant, model)
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
            payload = validate_judge_response(response.parsed)
        except Exception as exc:
            log.warning("item %s: failed validation: %s", item.item_id, exc)
            summary.n_invalid += 1
            continue

        results.append(
            JudgeResult(
                item_id=item.item_id,
                source_id=item.source_id,
                is_generated=item.is_generated,
                variant=variant,
                model=response.model,
                scores={k: v["score"] for k, v in payload.items()},
                evidence={k: v["evidence"] for k, v in payload.items()},
                from_cache=response.from_cache,
            )
        )
        summary.n_scored += 1

        billable = not response.from_cache and not is_stub_model(response.model)
        cost = cost_usd(response.model, response.usage) if billable else 0.0
        summary.total_cost_usd += cost
        ledger.record(
            LedgerEntry(
                run_id=run_id,
                provider=client.provider,
                model=response.model,
                call_kind="judge",
                usage=response.usage,
                from_cache=response.from_cache or is_stub_model(response.model),
            )
        )

    return results, summary


def results_to_rows(results: list[JudgeResult]) -> list[dict[str, Any]]:
    """Flatten JudgeResult into rows suitable for a Parquet table.

    Kept separate from JudgeResult itself so the dataclass stays a clean
    in-memory shape and the flattening (which columns, which prefix) is a
    decision this one function owns and tests can check directly.
    """
    rows = []
    for result in results:
        row: dict[str, Any] = {
            "item_id": result.item_id,
            "source_id": result.source_id,
            "is_generated": result.is_generated,
            "variant": result.variant,
            "model": result.model,
            "from_cache": result.from_cache,
        }
        for key, score in result.scores.items():
            row[f"score_{key}"] = score
            row[f"evidence_{key}"] = result.evidence[key]
        rows.append(row)
    return rows
