"""What each call costs, and an append-only ledger of what has been spent.

Two jobs, both of which exist to keep a student-budget research project from
an avoidable accident:

1. **Project a cost before spending it.** ``max_cost_usd`` in the run config is
   only a guard if something computes the projected cost and refuses to start.
   The projection uses the provider's own token counting, never an
   approximation -- ``tiktoken`` is OpenAI's tokenizer and undercounts Claude
   by 15-20% on prose and considerably more on code, so a budget guard built on
   it would be optimistic in exactly the direction that hurts.

2. **Record what was actually spent.** The ledger is append-only and
   per-call, so the total in the thesis is a sum over recorded rows rather
   than a remembered figure, and an unexpected bill can be attributed to a
   specific run rather than guessed at.

**Prices are deliberately the standard rates, not the promotional ones.**
Claude Sonnet 5 carries introductory pricing that expires 2026-08-31, well
before this project's December analysis runs. Budgeting at the promotional rate
would under-project every cost after that date, so the higher standing rate is
used throughout: a guard that trips slightly early is useful, one that trips
too late is not.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from thesis.llm.base import Usage

# USD per million tokens, at standard (non-promotional) rates.
_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Cache writes cost more than ordinary input; cache reads cost far less. Both
# multipliers apply to the input rate.
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.10

# The Batch API runs at half price. This is what makes the planned ~13,300
# calls affordable, so it is a first-class part of the cost model rather than
# an afterthought.
_BATCH_MULTIPLIER = 0.50


class UnknownModelPriceError(KeyError):
    """Raised when asked to price a model with no recorded rate.

    Guessing a price would defeat the purpose of a budget guard, so an unknown
    model is an error rather than a zero.
    """


def price_per_million(model: str) -> tuple[float, float]:
    """Return ``(input_usd, output_usd)`` per million tokens."""
    try:
        return _PER_MILLION[model]
    except KeyError:
        known = ", ".join(sorted(_PER_MILLION))
        msg = f"no recorded price for {model!r}; known models: {known}"
        raise UnknownModelPriceError(msg) from None


def cost_usd(model: str, usage: Usage, *, batch: bool = False) -> float:
    """Cost of one call in USD.

    Cached input is priced separately from fresh input: treating a cache read
    as full-price input would overstate a heavily-cached run several times
    over, and the simulator is designed around caching a large stable prefix.
    """
    input_rate, output_rate = price_per_million(model)
    per_token_in = input_rate / 1_000_000
    per_token_out = output_rate / 1_000_000

    total = (
        usage.input_tokens * per_token_in
        + usage.cache_creation_input_tokens * per_token_in * _CACHE_WRITE_MULTIPLIER
        + usage.cache_read_input_tokens * per_token_in * _CACHE_READ_MULTIPLIER
        + usage.output_tokens * per_token_out
    )
    if batch:
        total *= _BATCH_MULTIPLIER
    return total


def project_cost_usd(
    model: str,
    *,
    n_calls: int,
    input_tokens_each: int,
    expected_output_tokens_each: int,
    batch: bool = False,
) -> float:
    """Projected cost of a planned run, before any of it is spent.

    Assumes no cache reads -- the conservative case. A run that caches well
    comes in under this figure, which is the right direction for a guard to be
    wrong in.
    """
    usage = Usage(
        input_tokens=input_tokens_each * n_calls,
        output_tokens=expected_output_tokens_each * n_calls,
    )
    return cost_usd(model, usage, batch=batch)


class BudgetExceededError(RuntimeError):
    """Raised when a projected run would exceed the configured ceiling."""


def guard_budget(projected_usd: float, max_cost_usd: float) -> None:
    """Refuse to start a run whose projection exceeds the configured ceiling.

    Called before submission rather than during it: the point is to prevent the
    spend, not to notice it afterwards.
    """
    if projected_usd > max_cost_usd:
        msg = (
            f"projected cost ${projected_usd:.2f} exceeds the configured "
            f"ceiling of ${max_cost_usd:.2f}; raise run.max_cost_usd "
            f"deliberately if this is intended"
        )
        raise BudgetExceededError(msg)


_LEDGER_FIELDS = (
    "timestamp",
    "run_id",
    "provider",
    "model",
    "call_kind",
    "batch",
    "from_cache",
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
    "cost_usd",
)


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One recorded call."""

    run_id: str
    provider: str
    model: str
    call_kind: str
    usage: Usage
    batch: bool = False
    from_cache: bool = False

    @property
    def cost_usd(self) -> float:
        """A cache hit costs nothing -- it never reached the provider."""
        if self.from_cache:
            return 0.0
        return cost_usd(self.model, self.usage, batch=self.batch)


class CostLedger:
    """An append-only CSV of every call.

    CSV rather than a database because the file is meant to be opened,
    inspected, and summed by a human -- including a supervisor who wants to see
    where the budget went without running any of this code.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _ensure_header(self) -> None:
        if self.path.is_file():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(_LEDGER_FIELDS)

    def record(self, entry: LedgerEntry) -> float:
        """Append one call and return its cost."""
        self._ensure_header()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    entry.run_id,
                    entry.provider,
                    entry.model,
                    entry.call_kind,
                    int(entry.batch),
                    int(entry.from_cache),
                    entry.usage.input_tokens,
                    entry.usage.cache_creation_input_tokens,
                    entry.usage.cache_read_input_tokens,
                    entry.usage.output_tokens,
                    f"{entry.cost_usd:.6f}",
                ]
            )
        return entry.cost_usd

    def total_usd(self) -> float:
        """Everything spent so far, across every run."""
        if not self.path.is_file():
            return 0.0
        with self.path.open(newline="", encoding="utf-8") as handle:
            return sum(float(row["cost_usd"]) for row in csv.DictReader(handle))
