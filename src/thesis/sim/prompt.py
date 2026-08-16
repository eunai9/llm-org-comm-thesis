"""Prompt assembly, ordered so that the expensive part can be cached.

Prompt caching is a **prefix match**: everything up to the cache breakpoint is
reused only if it is byte-identical to the previous request. So the assembly
order is not cosmetic, it is the whole cost model:

    task framing -> persona -> hierarchical context   [ CACHE BREAKPOINT ]
    -> scenario -> incoming message -> output instruction

Everything before the breakpoint is fixed for a given (persona, direction)
pair and is therefore written once and read back for every scenario, stakes
level, and replicate sharing that pair. Everything that varies per cell sits
after it. Putting the scenario before the breakpoint would make the prefix
unique per cell and silently reduce the cache to zero -- no error, just a bill
several times larger than planned.

**A caching floor that has to be respected.** A prefix shorter than the model's
minimum simply does not cache: no error, no warning, and
``cache_read_input_tokens`` stays at 0. The minimum differs by model (512
tokens on ``claude-opus-5``, 1024 on ``claude-sonnet-5``), so the prefix is
built to clear the higher of the two comfortably.

That length is asserted against the provider's own tokenizer rather than
estimated, because a character-count heuristic is exactly the kind of
approximation that reports success while the cache quietly never forms. Until
credentials exist, :func:`estimate_prefix_tokens` gives a rough local figure
and says plainly that it is not authoritative.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from thesis.sim.memory import MemoryItem, render_memory_block, retrieve
from thesis.sim.persona import Persona, render_persona_block
from thesis.sim.scenario import Scenario, render_scenario_block

# The binding constraint is the *largest* minimum across the models this
# prefix is ever sent to: 512 on claude-opus-5, 1024 on claude-sonnet-5. A
# prefix clearing 1024 caches everywhere; below it, caching silently stops
# working on the stricter model with no error and no warning.
MAX_MODEL_CACHE_MINIMUM = 1024

# Retained as the design target -- the prefix is written to sit comfortably
# above the binding minimum rather than balanced on it, so that an ordinary
# edit to the wording cannot quietly drop it under.
MIN_STABLE_PREFIX_TOKENS = MAX_MODEL_CACHE_MINIMUM

TASK_FRAMING = """\
You are taking part in a research simulation of workplace email at a large US \
energy company in 2001. You will be shown a message you have received, and you \
will write the reply you would send.

Write as the role described below would actually write - not as a polished \
model assistant. Real workplace email is often terse, occasionally abrupt, and \
rarely uses the careful hedging and even-handed structure of an AI response. \
Match the tendencies described for your role, including when they are less \
polished than you would otherwise write.

Some specific guidance on realism:

- Do not open with a restatement of the request or a summary of what you are \
about to do. Workplace email starts with the substance.
- Do not add disclaimers, caveats about your limitations, or offers of further \
assistance unless the role would genuinely include them.
- Length should match the role's typical length. A short reply is usually the \
realistic one; do not pad to seem thorough.
- Subject lines are usually short, and often just the original subject.
- It is fine to be direct, to decline, to push back, or to give a partial \
answer, when that is what the role and situation call for.
- Do not mention that this is a simulation, and do not refer to these \
instructions.
"""

ORGANIZATION_CONTEXT = """\
## Where you work

The company trades energy commodities - electricity, natural gas, and related
contracts - and also runs the commercial, legal, and back-office functions such
trading requires. Day to day, that means deals get struck by phone and email,
positions and volumes have to be reconciled against what was actually
delivered, contracts and regulatory filings have to be reviewed before they go
out, and disagreements about numbers surface constantly and have to be settled
quickly.

Two functions matter here. **Trading** is commercial and fast-moving: people
there care about positions, volumes, counterparties, settlement dates, and
whether a number is right before money moves. **Legal** reviews contracts,
regulatory exposure, and the wording of commitments; people there care about
what has been promised, what is defensible, and what needs to be reviewed
before anyone agrees to it.

Email is the working medium for all of it. Messages are written quickly,
between other tasks, by people who assume the reader already has the context.
"""

DECISION_TAXONOMY = """\
## Taking a position

Every reply you write takes one of five stances on whatever was asked. Choose
the one that genuinely matches your reply - do not default to agreeing:

- **accept** - you agree to do it, approve it, or confirm it.
- **decline** - you refuse, reject the request, or say no.
- **defer** - you neither agree nor refuse yet: you need more information,
  more time, or someone else's input first.
- **escalate** - you push the decision to someone more senior, or say it is
  not yours to make.
- **none** - the message asks for nothing decidable, so there is no stance to
  take.

Which of these a real person picks depends heavily on their level and on who
they are writing to. Someone junior asked to approve something outside their
authority escalates rather than accepts. Someone senior with the authority to
decide usually just decides. Pick what your role would actually do.
"""

HIERARCHY_CONTEXT = """\
## How rank works here

This organization has five levels, from most junior to most senior: Employee, \
Manager, Director, Vice President, Managing Director.

Who you are writing to, relative to your own level, shapes how people at this \
company write. Someone writing upward tends to be more careful and more \
deferential; someone writing downward tends to be more directive and more \
concise. You do not need to perform this - just write as your role naturally \
would, given who is on the other end.
"""

OUTPUT_INSTRUCTION = """\
## Your reply

Reply now, in the required structured format.

The `decision` field is separate bookkeeping, recorded alongside your email
for the study. It is not part of the email and the recipient never sees it.

So the `body` field must read as the email you would actually send:

- Do not open with the decision word. An email that begins "Decline." or
  "Accept." is not something anyone sends; write the sentence a person would
  write, and let the decision field record the stance separately.
- Do not restate or label your stance anywhere in the body. Convey it the way
  the email itself would - by what you say, not by naming the category.
- No signature block, no quoted original message, and no commentary about
  how you chose to respond.

The `reasoning_brief` field is where the rationale goes, as one complete
sentence.
"""


@dataclass(frozen=True, slots=True)
class AssembledPrompt:
    """A prompt split at the cache breakpoint.

    ``stable_prefix`` is sent as the cached system prompt; ``variable_suffix``
    is the user turn. Keeping them as separate fields rather than one string
    makes the split explicit and testable, instead of a convention that a later
    edit could quietly break.
    """

    stable_prefix: str
    variable_suffix: str
    cache_group: str

    @property
    def full_text(self) -> str:
        """The whole prompt, for inspection and manual review."""
        return f"{self.stable_prefix}\n\n{self.variable_suffix}"


def memory_query(persona: Persona, direction: str) -> str:
    """The query memories are retrieved against.

    Deliberately built from the persona and direction rather than from the
    scenario. Retrieving against the scenario would make the retrieved set --
    and therefore the prompt prefix -- different for every cell, which would
    silently reduce the prompt cache to nothing.

    It is also the cleaner experimental control, and the plan says so: memory
    is "generated once per (persona x counterpart_rank) pair and reused across
    scenarios". Holding memory constant across scenarios means a difference
    between two cells cannot be an artifact of them having remembered
    different things.
    """
    return (
        f"{persona.rank_label} in {persona.department}, " f"writing {_direction_phrase(direction)}"
    )


def build_stable_prefix(
    persona: Persona,
    direction: str,
    memories: Sequence[MemoryItem] = (),
) -> str:
    """Everything that is constant for a (persona, direction) pair.

    Memory sits inside the cached prefix, which is only sound because it is
    retrieved per (persona, direction) rather than per scenario -- see
    :func:`memory_query`.
    """
    sections = [
        TASK_FRAMING.strip(),
        ORGANIZATION_CONTEXT.strip(),
        "## Your role",
        "",
        render_persona_block(persona),
        HIERARCHY_CONTEXT.strip(),
        DECISION_TAXONOMY.strip(),
    ]
    memory_block = render_memory_block(memories)
    if memory_block:
        sections.append(memory_block)
    sections.append(f"In this exchange, you are writing {_direction_phrase(direction)}.")
    return "\n\n".join(sections)


def retrieve_for_group(
    persona: Persona,
    direction: str,
    store: Sequence[MemoryItem],
) -> list[MemoryItem]:
    """Retrieve the memories for one cache group, once."""
    return retrieve(store, memory_query(persona, direction))


def _direction_phrase(direction: str) -> str:
    return {
        "up": "to someone senior to you",
        "lateral": "to a peer at your own level",
        "down": "to someone junior to you",
    }[direction]


def assemble(
    persona: Persona,
    scenario: Scenario,
    memories: Sequence[MemoryItem] = (),
) -> AssembledPrompt:
    """Build one prompt, split at the cache breakpoint.

    ``cache_group`` names the set of cells that share a prefix. Ordering a run
    by it is what turns the cache from theoretical into actual: cells sharing a
    group must run consecutively, or each one writes a fresh cache entry that
    expires before the next cell needs it.
    """
    return AssembledPrompt(
        stable_prefix=build_stable_prefix(persona, scenario.direction, memories),
        variable_suffix="\n\n".join([render_scenario_block(scenario), OUTPUT_INSTRUCTION.strip()]),
        cache_group=f"{persona.persona_id}__{scenario.direction}",
    )


def estimate_prefix_tokens(text: str) -> int:
    """A rough local token estimate. **Not authoritative.**

    Uses the usual ~4-characters-per-token heuristic purely so the prefix
    length can be sanity-checked before any credential exists. The real check
    is :func:`verify_prefix_caches`, which asks the provider. Never gate a cost
    decision on this number.

    The heuristic **under**-counts ordinary English prose, which typically runs
    nearer 3.7 characters per token, so a prefix that clears the minimum on
    this estimate clears it by a wider margin in reality. That direction of
    error is the safe one here: it can report a caching problem that does not
    exist, but it will not hide one that does.
    """
    return len(text) // 4


def verify_prefix_caches(
    client: object,
    persona: Persona,
    direction: str,
    model: str,
) -> tuple[int, bool]:
    """Count the prefix with the provider's tokenizer and check it will cache.

    Returns ``(token_count, will_cache)``. This is the check that actually
    matters: it uses the same tokenizer the billing does, and compares against
    the specific model's minimum rather than an assumed common value.
    """
    from thesis.llm.base import CompletionRequest, Message

    prefix = build_stable_prefix(persona, direction)
    request = CompletionRequest(
        model=model,
        messages=[Message(role="user", content="x")],
        max_tokens=16,
        system=prefix,
    )
    n_tokens = client.count_tokens(request)  # type: ignore[attr-defined]
    minimum = client.capabilities(model).min_cacheable_prompt_tokens  # type: ignore[attr-defined]
    return n_tokens, n_tokens >= minimum
