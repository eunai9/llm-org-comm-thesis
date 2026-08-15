"""The scenarios a persona responds to, and the experimental grid.

Q1 asks how hierarchy shapes communication, so the manipulation that matters is
**the rank of the person being written to, relative to the writer.** A Vice
President answering their Managing Director and the same Vice President
answering an analyst is the contrast the research question is about; a design
that only varied the writer's own rank could not see it.

The grid is therefore built around ``direction`` (up / lateral / down) rather
than the counterpart's absolute rank. Absolute rank pairs are unbalanced --
rank 1 has no one below and rank 5 no one above -- and pooling by direction
keeps every persona contributing to every level of the factor, which is what
makes the eventual mixed model estimable.

**Scenarios carry no Enron content.** They are synthetic business situations of
the kind the corpus contains, written from scratch. Real messages appear in the
study only as the separately-sampled ``S_shots`` stimuli, where they are used
as-is rather than paraphrased into a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

Direction = Literal["up", "lateral", "down"]
Stakes = Literal["routine", "high"]

TASK_TYPES: tuple[str, ...] = (
    "request_information",
    "approve_or_decline",
    "report_problem",
    "schedule_coordination",
    "resolve_disagreement",
)

DIRECTIONS: tuple[Direction, ...] = ("up", "lateral", "down")
STAKES: tuple[Stakes, ...] = ("routine", "high")


@dataclass(frozen=True, slots=True)
class Scenario:
    """One situation the persona must respond to."""

    scenario_id: str
    task_type: str
    direction: Direction
    stakes: Stakes
    situation: str
    incoming_message: str


# Situation templates per task type. Each is deliberately generic business
# content -- no Enron specifics, no real names, nothing traceable to the
# corpus.
_SITUATIONS: dict[str, tuple[str, str]] = {
    "request_information": (
        "A colleague needs figures from you to close out a quarterly summary.",
        "Could you send over the volume numbers for last month when you get a "
        "chance? I'm pulling together the quarterly summary.",
    ),
    "approve_or_decline": (
        "Someone is asking you to sign off on a change that carries some risk.",
        "We'd like to move the settlement date forward by a week. It saves us "
        "a cycle, but it does mean less review time. Can you approve?",
    ),
    "report_problem": (
        "You have found an error that someone else needs to know about.",
        "Just checking in on the reconciliation - is everything on track for " "Friday?",
    ),
    "schedule_coordination": (
        "A meeting needs to be arranged across several busy calendars.",
        "We need to get everyone in a room before the deadline. What does your "
        "availability look like next week?",
    ),
    "resolve_disagreement": (
        "Two positions are in conflict and you are being asked to weigh in.",
        "I don't think the approach in the draft holds up. It seems to me the "
        "numbers were run on the wrong basis. How do you want to handle this?",
    ),
}

_STAKES_FRAMING: dict[Stakes, str] = {
    "routine": "This is routine business, with no unusual pressure attached.",
    "high": (
        "This matters: there is real money and visible accountability "
        "attached, and the outcome will be noticed."
    ),
}

_DIRECTION_FRAMING: dict[Direction, str] = {
    "up": "You are writing to someone senior to you.",
    "lateral": "You are writing to a peer at your own level.",
    "down": "You are writing to someone who reports into your part of the organization.",
}


def build_scenarios() -> list[Scenario]:
    """The full scenario set: task type x direction x stakes."""
    scenarios = []
    for task_type, direction, stakes in product(TASK_TYPES, DIRECTIONS, STAKES):
        situation, incoming = _SITUATIONS[task_type]
        scenarios.append(
            Scenario(
                scenario_id=f"{task_type}__{direction}__{stakes}",
                task_type=task_type,
                direction=direction,
                stakes=stakes,
                situation=situation,
                incoming_message=incoming,
            )
        )
    return scenarios


def render_scenario_block(scenario: Scenario) -> str:
    """The scenario section of the prompt, placed after the cache breakpoint."""
    return "\n".join(
        [
            "## Situation",
            "",
            scenario.situation,
            _DIRECTION_FRAMING[scenario.direction],
            _STAKES_FRAMING[scenario.stakes],
            "",
            "## The message you received",
            "",
            scenario.incoming_message,
        ]
    )
