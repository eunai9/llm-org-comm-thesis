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

``stakes`` and ``style`` are secondary factors layered on top of the same
task/direction pair, added because hierarchy effects plausibly interact with
both: does deference toward a superior sharpen when something risky is on the
line (``stakes``), and does an explicit tone instruction interact with a
persona's own corpus-derived style rather than simply override it (``style``,
see :data:`Style`). Each factor is fully crossed with the others, so the grid
is ``task_type x direction x stakes x style``.

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
Style = Literal["deferential", "warm", "neutral", "assertive"]

TASK_TYPES: tuple[str, ...] = (
    "request_information",
    "approve_or_decline",
    "report_problem",
    "schedule_coordination",
    "resolve_disagreement",
    "confirm_details",
)

DIRECTIONS: tuple[Direction, ...] = ("up", "lateral", "down")
STAKES: tuple[Stakes, ...] = ("routine", "high")

# An explicit tone instruction layered on top of the persona's own,
# corpus-derived style (its deference_rate, hedge_rate, etc. -- see
# thesis.sim.persona). This is a deliberate second manipulation, not a
# restatement of the persona: "neutral" is the closest match to what a
# persona would produce unprompted, while the other three levels push the
# model away from that natural register in a specific, named direction.
# "Assertive" rather than "aggressive" by design -- real business email in
# this corpus essentially never reaches outright hostility, so an aggressive
# condition would test model steerability into an out-of-distribution
# register rather than a realistic organizational behavior.
STYLES: tuple[Style, ...] = ("deferential", "warm", "neutral", "assertive")


@dataclass(frozen=True, slots=True)
class Scenario:
    """One situation the persona must respond to."""

    scenario_id: str
    task_type: str
    direction: Direction
    stakes: Stakes
    style: Style
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
    "confirm_details": (
        "Someone wants a quick confirmation that a plan already in motion is "
        "still on track -- not a new decision from you.",
        "Just confirming -- we're still moving forward with the numbers from "
        "last week's call, right? Let me know if anything's changed before I "
        "send this out.",
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

_STYLE_FRAMING: dict[Style, str] = {
    "deferential": (
        "Write in a deferential tone: downplay your own authority, soften "
        "requests, and defer to the other person's judgment."
    ),
    "warm": (
        "Write in a warm, friendly tone: personable and cooperative, while "
        "still getting the point across."
    ),
    "neutral": "Write in a neutral, professional tone -- the standard register for this kind of message.",
    "assertive": (
        "Write in an assertive tone: direct and firm, with no hedging, while "
        "remaining professional."
    ),
}


def build_scenarios() -> list[Scenario]:
    """The full scenario set: task type x direction x stakes x style."""
    scenarios = []
    for task_type, direction, stakes, style in product(TASK_TYPES, DIRECTIONS, STAKES, STYLES):
        situation, incoming = _SITUATIONS[task_type]
        scenarios.append(
            Scenario(
                scenario_id=f"{task_type}__{direction}__{stakes}__{style}",
                task_type=task_type,
                direction=direction,
                stakes=stakes,
                style=style,
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
            _STYLE_FRAMING[scenario.style],
            "",
            "## The message you received",
            "",
            scenario.incoming_message,
        ]
    )
