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

``stakes`` and ``tone`` are secondary factors layered on top of the same
task/direction pair, added because hierarchy effects plausibly interact with
both: does deference toward a superior sharpen when something risky is on the
line (``stakes``), and does the *tone of the message a persona receives*
shape how they write back, on top of who they're writing to (``tone``, see
:data:`Tone`)? Each factor is fully crossed with the others, so the grid is
``task_type x direction x stakes x tone``.

**``tone`` describes the incoming message, not an instruction to the
persona.** Each task type has four hand-written phrasings of the same
underlying request -- deferential, warm, neutral, assertive -- so the
persona is never told how to write; it only ever sees a stimulus that
happens to be phrased in one of those four registers and replies however it
naturally would. This is a deliberate correction from an earlier version of
this module, which instead told the *persona* to write in a given tone --
that tested instruction-following, not whether an incoming message's tone
shapes the reply, which is what Q1 actually wants to know. (An earlier,
also-corrected version of this field was named ``style`` -- renamed to
``tone`` to avoid confusion with :class:`thesis.sim.persona.PersonaStyle`,
a persona's own corpus-derived writing style, which is an unrelated
concept.)

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
Tone = Literal["deferential", "warm", "neutral", "assertive"]

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

# The tone the *incoming* message is written in -- see the module docstring
# for why this is not an instruction to the persona. "Neutral" is each task
# type's plainest, most ordinary phrasing; the other three levels rephrase
# the same request in a different register. "Assertive" rather than
# "aggressive" by design -- real business email in this corpus essentially
# never reaches outright hostility, so an aggressive stimulus would be
# testing the model's reaction to an out-of-distribution register rather
# than a realistic organizational one.
TONES: tuple[Tone, ...] = ("deferential", "warm", "neutral", "assertive")


@dataclass(frozen=True, slots=True)
class Scenario:
    """One situation the persona must respond to."""

    scenario_id: str
    task_type: str
    direction: Direction
    stakes: Stakes
    tone: Tone
    situation: str
    incoming_message: str


# Situation templates per task type. Each is deliberately generic business
# content -- no Enron specifics, no real names, nothing traceable to the
# corpus.
_SITUATIONS: dict[str, str] = {
    "request_information": "A colleague needs figures from you to close out a quarterly summary.",
    "approve_or_decline": "Someone is asking you to sign off on a change that carries some risk.",
    "report_problem": "You have found an error that someone else needs to know about.",
    "schedule_coordination": "A meeting needs to be arranged across several busy calendars.",
    "resolve_disagreement": "Two positions are in conflict and you are being asked to weigh in.",
    "confirm_details": (
        "Someone wants a quick confirmation that a plan already in motion is "
        "still on track -- not a new decision from you."
    ),
}

# Four phrasings of the same request per task type -- same underlying ask,
# different register. This is the actual tone manipulation: the persona
# never receives a tone instruction, only a stimulus already written in one
# of these four voices.
_INCOMING_MESSAGES: dict[tuple[str, Tone], str] = {
    ("request_information", "deferential"): (
        "Sorry to bother you with this -- whenever you get a moment, would "
        "you mind sending over last month's volume numbers? I'm putting "
        "together the quarterly summary and don't want to add to your plate "
        "if this is a bad time."
    ),
    ("request_information", "warm"): (
        "Hope things are going well on your end! Whenever you get a chance, "
        "could you send over last month's volume numbers? Pulling together "
        "the quarterly summary and always appreciate your help with this."
    ),
    ("request_information", "neutral"): (
        "Could you send over the volume numbers for last month when you get "
        "a chance? I'm pulling together the quarterly summary."
    ),
    ("request_information", "assertive"): (
        "Send over last month's volume numbers today. I need them to finish "
        "the quarterly summary and I'm on a deadline."
    ),
    ("approve_or_decline", "deferential"): (
        "I hate to ask, but would it be alright if we moved the settlement "
        "date forward by a week? I know it cuts into review time, and I "
        "completely understand if that's not something you're comfortable "
        "signing off on."
    ),
    ("approve_or_decline", "warm"): (
        "Quick one for you -- we're hoping to move the settlement date "
        "forward by a week. It'd save us a cycle, though it does mean a bit "
        "less review time. Let me know what you think whenever you get a "
        "chance!"
    ),
    ("approve_or_decline", "neutral"): (
        "We'd like to move the settlement date forward by a week. It saves "
        "us a cycle, but it does mean less review time. Can you approve?"
    ),
    ("approve_or_decline", "assertive"): (
        "We're moving the settlement date forward by a week. It saves us a "
        "cycle. I need your approval today -- the reduced review time is "
        "acceptable given the timeline."
    ),
    ("report_problem", "deferential"): (
        "Sorry to check in again -- I don't want to be a nuisance, but is "
        "everything still on track for the reconciliation by Friday? No "
        "worries at all if you need more time."
    ),
    ("report_problem", "warm"): (
        "Hey! Just wanted to check in -- how's the reconciliation coming "
        "along? Let me know if there's anything you need from me to hit "
        "Friday."
    ),
    ("report_problem", "neutral"): (
        "Just checking in on the reconciliation - is everything on track " "for Friday?"
    ),
    ("report_problem", "assertive"): (
        "I need a status update on the reconciliation now. Confirm it will "
        "be done by Friday or tell me what's blocking it."
    ),
    ("schedule_coordination", "deferential"): (
        "I know everyone's calendars are packed, so I'm sorry to add to the "
        "scheduling headache -- whenever it's convenient, could you let me "
        "know your availability next week? We do need to get everyone in a "
        "room before the deadline, but happy to work around you."
    ),
    ("schedule_coordination", "warm"): (
        "Trying to wrangle everyone's calendars for a meeting before the "
        "deadline -- no small task! What's your availability looking like "
        "next week?"
    ),
    ("schedule_coordination", "neutral"): (
        "We need to get everyone in a room before the deadline. What does "
        "your availability look like next week?"
    ),
    ("schedule_coordination", "assertive"): (
        "We need everyone in a room before the deadline. Send me your "
        "availability for next week by end of day."
    ),
    ("resolve_disagreement", "deferential"): (
        "I might be missing something, but I'm a little unsure about the "
        "approach in the draft -- it seems like the numbers might have been "
        "run on the wrong basis? I could easily be wrong here, so let me "
        "know how you'd like to handle it."
    ),
    ("resolve_disagreement", "warm"): (
        "So I took a closer look at the draft and I think the numbers might "
        "be run on the wrong basis -- easy mistake to make! How do you want "
        "to handle it?"
    ),
    ("resolve_disagreement", "neutral"): (
        "I don't think the approach in the draft holds up. It seems to me "
        "the numbers were run on the wrong basis. How do you want to handle "
        "this?"
    ),
    ("resolve_disagreement", "assertive"): (
        "The approach in the draft doesn't hold up. The numbers were run on "
        "the wrong basis. This needs to be fixed before it goes any further "
        "-- how do you want to handle it?"
    ),
    ("confirm_details", "deferential"): (
        "Sorry to double-check, but I just wanted to confirm -- are we "
        "still moving forward with the numbers from last week's call? "
        "Please let me know if anything's changed, no rush."
    ),
    ("confirm_details", "warm"): (
        "Quick check-in before I send this out -- we're still good to go "
        "with the numbers from last week's call, right? Let me know if "
        "anything's shifted!"
    ),
    ("confirm_details", "neutral"): (
        "Just confirming -- we're still moving forward with the numbers "
        "from last week's call, right? Let me know if anything's changed "
        "before I send this out."
    ),
    ("confirm_details", "assertive"): (
        "Confirm now: are we still moving forward with the numbers from "
        "last week's call? I need to know before this goes out."
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
    """The full scenario set: task type x direction x stakes x tone."""
    scenarios = []
    for task_type, direction, stakes, tone in product(TASK_TYPES, DIRECTIONS, STAKES, TONES):
        scenarios.append(
            Scenario(
                scenario_id=f"{task_type}__{direction}__{stakes}__{tone}",
                task_type=task_type,
                direction=direction,
                stakes=stakes,
                tone=tone,
                situation=_SITUATIONS[task_type],
                incoming_message=_INCOMING_MESSAGES[(task_type, tone)],
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
