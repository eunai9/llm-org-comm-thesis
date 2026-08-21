"""Judge prompt assembly, blinding, and the three frozen phrasing variants.

**Blind interleaving is the whole calibration argument.** A judge that scores
real and generated messages differently only because it can *tell which is
which* is not measuring fidelity, it is measuring its own bias about which
kind of text it prefers. So :func:`build_judge_item` strips every marker of
provenance -- no field, no ordering, nothing in the prompt text -- states
that a message could be either, and never once uses the word "generated" or
"model" anywhere the judge can see. Provenance is tracked only on the
:class:`JudgeItem` the calling code holds, never inside the request itself.

**Three frozen phrasing variants, not one.** The plan calls for measuring how
sensitive judge scores are to how the *same* rubric is phrased -- neutral,
reordered-and-rephrased, and role-framed. All three ask the identical six
questions; only the wording and the item order differ. Consistency across
variants is itself a finding (a judge stable in substance but sensitive in
phrasing is a different result than one stable in both), so this project
should never resolve variant drift by picking whichever phrasing scores
best -- all three are used and reported, not selected among.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from thesis.judge.rubric import RUBRIC_ITEMS

Variant = Literal["neutral", "reordered", "role_framed"]
VARIANTS: tuple[Variant, ...] = ("neutral", "reordered", "role_framed")


@dataclass(frozen=True, slots=True)
class JudgeItem:
    """One thing to be scored.

    ``is_generated`` and ``source_id`` are read only by the calling code that
    tracks results -- never rendered into the prompt text itself, which is
    what keeps scoring genuinely blind.
    """

    item_id: str
    text: str
    is_generated: bool
    source_id: str


_TASK_FRAMING: dict[Variant, str] = {
    "neutral": (
        "You are evaluating a single workplace email. It may be a real email "
        "from a corporate archive, or a message written for a research study "
        "meant to resemble one -- you are not told which, and should not "
        "guess; score only what is in front of you.\n\n"
        "Read the message, then rate it on each of the following dimensions."
    ),
    "reordered": (
        "Below is one workplace email, of unknown origin -- possibly drawn "
        "from a real archive, possibly written for a study designed to "
        "resemble one. Score strictly what the text itself shows you; do not "
        "try to infer or guess where it came from.\n\n"
        "Rate the message on each dimension below, working through them in "
        "the order given."
    ),
    "role_framed": (
        "Imagine you are an experienced manager reviewing workplace "
        "correspondence, asked to judge one email on its merits. You have "
        "not been told, and should not guess, whether it is a real message "
        "from a corporate archive or one written to resemble one -- treat "
        "that as unknowable and irrelevant to your scoring.\n\n"
        "Evaluate the message against each of the following criteria."
    ),
}

# Same six items in each variant; "reordered" is the swap of the two groups'
# order plus a within-group reversal, so the item text a judge reads first
# and last both differ from the neutral variant.
_ITEM_ORDER: dict[Variant, tuple[str, ...]] = {
    "neutral": tuple(item.key for item in RUBRIC_ITEMS),
    "reordered": tuple(
        item.key
        for item in (
            *[i for i in RUBRIC_ITEMS if i.group == "communication_performance"][::-1],
            *[i for i in RUBRIC_ITEMS if i.group == "empirical_fidelity"][::-1],
        )
    ),
    "role_framed": tuple(item.key for item in RUBRIC_ITEMS),
}

_ITEM_PHRASING: dict[Variant, dict[str, str]] = {
    "neutral": {item.key: item.prompt for item in RUBRIC_ITEMS},
    "reordered": {
        # Same question, reworded -- not merely reordered -- so this variant
        # tests phrasing sensitivity, not only sequence sensitivity.
        "role_consistency": (
            "Judging by the writing alone, does it match someone at the "
            "stated role and seniority level?"
        ),
        "contextual_fit": (
            "Does the reply speak to the actual specifics of the situation "
            "and message in front of it, rather than in general terms?"
        ),
        "corpus_plausibility": (
            "Would this message blend in among real corporate email from a "
            "company like this one?"
        ),
        "clarity": "Is it obvious, from a single read, what the message is saying?",
        "politeness_appropriateness": (
            "Given who is writing to whom and why, is the tone well-judged -- "
            "neither too blunt nor needlessly formal?"
        ),
        "conflict_management": (
            "If the message pushes back or disagrees, is that handled in a "
            "way that protects the relationship? (Score the middle if there "
            "is nothing to push back on.)"
        ),
    },
    "role_framed": {item.key: item.prompt for item in RUBRIC_ITEMS},
}

_OUTPUT_INSTRUCTION = (
    "\n\nFor each dimension, give a short piece of supporting evidence from "
    "the message before your score, then a score from 1 (low) to 5 (high) "
    "using the anchors given for that dimension."
)


def render_rubric_block(variant: Variant) -> str:
    """The fixed part of a judge prompt for one phrasing variant.

    Cacheable across every item scored in that variant -- the rubric text
    itself never changes per item, only the message being judged does.
    """
    by_key = {item.key: item for item in RUBRIC_ITEMS}
    order = _ITEM_ORDER[variant]
    phrasing = _ITEM_PHRASING[variant]

    lines = [_TASK_FRAMING[variant], ""]
    for key in order:
        item = by_key[key]
        lines.append(
            f"- **{key}**: {phrasing[key]} " f"(1 = {item.low_anchor} 5 = {item.high_anchor})"
        )
    lines.append(_OUTPUT_INSTRUCTION.strip())
    return "\n".join(lines)


def render_item_block(item: JudgeItem) -> str:
    """The variable part: the message to be judged, and nothing else about it.

    No item id, no source id, no provenance -- anything here is visible to
    the judge and must therefore contain nothing that would break blinding.
    """
    return f"## The message\n\n{item.text}"
