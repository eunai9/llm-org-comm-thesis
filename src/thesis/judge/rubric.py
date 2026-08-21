"""The judge rubric: what the judge scores, and the shape it scores in.

Two groups of three items, each a 1-5 anchored scale with a required
free-text ``evidence`` field:

- **Empirical fidelity** -- does this read as something this role, in this
  situation, would actually write? Role consistency, contextual fit, corpus
  plausibility.
- **Communication performance** -- is it well-written, on its own terms?
  Clarity, politeness appropriateness (*appropriate* to the situation, not
  maximally polite -- a terse reply to a routine request is not a defect),
  conflict management.

**Evidence before score, in the schema's own property order.** This is not
cosmetic: asking a model to quote the specific text that supports a score
before committing to the number measurably improves calibration, and gives a
human reviewer something concrete to check a score against rather than a bare
digit to take on faith.

**Every item must be answerable for a real Enron message, not only a
generated one.** No item's wording may say "the generated message" or "the
model" -- phrasing that would make blind interleaving impossible. That
interleaving (:mod:`thesis.judge.prompt` strips provenance and shuffles real
and generated items together before scoring) *is* the calibration argument:
a judge that can tell them apart by wording alone was never blind in the
first place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

RubricGroup = Literal["empirical_fidelity", "communication_performance"]

SCORE_MIN = 1
SCORE_MAX = 5
SCORE_VALUES: Final[tuple[int, ...]] = tuple(range(SCORE_MIN, SCORE_MAX + 1))


@dataclass(frozen=True, slots=True)
class RubricItem:
    """One scored dimension: a key, its group, and the anchors shown to the judge.

    ``low_anchor``/``high_anchor`` are what "1" and "5" concretely mean --
    without them, "rate clarity 1-5" is under-specified and a judge (human or
    model) has to invent its own scale, which is exactly what destroys
    cross-item and cross-rater comparability.
    """

    key: str
    group: RubricGroup
    prompt: str
    low_anchor: str
    high_anchor: str


RUBRIC_ITEMS: Final[tuple[RubricItem, ...]] = (
    RubricItem(
        key="role_consistency",
        group="empirical_fidelity",
        prompt=(
            "Does this message read as something written by someone in the "
            "stated role, at the stated seniority level?"
        ),
        low_anchor="Nothing about the writing suggests this role or level -- it "
        "could have been written by anyone.",
        high_anchor="The language, authority, and concerns expressed are exactly "
        "what you would expect from this specific role and level.",
    ),
    RubricItem(
        key="contextual_fit",
        group="empirical_fidelity",
        prompt=(
            "Does this reply actually engage with the specific situation and "
            "the specific message it is responding to?"
        ),
        low_anchor="Generic -- it could be a reply to almost any message with "
        "similar surface features.",
        high_anchor="Precisely engaged with the specific details, stakes, and "
        "relationship at hand.",
    ),
    RubricItem(
        key="corpus_plausibility",
        group="empirical_fidelity",
        prompt=(
            "Could this message plausibly have appeared in a real corporate "
            "email archive from this kind of company?"
        ),
        low_anchor="Reads as obviously synthetic -- too polished, too "
        "explanatory, or otherwise unlike real workplace email.",
        high_anchor="Indistinguishable in register and habits from real "
        "workplace correspondence.",
    ),
    RubricItem(
        key="clarity",
        group="communication_performance",
        prompt="Is the message clear about what it is saying and what it wants?",
        low_anchor="Confusing or ambiguous about its own point.",
        high_anchor="Immediately clear what is being said and what, if anything, "
        "is being asked for.",
    ),
    RubricItem(
        key="politeness_appropriateness",
        group="communication_performance",
        prompt=(
            "Is the level of politeness appropriate to the relationship and "
            "situation -- not necessarily maximally polite, but fitting?"
        ),
        low_anchor="Noticeably wrong for the relationship -- too brusque for the "
        "context, or oddly over-formal for a routine exchange.",
        high_anchor="The tone fits the relationship and stakes exactly as a "
        "real colleague's would.",
    ),
    RubricItem(
        key="conflict_management",
        group="communication_performance",
        prompt=(
            "Where the message disagrees, declines, or pushes back, does it "
            "do so in a way that manages the working relationship well? (Score "
            "the middle of the scale, not low, if there is no disagreement to "
            "manage.)"
        ),
        low_anchor="Handles disagreement in a way that would needlessly damage "
        "the relationship (or is unnecessarily combative where no "
        "disagreement existed).",
        high_anchor="Disagreement, where present, is handled in a way that "
        "preserves the working relationship.",
    ),
)

RUBRIC_BY_KEY: Final[dict[str, RubricItem]] = {item.key: item for item in RUBRIC_ITEMS}


def _item_schema(item: RubricItem) -> dict[str, Any]:
    """One rubric item's structured-output shape.

    ``evidence`` is declared before ``score`` in the properties dict -- the
    schema's own property order -- for the calibration reason given in the
    module docstring. ``score`` is an integer **enum**, not a ``minimum``/
    ``maximum`` range: this project has already found, building the
    simulator's own schema, that structured-output APIs do not reliably
    enforce numeric bounds, so a range constraint could not actually be
    trusted to keep the score in 1-5. An explicit enum is exact instead of
    hopeful.
    """
    return {
        "type": "object",
        "properties": {
            "evidence": {
                "type": "string",
                "description": (
                    f"A short, specific quote or paraphrase from the message "
                    f"that supports your score for: {item.prompt}"
                ),
            },
            "score": {
                "type": "integer",
                "enum": list(SCORE_VALUES),
                "description": (f"1 = {item.low_anchor} 5 = {item.high_anchor}"),
            },
        },
        "required": ["evidence", "score"],
        "additionalProperties": False,
    }


def build_judge_schema() -> dict[str, Any]:
    """The full structured-output schema: one sub-object per rubric item."""
    return {
        "type": "object",
        "properties": {item.key: _item_schema(item) for item in RUBRIC_ITEMS},
        "required": [item.key for item in RUBRIC_ITEMS],
        "additionalProperties": False,
    }


class InvalidJudgeResponseError(ValueError):
    """Raised when a judge response does not satisfy the rubric's shape."""


def validate_judge_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Check a decoded judge response and return it unchanged.

    A second line of defence behind the API's own schema enforcement -- the
    same role :func:`thesis.sim.schemas.validate_response` plays for the
    simulator, catching a cached response written under an older rubric
    version before it reaches analysis unnoticed.
    """
    missing = set(RUBRIC_BY_KEY) - set(payload)
    if missing:
        msg = f"judge response missing rubric item(s): {sorted(missing)}"
        raise InvalidJudgeResponseError(msg)

    for key, entry in payload.items():
        if key not in RUBRIC_BY_KEY:
            continue
        if not isinstance(entry, dict) or "score" not in entry or "evidence" not in entry:
            msg = f"item {key!r} missing 'score' or 'evidence'"
            raise InvalidJudgeResponseError(msg)
        if entry["score"] not in SCORE_VALUES:
            msg = f"item {key!r} score {entry['score']!r} not in {SCORE_VALUES}"
            raise InvalidJudgeResponseError(msg)

    return payload
