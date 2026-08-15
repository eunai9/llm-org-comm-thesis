"""The structured shape every simulated response must take.

Constraining the output instead of parsing free text matters for a specific
reason: ``decision`` is a **dependent variable in Q1**. If it were recovered by
running a second model over generated prose, the analysis would inherit that
extractor's error rate and its biases, and every decision-attitude result would
be a measurement of two models rather than one. Making it a required field
means the simulator states its decision directly.

Two constraints of the structured-output API shape this schema:

- ``additionalProperties: false`` is required on every object.
- **Numeric bounds are not supported.** ``minimum`` / ``maximum`` /
  ``multipleOf`` are silently unavailable, so a 0-1 confidence score could not
  actually be constrained to 0-1 by the schema.

That second point is why ``confidence`` is an ordinal enum rather than a float.
The schema can enforce an enum exactly, an LLM produces a coarse ordinal far
more reliably than a calibrated decimal, and three ordered levels invite
ordinal analysis instead of implying a precision that an uncalibrated 0.87
does not have.
"""

from __future__ import annotations

from typing import Any, Final

DECISIONS: Final[tuple[str, ...]] = ("accept", "decline", "defer", "escalate", "none")
CONFIDENCE_LEVELS: Final[tuple[str, ...]] = ("low", "medium", "high")

# Ordinal coding for analysis. Kept beside the enum so the mapping used in the
# results is the same one the schema defines, rather than being re-invented in
# a notebook.
CONFIDENCE_ORDINAL: Final[dict[str, int]] = {"low": 1, "medium": 2, "high": 3}

RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "subject": {
            "type": "string",
            "description": "Subject line of the reply.",
        },
        "body": {
            "type": "string",
            "description": "The body of the reply, as it would be sent.",
        },
        "decision": {
            "type": "string",
            "enum": list(DECISIONS),
            "description": (
                "The stance this reply takes on what was asked. Use 'none' "
                "when the message asks for nothing decidable."
            ),
        },
        "confidence": {
            "type": "string",
            "enum": list(CONFIDENCE_LEVELS),
            "description": "How firmly the reply commits to that decision.",
        },
        "reasoning_brief": {
            "type": "string",
            "description": ("One sentence on why this stance, from the role's point of view."),
        },
    },
    "required": ["subject", "body", "decision", "confidence", "reasoning_brief"],
    "additionalProperties": False,
}


class InvalidResponseError(ValueError):
    """Raised when a response does not satisfy the agreed shape."""


def validate_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Check a decoded response and return it unchanged.

    The API enforces the schema, so this is a second line of defence rather
    than the primary one -- it catches a cached response written under an older
    schema version, which would otherwise flow into the analysis unnoticed.
    """
    missing = set(RESPONSE_SCHEMA["required"]) - set(payload)
    if missing:
        msg = f"response missing required field(s): {sorted(missing)}"
        raise InvalidResponseError(msg)

    if payload["decision"] not in DECISIONS:
        msg = f"decision {payload['decision']!r} not one of {list(DECISIONS)}"
        raise InvalidResponseError(msg)

    if payload["confidence"] not in CONFIDENCE_LEVELS:
        msg = f"confidence {payload['confidence']!r} not one of {list(CONFIDENCE_LEVELS)}"
        raise InvalidResponseError(msg)

    return payload
