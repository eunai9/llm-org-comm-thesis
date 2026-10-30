from __future__ import annotations

import pandas as pd
import pytest

from thesis.analysis.tone_judge import (
    InvalidToneResponseError,
    balanced_sample,
    build_tone_request,
    validate_tone_response,
    weighted_kappa,
)
from thesis.judge.prompt import JudgeItem


def test_validate_accepts_scale_and_rejects_everything_else() -> None:
    assert validate_tone_response({"impression": "calm", "authority": 7}) == 7
    for bad in (0, 8, "5", 4.5, True, None):
        with pytest.raises(InvalidToneResponseError):
            validate_tone_response({"impression": "calm", "authority": bad})


def test_prompt_names_no_cues() -> None:
    item = JudgeItem(item_id="x", text="Send it today.", is_generated=True, source_id="x")
    request = build_tone_request(item, "stub", replicate=2)
    assert request.variant == 2
    framing = (request.system or "").lower()
    for cue in ("hedge", "imperative", "instruction", "word choice", "senior", "junior"):
        assert cue not in framing


def test_weighted_kappa_is_one_for_identical_passes_and_uses_shared_items() -> None:
    first = {"a": 1, "b": 4, "c": 7}
    kappa, n = weighted_kappa([first, dict(first), {"a": 1, "b": 4, "c": 7, "d": 2}])
    assert kappa == pytest.approx(1.0)
    assert n == 3


def test_balanced_sample_takes_the_same_count_per_direction() -> None:
    frame = pd.DataFrame({"direction": ["up"] * 5 + ["down"] * 7 + ["lateral"] * 9})
    picked = balanced_sample(frame, per_direction=3, seed=1)
    assert picked["direction"].value_counts().to_dict() == {"up": 3, "down": 3, "lateral": 3}
