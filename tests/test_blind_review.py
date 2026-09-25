"""Tests for the blind coding page.

What matters most is that the page cannot reveal which model wrote which
reply, and that no reply is shown twice.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from thesis.analysis.blind_review import (
    YES_NO_QUESTIONS,
    assign_slots,
    build_page_data,
    draw_blind_sample,
    prompt_key,
    render_page,
)
from thesis.analysis.review_pack import CODING_COLUMNS, FAILURE_MODES

FIRST_MODEL = "local/llama3.2:3b"
SECOND_MODEL = "nim/deepseek-ai/deepseek-v4-flash-0731"


def _pairs(
    label: str, model: str, replies: list[str], stimuli: list[str] | None = None
) -> pd.DataFrame:
    n = len(replies)
    return pd.DataFrame(
        {
            "cell_id": [f"{label}__real_t{i}__r{i}" for i in range(n)],
            "thread_id": [f"t{i:03d}" for i in range(n)],
            "persona_id": ["r2_trading"] * n,
            "seniority_rank": [2] * n,
            "department": ["Trading"] * n,
            "direction": ["down"] * n,
            "stimulus_text": stimuli or [f"Please send report {i}." for i in range(n)],
            "real_reply_body_recleaned": [f"Real reply {i}." for i in range(n)],
            "generated_reply": replies,
            "model": [model] * n,
        }
    )


def _both(n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = _pairs("sim_local", FIRST_MODEL, [f"first reply {i}" for i in range(n)])
    second = _pairs("sim_nvidia", SECOND_MODEL, [f"second reply {i}" for i in range(n)])
    return first, second


def test_pairs_with_the_same_prompt_are_kept_once() -> None:
    """Two real replies to one email give one prompt and one generated reply."""
    stimuli = ["Please send the deck.", "Please send the deck.", "Can you call me?"]
    first = _pairs("sim_local", FIRST_MODEL, ["a", "a", "b"], stimuli)
    second = _pairs("sim_nvidia", SECOND_MODEL, ["x", "x", "y"], stimuli)
    assert prompt_key(first).nunique() == 2
    sample = draw_blind_sample(first, second, n=50)
    assert len(sample) == 2


def test_both_replies_belong_to_the_same_email() -> None:
    first, second = _both(4)
    shuffled = second.iloc[::-1].reset_index(drop=True)
    sample = draw_blind_sample(first, shuffled, n=50)
    for row in sample.itertuples(index=False):
        assert row.generated_reply_first.split()[-1] == row.generated_reply_second.split()[-1]


def test_a_missing_reply_from_the_second_model_is_refused() -> None:
    first, second = _both(3)
    with pytest.raises(ValueError, match="no reply from the second model"):
        draw_blind_sample(first, second.iloc[:2], n=50)


def test_each_email_gets_one_reply_per_model_in_balanced_slots() -> None:
    first, second = _both(10)
    sample = draw_blind_sample(first, second, n=50)
    items, key = assign_slots(sample, (FIRST_MODEL, SECOND_MODEL))
    assert len(items) == 20
    for _, group in key.groupby("email"):
        assert sorted(group["model"]) == sorted([FIRST_MODEL, SECOND_MODEL])
        assert sorted(group["slot"]) == ["A", "B"]
    second_in_a = key[(key["slot"] == "A") & (key["model"] == SECOND_MODEL)]
    assert len(second_in_a) == 5


def test_slot_assignment_is_reproducible() -> None:
    first, second = _both(10)
    sample = draw_blind_sample(first, second, n=50)
    _, key_one = assign_slots(sample, (FIRST_MODEL, SECOND_MODEL), seed=7)
    _, key_two = assign_slots(sample, (FIRST_MODEL, SECOND_MODEL), seed=7)
    pd.testing.assert_frame_equal(key_one, key_two)


def test_the_page_does_not_reveal_the_model() -> None:
    first, second = _both(6)
    sample = draw_blind_sample(first, second, n=50)
    items, _ = assign_slots(sample, (FIRST_MODEL, SECOND_MODEL))
    page = render_page(build_page_data(sample, items)).lower()
    for leak in ("llama", "deepseek", "sim_local", "sim_nvidia", "nim/", "local/", "__real_"):
        assert leak not in page
    for i in range(6):
        assert f"first reply {i}" in page
        assert f"second reply {i}" in page


def test_a_script_tag_inside_an_email_cannot_end_the_data_block() -> None:
    attack = "</script><script>alert(1)</script>"
    first = _pairs("sim_local", FIRST_MODEL, [attack])
    second = _pairs("sim_nvidia", SECOND_MODEL, ["fine"])
    sample = draw_blind_sample(first, second, n=50)
    items, _ = assign_slots(sample, (FIRST_MODEL, SECOND_MODEL))
    page = render_page(build_page_data(sample, items))
    assert attack not in page
    data = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)
    assert data is not None
    assert attack in json.dumps(json.loads(data.group(1)))


def test_questions_and_codebook_match_the_coding_sheet() -> None:
    asked = [column for column, _ in YES_NO_QUESTIONS]
    assert [*asked, "failure_mode", "notes"] == list(CODING_COLUMNS)
    labels = [name for name, _ in FAILURE_MODES]
    assert "wrong_register" in labels
    first, second = _both(2)
    sample = draw_blind_sample(first, second, n=50)
    items, _ = assign_slots(sample, (FIRST_MODEL, SECOND_MODEL))
    data = build_page_data(sample, items)
    assert [name for name, _ in data["codebook"]] == labels  # type: ignore[attr-defined]
    assert data["columns"] == ["email", "slot", *CODING_COLUMNS, "coder"]


def test_the_writer_and_direction_are_described_in_plain_words() -> None:
    first, second = _both(1)
    sample = draw_blind_sample(first, second, n=50)
    items, _ = assign_slots(sample, (FIRST_MODEL, SECOND_MODEL))
    email = build_page_data(sample, items)["emails"][0]  # type: ignore[index]
    assert email["writer"] == "a Manager in Trading (seniority rank 2 of 6)"
    assert email["direction"] == "writing down to someone more junior"
