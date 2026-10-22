from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis.analysis.role_cues import (
    CacheMissError,
    CacheOnlyClient,
    fit_cue_model,
    judge_guesses,
    reply_cues,
)
from thesis.analysis.role_inference import build_absolute_request
from thesis.data.features import _load_nlp
from thesis.judge.prompt import JudgeItem
from thesis.llm.base import CompletionResponse, Usage
from thesis.llm.cache import ResponseCache, cache_key


def test_reply_cues_measure_orders_you_gratitude_phrases_and_length() -> None:
    text = "Send me the report today. Thank you for your help."
    cues = reply_cues(text, _load_nlp()(text), {("thank", "you"): -2.0})
    assert cues["imperative_ratio"] > 0
    assert cues["you_rate"] == pytest.approx(2 / 10)
    assert cues["gratitude"] == 1.0
    assert cues["gilbert_score"] < 0
    assert cues["log_words"] == pytest.approx(np.log(10))


def test_fit_cue_model_finds_the_cue_that_drives_the_guess() -> None:
    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame({"driver": rng.normal(size=n), "noise": rng.normal(size=n)})
    frame["guess"] = np.where(
        frame["driver"] + rng.normal(scale=0.5, size=n) > 0, "down", "lateral"
    )
    table = fit_cue_model(frame, cues=("driver", "noise"), outcome="down").set_index("cue")
    assert table.loc["driver", "coefficient"] > 1
    assert table.loc["driver", "p"] < 0.001
    assert table.loc["noise", "p"] > 0.01


def test_fit_cue_model_skips_a_cue_that_never_varies() -> None:
    frame = pd.DataFrame(
        {"driver": np.linspace(-1, 1, 50), "flat": 0.0, "guess": ["down", "up"] * 25}
    )
    table = fit_cue_model(frame, cues=("driver", "flat"), outcome="down")
    assert list(table["cue"]) == ["driver"]


def test_cache_only_client_refuses_live_calls() -> None:
    item = JudgeItem(item_id="i1", text="Hi.", is_generated=True, source_id="i1")
    with pytest.raises(CacheMissError):
        CacheOnlyClient().complete(build_absolute_request(item, "openai/gpt-oss-120b@low"))


def test_judge_guesses_are_read_from_the_cache(tmp_path: Path) -> None:
    model = "openai/gpt-oss-120b@low"
    item = JudgeItem(item_id="i1", text="Do this now.", is_generated=True, source_id="i1")
    request = build_absolute_request(item, model)
    cache = ResponseCache(tmp_path / "cache")
    cache.put(
        cache_key(request, "groq"),
        request,
        CompletionResponse(
            text="{}",
            usage=Usage(input_tokens=10, output_tokens=5),
            model=model,
            parsed={"inferred_direction": "down", "evidence": "an order"},
        ),
        "groq",
    )
    guesses = judge_guesses([item], {"i1": "down"}, model=model, cache=cache)
    assert guesses.to_dict("records") == [
        {"item_id": "i1", "true_direction": "down", "guess": "down"}
    ]
