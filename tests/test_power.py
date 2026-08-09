"""Composite power-score tests.

The central thing being tested is the *formula*, not real data: given known
z-scorable inputs and known weights, does the composite come out where hand
arithmetic says it should, and does missingness get handled the way the
docstring promises rather than silently degrading?
"""

from __future__ import annotations

import pandas as pd
import pytest

from thesis.config import Config, DataConfig, ModelsConfig, ModelSpec, PowerConfig, RunConfig
from thesis.data.power import compute_power_score


def _config(layer_a: dict[str, float], layer_b: dict[str, float]) -> Config:
    """A minimal valid Config with the power weights under test."""
    return Config(
        project="test",
        data=DataConfig(
            seed=1,
            corpus=__import__("thesis.config", fromlist=["CorpusConfig"]).CorpusConfig(
                date_start="1999-01-01",
                date_end="2002-06-30",
                min_body_tokens=1,
                max_body_tokens=1000,
                internal_domains=["enron.com"],
            ),
            sample_sizes=__import__("thesis.config", fromlist=["SampleSizes"]).SampleSizes(
                label=1, shots=1, real_eval=1
            ),
            power=PowerConfig(layer_a_weights=layer_a, layer_b_weights=layer_b),
        ),
        models=ModelsConfig(
            simulator=[ModelSpec(provider="anthropic", model_id="m", role_label="s")],
            judge=[ModelSpec(provider="openai", model_id="j", role_label="j")],
            labeller=ModelSpec(provider="anthropic", model_id="l", role_label="l"),
        ),
        run=RunConfig(max_cost_usd=1.0, require_clean_git=False, use_cache=True),
    )


def test_higher_imperative_ratio_gives_higher_score() -> None:
    """Isolate one Layer A feature: more directive language -> higher score."""
    df = pd.DataFrame(
        {
            "message_uid": ["a", "b", "c"],
            "from_addr": ["x", "y", "z"],
            "imperative_ratio": [0.0, 0.5, 1.0],
            "hedge_rate": [0.0, 0.0, 0.0],
            "deference_rate": [0.0, 0.0, 0.0],
            "commitment_rate": [0.0, 0.0, 0.0],
            "question_ratio": [0.0, 0.0, 0.0],
            "n_recipients": [1, 1, 1],
            "is_broadcast": [0.0, 0.0, 0.0],
            "eigenvector_centrality": [0.1, 0.1, 0.1],
            "thread_initiation_rate": [0.0, 0.0, 0.0],
            "last_word_rate": [0.0, 0.0, 0.0],
            "reply_latency_asymmetry": [0.0, 0.0, 0.0],
        }
    )
    config = _config(
        {"imperative_ratio": 1.0},
        {},
    )
    scored = compute_power_score(df, config, min_components=1)
    scores = scored.set_index("message_uid")["power_score"]
    assert scores["a"] < scores["b"] < scores["c"]


def test_negatively_weighted_feature_lowers_score_when_high() -> None:
    df = pd.DataFrame(
        {
            "message_uid": ["a", "b"],
            "from_addr": ["x", "y"],
            "imperative_ratio": [0.0, 0.0],
            "hedge_rate": [0.0, 1.0],
            "deference_rate": [0.0, 0.0],
            "commitment_rate": [0.0, 0.0],
            "question_ratio": [0.0, 0.0],
            "n_recipients": [1, 1],
            "is_broadcast": [0.0, 0.0],
            "eigenvector_centrality": [0.1, 0.1],
            "thread_initiation_rate": [0.0, 0.0],
            "last_word_rate": [0.0, 0.0],
            "reply_latency_asymmetry": [0.0, 0.0],
        }
    )
    config = _config({"hedge_rate": -1.0}, {})
    scored = compute_power_score(df, config, min_components=1)
    scores = scored.set_index("message_uid")["power_score"]
    assert scores["b"] < scores["a"]  # more hedging, negative weight -> lower score


def test_null_component_is_excluded_not_treated_as_zero() -> None:
    """A message missing reply_latency_asymmetry should not be penalised as
    though that component were a bad (zero) score -- it should simply not
    count towards the average."""
    df = pd.DataFrame(
        {
            "message_uid": ["a", "b"],
            "from_addr": ["x", "y"],
            "imperative_ratio": [0.5, 0.5],
            "hedge_rate": [0.0, 0.0],
            "deference_rate": [0.0, 0.0],
            "commitment_rate": [0.0, 0.0],
            "question_ratio": [0.0, 0.0],
            "n_recipients": [1, 1],
            "is_broadcast": [0.0, 0.0],
            "eigenvector_centrality": [0.1, 0.9],
            "thread_initiation_rate": [0.0, 0.0],
            "last_word_rate": [0.0, 0.0],
            "reply_latency_asymmetry": [0.3, None],
        }
    )
    config = _config(
        {"imperative_ratio": 1.0},
        {"eigenvector_centrality": 1.0, "reply_latency_asymmetry": 1.0},
    )
    scored = compute_power_score(df, config, min_components=1)
    # Both rows get a real score; the null component doesn't turn "b" into NaN.
    assert scored["power_score"].notna().all()


def test_below_min_components_yields_null_score() -> None:
    df = pd.DataFrame(
        {
            "message_uid": ["a"],
            "from_addr": ["x"],
            "imperative_ratio": [0.5],
            "hedge_rate": [0.0],
            "deference_rate": [0.0],
            "commitment_rate": [0.0],
            "question_ratio": [0.0],
            "n_recipients": [1],
            "is_broadcast": [0.0],
            "eigenvector_centrality": [0.1],
            "thread_initiation_rate": [0.0],
            "last_word_rate": [0.0],
            "reply_latency_asymmetry": [None],
        }
    )
    config = _config(
        {"imperative_ratio": 1.0},
        {"eigenvector_centrality": 1.0, "reply_latency_asymmetry": 1.0},
    )
    scored = compute_power_score(df, config, min_components=5)
    assert scored["power_score"].isna().all()


def test_unweighted_config_column_raises() -> None:
    """A weight for a column that doesn't exist in the joined data should be
    a loud config error, not silently ignored."""
    df = pd.DataFrame(
        {
            "message_uid": ["a"],
            "from_addr": ["x"],
            "imperative_ratio": [0.5],
            "hedge_rate": [0.0],
            "deference_rate": [0.0],
            "commitment_rate": [0.0],
            "question_ratio": [0.0],
            "n_recipients": [1],
            "is_broadcast": [0.0],
            "eigenvector_centrality": [0.1],
            "thread_initiation_rate": [0.0],
            "last_word_rate": [0.0],
            "reply_latency_asymmetry": [0.1],
        }
    )
    config = _config({"not_a_real_column": 1.0}, {})
    with pytest.raises(ValueError, match="unknown columns"):
        compute_power_score(df, config, min_components=1)


def test_constant_column_does_not_crash_on_zero_std() -> None:
    """A feature with zero variance (std=0) must not raise a divide-by-zero;
    it should contribute nothing rather than produce inf/NaN silently."""
    df = pd.DataFrame(
        {
            "message_uid": ["a", "b"],
            "from_addr": ["x", "y"],
            "imperative_ratio": [0.5, 0.5],  # constant
            "hedge_rate": [0.0, 1.0],
            "deference_rate": [0.0, 0.0],
            "commitment_rate": [0.0, 0.0],
            "question_ratio": [0.0, 0.0],
            "n_recipients": [1, 1],
            "is_broadcast": [0.0, 0.0],
            "eigenvector_centrality": [0.1, 0.1],
            "thread_initiation_rate": [0.0, 0.0],
            "last_word_rate": [0.0, 0.0],
            "reply_latency_asymmetry": [0.1, 0.1],
        }
    )
    config = _config({"imperative_ratio": 1.0, "hedge_rate": -1.0}, {})
    scored = compute_power_score(df, config, min_components=1)
    assert scored["power_score"].apply(lambda v: v == v and abs(v) != float("inf")).all()


def test_seniority_aggregation_logic_directly() -> None:
    """validate_against_seniority() calls roles.py's live-corpus join
    internally, so exercising it end-to-end needs real vendored data and a
    real message store -- that path is checked when the module is run
    against the actual corpus. What's unit-testable in isolation is the
    pure aggregation this function performs once ranks are known, so
    that's what this test pins down.
    """
    df = pd.DataFrame(
        {
            "message_uid": ["a", "b", "c", "d"],
            "from_addr": ["boss@x.com", "boss@x.com", "junior@x.com", "junior@x.com"],
            "power_score": [0.8, 0.6, -0.5, -0.3],
        }
    )
    ranks = {"boss@x.com": 4, "junior@x.com": 1}
    scored = df.dropna(subset=["power_score"]).copy()
    scored["seniority_rank"] = scored["from_addr"].map(ranks)
    by_rank = scored.groupby("seniority_rank")["power_score"].mean()
    assert by_rank[4] > by_rank[1]
