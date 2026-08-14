"""Sampling tests.

Fixtures write real Parquet stores (messages, threads) so the SQL runs
exactly as it does in production, the same pattern used for network.py and
identity.py. ``role_by_address`` is injected directly rather than routed
through the real, committed employee list, since matching synthetic test
addresses against real Enron employees would be coincidental at best.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from thesis.config import (
    Config,
    CorpusConfig,
    DataConfig,
    ModelsConfig,
    ModelSpec,
    PowerConfig,
    RunConfig,
    SampleSizes,
)
from thesis.data.sampling import (
    _stratified_sample,
    eligible_pool,
    sample_label,
    sample_shots_and_real_eval,
)

BASE = datetime(2001, 3, 1, 9, 0, 0)

MESSAGE_SCHEMA = pa.schema(
    [
        pa.field("message_uid", pa.string(), nullable=False),
        pa.field("from_addr", pa.string()),
        pa.field("date", pa.timestamp("ms")),
        pa.field("n_tokens_clean", pa.int32()),
        pa.field("is_empty_after_clean", pa.bool_()),
    ]
)
THREAD_SCHEMA = pa.schema(
    [
        pa.field("message_uid", pa.string(), nullable=False),
        pa.field("thread_id", pa.string(), nullable=False),
        pa.field("position_in_thread", pa.int32(), nullable=False),
        pa.field("thread_size", pa.int32(), nullable=False),
        pa.field("is_conversation", pa.bool_(), nullable=False),
    ]
)


def _config() -> Config:
    return Config(
        project="test",
        data=DataConfig(
            seed=7,
            corpus=CorpusConfig(
                date_start="2001-01-01",
                date_end="2001-12-31",
                min_body_tokens=20,
                max_body_tokens=600,
                internal_domains=["enron.com"],
            ),
            sample_sizes=SampleSizes(label=10, shots=2, real_eval=5),
            power=PowerConfig(layer_a_weights={}, layer_b_weights={}),
        ),
        models=ModelsConfig(
            simulator=[ModelSpec(provider="anthropic", model_id="m", role_label="s")],
            judge=[ModelSpec(provider="openai", model_id="j", role_label="j")],
            labeller=ModelSpec(provider="anthropic", model_id="l", role_label="l"),
        ),
        run=RunConfig(max_cost_usd=1.0, require_clean_git=False, use_cache=True),
    )


def _write_messages(tmp_path: Path, rows: list[dict[str, object]]) -> str:
    out = tmp_path / "messages"
    out.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=MESSAGE_SCHEMA), out / "part-00000.parquet")
    return str(out / "*.parquet")


def _write_threads(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    out = tmp_path / "threads.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=THREAD_SCHEMA), out)
    return out


def test_stratified_sample_hits_requested_size_and_is_proportional() -> None:
    pool = pd.DataFrame(
        {
            "seniority_rank": [1] * 80 + [2] * 20,
            "value": range(100),
        }
    )
    sample = _stratified_sample(pool, ["seniority_rank"], n=10, seed=1)
    assert len(sample) == 10
    counts = sample["seniority_rank"].value_counts()
    assert counts.get(1, 0) == 8
    assert counts.get(2, 0) == 2


def test_stratified_sample_is_deterministic_given_seed() -> None:
    pool = pd.DataFrame({"seniority_rank": [1, 1, 1, 2, 2, 2] * 10, "value": range(60)})
    a = _stratified_sample(pool, ["seniority_rank"], n=12, seed=42)
    b = _stratified_sample(pool, ["seniority_rank"], n=12, seed=42)
    assert sorted(a["value"]) == sorted(b["value"])


def test_stratified_sample_returns_whole_pool_when_n_exceeds_size() -> None:
    pool = pd.DataFrame({"seniority_rank": [1, 2, 3], "value": [10, 20, 30]})
    sample = _stratified_sample(pool, ["seniority_rank"], n=100, seed=1)
    assert len(sample) == 3


def test_eligible_pool_applies_all_filters(tmp_path: Path) -> None:
    rows = [
        {
            "message_uid": "ok",
            "from_addr": "a@enron.com",
            "date": BASE,
            "n_tokens_clean": 50,
            "is_empty_after_clean": False,
        },
        {
            "message_uid": "too_short",
            "from_addr": "a@enron.com",
            "date": BASE,
            "n_tokens_clean": 5,
            "is_empty_after_clean": False,
        },
        {
            "message_uid": "empty",
            "from_addr": "a@enron.com",
            "date": BASE,
            "n_tokens_clean": 50,
            "is_empty_after_clean": True,
        },
        {
            "message_uid": "external",
            "from_addr": "a@gmail.com",
            "date": BASE,
            "n_tokens_clean": 50,
            "is_empty_after_clean": False,
        },
        {
            "message_uid": "unknown_sender",
            "from_addr": "unknown@enron.com",
            "date": BASE,
            "n_tokens_clean": 50,
            "is_empty_after_clean": False,
        },
        {
            "message_uid": "out_of_window",
            "from_addr": "a@enron.com",
            "date": datetime(2003, 1, 1),
            "n_tokens_clean": 50,
            "is_empty_after_clean": False,
        },
    ]
    glob = _write_messages(tmp_path, rows)
    pool = eligible_pool(glob, config=_config(), role_by_address={"a@enron.com": 3})
    assert list(pool["message_uid"]) == ["ok"]
    assert pool.iloc[0]["seniority_rank"] == 3
    assert pool.iloc[0]["year"] == 2001


def test_sample_label_stratifies_by_rank_and_year() -> None:
    pool = pd.DataFrame(
        {
            "message_uid": [f"m{i}" for i in range(40)],
            "seniority_rank": [1] * 30 + [2] * 10,
            "year": [2000] * 20 + [2001] * 10 + [2000] * 10,
        }
    )
    sample = sample_label(pool, n=8, seed=3)
    assert len(sample) == 8


def test_sample_shots_and_real_eval_only_uses_fully_eligible_threads(tmp_path: Path) -> None:
    pool = pd.DataFrame(
        {
            "message_uid": ["t1_a", "t1_b", "t2_a"],
            "from_addr": ["x@enron.com"] * 3,
            "seniority_rank": [1, 1, 2],
            "year": [2001, 2001, 2001],
        }
    )
    threads = [
        # t1: both messages eligible -- should qualify.
        {
            "message_uid": "t1_a",
            "thread_id": "t1",
            "position_in_thread": 0,
            "thread_size": 2,
            "is_conversation": True,
        },
        {
            "message_uid": "t1_b",
            "thread_id": "t1",
            "position_in_thread": 1,
            "thread_size": 2,
            "is_conversation": True,
        },
        # t2: second message not in the eligible pool -- must not qualify.
        {
            "message_uid": "t2_a",
            "thread_id": "t2",
            "position_in_thread": 0,
            "thread_size": 2,
            "is_conversation": True,
        },
        {
            "message_uid": "t2_b",
            "thread_id": "t2",
            "position_in_thread": 1,
            "thread_size": 2,
            "is_conversation": True,
        },
        # t3: not a real conversation (single sender) -- must not qualify.
        {
            "message_uid": "t3_a",
            "thread_id": "t3",
            "position_in_thread": 0,
            "thread_size": 2,
            "is_conversation": False,
        },
    ]
    threads_path = _write_threads(tmp_path, threads)

    shots, real_eval = sample_shots_and_real_eval(
        pool, n_shots=5, n_real_eval=5, seed=1, threads_path=threads_path
    )

    assert list(shots["thread_id"]) == ["t1"]
    assert list(shots["message_uid"]) == ["t1_a"]
    assert list(real_eval["message_uid"]) == ["t1_b"]


def test_sample_real_eval_caps_at_requested_size(tmp_path: Path) -> None:
    pool = pd.DataFrame(
        {
            "message_uid": ["s", "r1", "r2", "r3"],
            "from_addr": ["x@enron.com"] * 4,
            "seniority_rank": [1, 1, 1, 1],
            "year": [2001, 2001, 2001, 2001],
        }
    )
    threads = [
        {
            "message_uid": "s",
            "thread_id": "t1",
            "position_in_thread": 0,
            "thread_size": 4,
            "is_conversation": True,
        },
        {
            "message_uid": "r1",
            "thread_id": "t1",
            "position_in_thread": 1,
            "thread_size": 4,
            "is_conversation": True,
        },
        {
            "message_uid": "r2",
            "thread_id": "t1",
            "position_in_thread": 2,
            "thread_size": 4,
            "is_conversation": True,
        },
        {
            "message_uid": "r3",
            "thread_id": "t1",
            "position_in_thread": 3,
            "thread_size": 4,
            "is_conversation": True,
        },
    ]
    threads_path = _write_threads(tmp_path, threads)

    _, real_eval = sample_shots_and_real_eval(
        pool, n_shots=1, n_real_eval=2, seed=1, threads_path=threads_path
    )
    assert len(real_eval) == 2
