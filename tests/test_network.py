"""Layer B network-feature tests.

Fixtures write real Parquet stores (messages, recipients, threads) so the
SQL runs exactly as it does in production, the same pattern used for
identity.py and roles.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import pyarrow as pa
import pyarrow.parquet as pq

from thesis.data.network import (
    BROADCAST_THRESHOLD,
    MIN_LATENCY_OBSERVATIONS,
    _consecutive_turn_latencies,
    _latency_asymmetry,
    build_correspondence_graph,
    build_sender_features,
    compute_centrality,
)

BASE = datetime(2001, 5, 14, 9, 0, 0)


def _write(tmp_path: Path, name: str, rows: list[dict[str, object]], schema: pa.Schema) -> Path:
    out = tmp_path / name
    out.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / "part-00000.parquet")
    return out


MESSAGE_SCHEMA = pa.schema(
    [
        pa.field("message_uid", pa.string()),
        pa.field("from_addr", pa.string()),
        pa.field("n_recipients", pa.int32()),
        pa.field("date", pa.timestamp("ms")),
    ]
)
RECIPIENT_SCHEMA = pa.schema(
    [pa.field("message_uid", pa.string()), pa.field("address", pa.string())]
)
THREAD_SCHEMA = pa.schema(
    [
        pa.field("message_uid", pa.string()),
        pa.field("thread_id", pa.string()),
        pa.field("position_in_thread", pa.int32()),
        pa.field("thread_size", pa.int32()),
        pa.field("is_root", pa.bool_()),
        pa.field("is_conversation", pa.bool_()),
    ]
)


# --------------------------------------------------------- correspondence graph


def test_broadcast_messages_are_excluded_from_the_graph(tmp_path: Path) -> None:
    messages = [
        {"message_uid": "1", "from_addr": "a@x.com", "n_recipients": 2, "date": BASE},
        {
            "message_uid": "2",
            "from_addr": "a@x.com",
            "n_recipients": BROADCAST_THRESHOLD,
            "date": BASE,
        },
    ]
    recipients: list[dict[str, object]] = [
        {"message_uid": "1", "address": "b@x.com"},
        {"message_uid": "2", "address": "c@x.com"},
    ]
    m = _write(tmp_path, "messages", messages, MESSAGE_SCHEMA)
    r = _write(tmp_path, "recipients", recipients, RECIPIENT_SCHEMA)
    graph = build_correspondence_graph(str(m / "*.parquet"), r / "part-00000.parquet")
    assert graph.has_edge("a@x.com", "b@x.com")
    assert not graph.has_edge("a@x.com", "c@x.com")


def test_self_addressed_recipients_are_excluded(tmp_path: Path) -> None:
    messages = [{"message_uid": "1", "from_addr": "a@x.com", "n_recipients": 2, "date": BASE}]
    recipients: list[dict[str, object]] = [
        {"message_uid": "1", "address": "a@x.com"},  # cc'd self
        {"message_uid": "1", "address": "b@x.com"},
    ]
    m = _write(tmp_path, "messages", messages, MESSAGE_SCHEMA)
    r = _write(tmp_path, "recipients", recipients, RECIPIENT_SCHEMA)
    graph = build_correspondence_graph(str(m / "*.parquet"), r / "part-00000.parquet")
    assert not graph.has_edge("a@x.com", "a@x.com")
    assert graph.has_edge("a@x.com", "b@x.com")


def test_repeated_correspondence_increments_edge_weight(tmp_path: Path) -> None:
    messages = [
        {"message_uid": str(i), "from_addr": "a@x.com", "n_recipients": 1, "date": BASE}
        for i in range(3)
    ]
    recipients: list[dict[str, object]] = [
        {"message_uid": str(i), "address": "b@x.com"} for i in range(3)
    ]
    m = _write(tmp_path, "messages", messages, MESSAGE_SCHEMA)
    r = _write(tmp_path, "recipients", recipients, RECIPIENT_SCHEMA)
    graph = build_correspondence_graph(str(m / "*.parquet"), r / "part-00000.parquet")
    assert graph["a@x.com"]["b@x.com"]["weight"] == 3


# --------------------------------------------------------------------- centrality


def test_centrality_favours_connection_to_well_connected_nodes() -> None:
    graph = nx.DiGraph()
    # hub is well-connected; leaf is a dead end
    for i in range(5):
        graph.add_edge(f"n{i}", "hub", weight=1)
        graph.add_edge("hub", f"n{i}", weight=1)
    graph.add_edge("leaf", "n0", weight=1)
    centrality = compute_centrality(graph)
    assert centrality["hub"] > centrality["leaf"]


def test_centrality_falls_back_gracefully_on_a_trivial_graph() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1)
    centrality = compute_centrality(graph)
    assert set(centrality) == {"a", "b"}
    assert all(isinstance(v, float) for v in centrality.values())


# -------------------------------------------------------------- reply latency


def test_consecutive_turn_latency_uses_immediately_preceding_message(tmp_path: Path) -> None:
    messages = [
        {"message_uid": "1", "from_addr": "a@x.com", "n_recipients": 1, "date": BASE},
        {
            "message_uid": "2",
            "from_addr": "b@x.com",
            "n_recipients": 1,
            "date": BASE + timedelta(hours=2),
        },
        {
            "message_uid": "3",
            "from_addr": "a@x.com",
            "n_recipients": 1,
            "date": BASE + timedelta(hours=5),
        },
    ]
    threads = [
        {
            "message_uid": "1",
            "thread_id": "t1",
            "position_in_thread": 0,
            "thread_size": 3,
            "is_root": True,
            "is_conversation": True,
        },
        {
            "message_uid": "2",
            "thread_id": "t1",
            "position_in_thread": 1,
            "thread_size": 3,
            "is_root": False,
            "is_conversation": True,
        },
        {
            "message_uid": "3",
            "thread_id": "t1",
            "position_in_thread": 2,
            "thread_size": 3,
            "is_root": False,
            "is_conversation": True,
        },
    ]
    m = _write(tmp_path, "messages", messages, MESSAGE_SCHEMA)
    t = _write(tmp_path, "threads", threads, THREAD_SCHEMA)
    pairs = _consecutive_turn_latencies(str(m / "*.parquet"), t / "part-00000.parquet")

    assert ("b@x.com", "a@x.com", 2 * 3600.0) in pairs  # b replied to a after 2h
    assert ("a@x.com", "b@x.com", 3 * 3600.0) in pairs  # a replied to b after 3h


def test_non_conversation_threads_contribute_no_latency_pairs(tmp_path: Path) -> None:
    messages = [
        {"message_uid": "1", "from_addr": "a@x.com", "n_recipients": 1, "date": BASE},
        {
            "message_uid": "2",
            "from_addr": "a@x.com",
            "n_recipients": 1,
            "date": BASE + timedelta(hours=1),
        },
    ]
    threads = [
        {
            "message_uid": "1",
            "thread_id": "t1",
            "position_in_thread": 0,
            "thread_size": 2,
            "is_root": True,
            "is_conversation": False,
        },
        {
            "message_uid": "2",
            "thread_id": "t1",
            "position_in_thread": 1,
            "thread_size": 2,
            "is_root": False,
            "is_conversation": False,
        },
    ]
    m = _write(tmp_path, "messages", messages, MESSAGE_SCHEMA)
    t = _write(tmp_path, "threads", threads, THREAD_SCHEMA)
    pairs = _consecutive_turn_latencies(str(m / "*.parquet"), t / "part-00000.parquet")
    assert pairs == []


def test_asymmetry_is_none_below_the_minimum_observation_count() -> None:
    pairs = [("a", "b", 100.0), ("b", "a", 200.0)]  # 1 observation each way
    result = _latency_asymmetry(pairs)
    assert result["a"][0] is None
    assert result["a"][1] == 1  # n_out
    assert result["a"][2] == 1  # n_in


def test_asymmetry_sign_matches_the_powerful_direction() -> None:
    """Fast replies received, slow replies sent -> positive asymmetry."""
    pairs: list[tuple[str, str, float]] = []
    # "boss" receives fast replies (short latency FROM others TO boss)
    for _ in range(MIN_LATENCY_OBSERVATIONS):
        pairs.append(("junior", "boss", 60.0))
    # "boss" sends slow replies (long latency FROM boss TO others)
    for _ in range(MIN_LATENCY_OBSERVATIONS):
        pairs.append(("boss", "junior", 3600.0))
    result = _latency_asymmetry(pairs)
    asymmetry, n_out, n_in = result["boss"]
    assert asymmetry is not None
    assert asymmetry > 0
    assert n_out == MIN_LATENCY_OBSERVATIONS
    assert n_in == MIN_LATENCY_OBSERVATIONS


# --------------------------------------------------------------- end-to-end


def test_build_sender_features_end_to_end(tmp_path: Path) -> None:
    messages = [
        {"message_uid": "1", "from_addr": "a@x.com", "n_recipients": 1, "date": BASE},
        {
            "message_uid": "2",
            "from_addr": "b@x.com",
            "n_recipients": 1,
            "date": BASE + timedelta(hours=1),
        },
    ]
    recipients: list[dict[str, object]] = [
        {"message_uid": "1", "address": "b@x.com"},
        {"message_uid": "2", "address": "a@x.com"},
    ]
    threads = [
        {
            "message_uid": "1",
            "thread_id": "t1",
            "position_in_thread": 0,
            "thread_size": 2,
            "is_root": True,
            "is_conversation": True,
        },
        {
            "message_uid": "2",
            "thread_id": "t1",
            "position_in_thread": 1,
            "thread_size": 2,
            "is_root": False,
            "is_conversation": True,
        },
    ]
    m = _write(tmp_path, "messages", messages, MESSAGE_SCHEMA)
    r = _write(tmp_path, "recipients", recipients, RECIPIENT_SCHEMA)
    t = _write(tmp_path, "threads", threads, THREAD_SCHEMA)

    rows = build_sender_features(
        str(m / "*.parquet"), r / "part-00000.parquet", t / "part-00000.parquet"
    )
    addresses = {row.address for row in rows}
    assert {"a@x.com", "b@x.com"} <= addresses

    a_row = next(row for row in rows if row.address == "a@x.com")
    assert a_row.out_degree == 1
    assert a_row.in_degree == 1
    assert a_row.thread_initiation_rate == 1.0  # a's only conversation, a started it
