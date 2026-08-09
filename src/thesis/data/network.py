"""Layer B of the power score: network and structural features, every message.

No LLM calls, no linguistic parsing -- these come from the shape of who
talks to whom, and how fast. The corpus is small enough for networkx to
handle directly (~20k addresses; the plan called this "trivial" and it is:
seconds, not minutes).

Five per-sender measures:

- **Centrality** (degree and eigenvector, on the sender -> recipient graph):
  eigenvector centrality captures being connected to *well-connected* people,
  not just many people -- the standard structural proxy for organizational
  influence.
- **Broadcast rate**: share of a sender's messages sent to a large
  distribution list. A high broadcast rate is a different communication
  style (announcements) from one-to-one correspondence, not necessarily
  higher power, so it is reported as its own feature rather than folded into
  centrality.
- **Thread-initiation rate**: share of a sender's real conversations (see
  ``threads.py`` for what counts as one -- a newsletter is not a
  conversation) that they started rather than joined.
- **Last-word rate**: share of a sender's real conversations where their
  message was the final one.
- **Reply-latency asymmetry**: the classic power signal from organizational
  communication research -- people reply quickly to their superiors and
  slowly to their subordinates. Computed as
  ``median(latency of the sender's own replies) - median(latency of replies
  the sender received)``, both restricted to consecutive turns within a real
  conversation. Positive means "this person's incoming mail gets answered
  faster than they answer other people's" -- the powerful direction.

**Important limitation, stated rather than glossed over:** because this
corpus has no ``In-Reply-To``/``References`` headers (see ``threads.py``),
"replying to" a message is approximated as *the very next message,
chronologically, in the same reconstructed conversation*. In a thread with
more than two participants this is not always the literal message being
answered. It is the best signal the data supports, not a claim of certainty.

Run with ``python -m thesis.data.network``.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import duckdb
import networkx as nx
import pyarrow as pa
import pyarrow.parquet as pq

from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import (
    INTERIM_DIR,
    MESSAGES_PARQUET_GLOB,
    RECIPIENTS_PARQUET,
    ensure_dirs,
)

log = get_logger(__name__)

# A message to at least this many recipients is a broadcast, not
# correspondence. Chosen as a round, defensible cut rather than tuned to any
# outcome; revisit only with a reason independent of the power-score result.
BROADCAST_THRESHOLD = 10

# A sender needs at least this many latency observations in a direction
# before its median is reported, so a person with two lucky/unlucky replies
# doesn't produce a noisy extreme value.
MIN_LATENCY_OBSERVATIONS = 3


@dataclass(frozen=True, slots=True)
class SenderNetworkFeatures:
    """Layer B features for one sender address."""

    address: str
    out_degree: int
    in_degree: int
    eigenvector_centrality: float
    broadcast_rate: float
    thread_initiation_rate: float
    last_word_rate: float
    reply_latency_asymmetry: float | None
    n_latency_out: int
    n_latency_in: int


def build_correspondence_graph(messages_glob: str, recipients_path: object) -> nx.DiGraph:
    """Build the sender -> recipient graph, weighted by message count.

    Excludes broadcasts: a 200-recipient announcement should not count as
    200 edges of equal weight to 200 one-to-one exchanges, or the graph
    would be dominated by mailing-list behaviour rather than correspondence.
    """
    con = duckdb.connect()
    rows = con.execute(
        """
        SELECT m.from_addr, r.address, count(*) AS weight
        FROM read_parquet(?) AS m
        JOIN read_parquet(?) AS r ON r.message_uid = m.message_uid
        WHERE m.from_addr IS NOT NULL
          AND m.n_recipients < ?
          AND r.address != m.from_addr
        GROUP BY m.from_addr, r.address
        """,
        [messages_glob, str(recipients_path), BROADCAST_THRESHOLD],
    ).fetchall()
    con.close()

    graph = nx.DiGraph()
    for sender, recipient, weight in rows:
        graph.add_edge(sender, recipient, weight=weight)
    return graph


def compute_centrality(graph: nx.DiGraph) -> dict[str, float]:
    """Eigenvector centrality, falling back to degree centrality if it fails.

    Eigenvector centrality can fail to converge on graphs with certain
    disconnected structures. A fallback is used rather than letting the
    whole pipeline abort on a single non-convergent component -- this is
    logged, not silent, so it is visible in the run output if it happens.
    """
    try:
        raw = nx.eigenvector_centrality(graph, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        log.warning("eigenvector centrality did not converge; falling back to degree centrality")
        raw = nx.degree_centrality(graph)
    # networkx has no type stubs, so its return value is Any; construct the
    # dict explicitly rather than returning it straight through.
    return {str(node): float(score) for node, score in raw.items()}


def _broadcast_rates(messages_glob: str) -> dict[str, float]:
    con = duckdb.connect()
    rows = con.execute(
        """
        SELECT from_addr,
               avg(CASE WHEN n_recipients >= ? THEN 1.0 ELSE 0.0 END) AS broadcast_rate
        FROM read_parquet(?)
        WHERE from_addr IS NOT NULL
        GROUP BY from_addr
        """,
        [BROADCAST_THRESHOLD, messages_glob],
    ).fetchall()
    con.close()
    return {addr: float(rate) for addr, rate in rows}


def _thread_participation_rates(
    messages_glob: str, threads_path: object
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-sender thread-initiation rate and last-word rate.

    Restricted to real conversations (``is_conversation``) -- a sender's
    newsletter, which is always "position 0" and always "the last word" by
    construction, would otherwise inflate both rates meaninglessly.
    """
    con = duckdb.connect()
    rows = con.execute(
        """
        SELECT m.from_addr,
               avg(CASE WHEN t.is_root THEN 1.0 ELSE 0.0 END) AS initiation_rate,
               avg(CASE WHEN t.position_in_thread = t.thread_size - 1 THEN 1.0 ELSE 0.0 END)
                   AS last_word_rate
        FROM read_parquet(?) AS t
        JOIN read_parquet(?) AS m ON m.message_uid = t.message_uid
        WHERE t.is_conversation AND m.from_addr IS NOT NULL
        GROUP BY m.from_addr
        """,
        [str(threads_path), messages_glob],
    ).fetchall()
    con.close()
    initiation = {addr: float(v) for addr, v, _ in rows}
    last_word = {addr: float(v) for addr, _, v in rows}
    return initiation, last_word


def _consecutive_turn_latencies(
    messages_glob: str, threads_path: object
) -> list[tuple[str, str, float]]:
    """(replier, repliee, latency_seconds) for every consecutive pair in a
    real conversation, ordered by position within the thread.

    See the module docstring's limitation note: "repliee" is the sender of
    the immediately preceding message, not necessarily who the reply-writer
    had in mind, since this corpus has no reply-to headers to confirm it.
    """
    con = duckdb.connect()
    rows = con.execute(
        """
        SELECT t.thread_id, t.position_in_thread, m.from_addr, m.date
        FROM read_parquet(?) AS t
        JOIN read_parquet(?) AS m ON m.message_uid = t.message_uid
        WHERE t.is_conversation
        ORDER BY t.thread_id, t.position_in_thread
        """,
        [str(threads_path), messages_glob],
    ).fetchall()
    con.close()

    pairs: list[tuple[str, str, float]] = []
    previous: tuple[str, str, object] | None = None  # (thread_id, addr, date)
    for thread_id, _position, addr, date in rows:
        if previous is not None and previous[0] == thread_id and addr and previous[1]:
            latency = (date - previous[2]).total_seconds()
            if latency >= 0:  # guard against same-second or malformed ordering
                pairs.append((addr, previous[1], latency))
        previous = (thread_id, addr, date)
    return pairs


def _latency_asymmetry(
    pairs: list[tuple[str, str, float]],
) -> dict[str, tuple[float | None, int, int]]:
    """Per-address (asymmetry, n_out_observations, n_in_observations)."""
    out_latencies: dict[str, list[float]] = {}
    in_latencies: dict[str, list[float]] = {}
    for replier, repliee, latency in pairs:
        out_latencies.setdefault(replier, []).append(latency)
        in_latencies.setdefault(repliee, []).append(latency)

    result: dict[str, tuple[float | None, int, int]] = {}
    for address in set(out_latencies) | set(in_latencies):
        out_vals = out_latencies.get(address, [])
        in_vals = in_latencies.get(address, [])
        if len(out_vals) >= MIN_LATENCY_OBSERVATIONS and len(in_vals) >= MIN_LATENCY_OBSERVATIONS:
            asymmetry = statistics.median(out_vals) - statistics.median(in_vals)
        else:
            asymmetry = None
        result[address] = (asymmetry, len(out_vals), len(in_vals))
    return result


SCHEMA = pa.schema(
    [
        pa.field("address", pa.string(), nullable=False),
        pa.field("out_degree", pa.int32(), nullable=False),
        pa.field("in_degree", pa.int32(), nullable=False),
        pa.field("eigenvector_centrality", pa.float32(), nullable=False),
        pa.field("broadcast_rate", pa.float32(), nullable=False),
        pa.field("thread_initiation_rate", pa.float32(), nullable=False),
        pa.field("last_word_rate", pa.float32(), nullable=False),
        pa.field("reply_latency_asymmetry", pa.float32()),
        pa.field("n_latency_out", pa.int32(), nullable=False),
        pa.field("n_latency_in", pa.int32(), nullable=False),
    ]
)


def build_sender_features(
    messages_glob: str = MESSAGES_PARQUET_GLOB,
    recipients_path: object = RECIPIENTS_PARQUET,
    threads_path: object = INTERIM_DIR / "threads.parquet",
) -> list[SenderNetworkFeatures]:
    graph = build_correspondence_graph(messages_glob, recipients_path)
    centrality = compute_centrality(graph)
    broadcast = _broadcast_rates(messages_glob)
    initiation, last_word = _thread_participation_rates(messages_glob, threads_path)
    pairs = _consecutive_turn_latencies(messages_glob, threads_path)
    latency = _latency_asymmetry(pairs)

    addresses = set(graph.nodes) | set(broadcast) | set(initiation) | set(latency)
    rows: list[SenderNetworkFeatures] = []
    for address in addresses:
        asymmetry, n_out, n_in = latency.get(address, (None, 0, 0))
        rows.append(
            SenderNetworkFeatures(
                address=address,
                out_degree=graph.out_degree(address) if address in graph else 0,
                in_degree=graph.in_degree(address) if address in graph else 0,
                eigenvector_centrality=centrality.get(address, 0.0),
                broadcast_rate=broadcast.get(address, 0.0),
                thread_initiation_rate=initiation.get(address, 0.0),
                last_word_rate=last_word.get(address, 0.0),
                reply_latency_asymmetry=asymmetry,
                n_latency_out=n_out,
                n_latency_in=n_in,
            )
        )
    return rows


def _row(f: SenderNetworkFeatures) -> dict[str, object]:
    return {
        "address": f.address,
        "out_degree": f.out_degree,
        "in_degree": f.in_degree,
        "eigenvector_centrality": f.eigenvector_centrality,
        "broadcast_rate": f.broadcast_rate,
        "thread_initiation_rate": f.thread_initiation_rate,
        "last_word_rate": f.last_word_rate,
        "reply_latency_asymmetry": f.reply_latency_asymmetry,
        "n_latency_out": f.n_latency_out,
        "n_latency_in": f.n_latency_in,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default=MESSAGES_PARQUET_GLOB)
    parser.add_argument("--recipients", default=str(RECIPIENTS_PARQUET))
    parser.add_argument("--threads", default=str(INTERIM_DIR / "threads.parquet"))
    parser.add_argument("--out", default=str(INTERIM_DIR / "sender_network_features.parquet"))
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    rows = build_sender_features(args.messages, args.recipients, args.threads)
    table = pa.Table.from_pylist([_row(r) for r in rows], schema=SCHEMA)
    pq.write_table(table, args.out, compression="zstd")

    with_asymmetry = sum(1 for r in rows if r.reply_latency_asymmetry is not None)
    log.info("sender addresses            %8d", len(rows))
    log.info(
        "with reply-latency signal   %8d  (>= %d obs each direction)",
        with_asymmetry,
        MIN_LATENCY_OBSERVATIONS,
    )
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
