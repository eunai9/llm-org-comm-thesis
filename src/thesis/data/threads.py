"""Reconstruct conversation threads.

Normally this is a two-tier job: follow ``In-Reply-To``/``References`` headers
first, then fall back to heuristics. **That first tier is unavailable here.**
Zero of the 254,359 unique messages carry either header -- the JavaMail export
that produced this corpus stripped them -- so the heuristic tier is the only
tier, and it must carry the whole load.

The heuristic links two messages when all three hold:

1. their subjects match after stripping ``Re:``/``Fw:`` prefixes,
2. they share at least two participants (sender plus recipients), and
3. they fall within a 30-day window of each other.

Linking is deliberately conservative, because threads become the stimuli fed
to the agent simulator: a thread that wrongly staples two conversations
together produces an incoherent prompt, which is far more damaging than a
long conversation split into two shorter ones. Two guards enforce that:

* Degenerate subjects (empty, or shorter than :data:`MIN_SUBJECT_CHARS`) never
  link. "fyi" recurs across hundreds of unrelated conversations.
* Subjects appearing more than :data:`MAX_GROUP_SIZE` times never link, since
  at that frequency the subject is a template ("Daily Report") rather than a
  conversation. Both cases yield singleton threads.

Precision is not assumed. ``sample_for_review`` writes a plain-text sample of
reconstructed threads for hand-checking, and the measured precision belongs in
the thesis alongside the counts.

Run with ``python -m thesis.data.threads``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from thesis.logging_setup import configure_logging, get_logger
from thesis.paths import INTERIM_DIR, MESSAGES_PARQUET_GLOB, RECIPIENTS_PARQUET, ensure_dirs

log = get_logger(__name__)

MIN_SHARED_PARTICIPANTS = 2
WINDOW_DAYS = 30
MIN_SUBJECT_CHARS = 4
MAX_GROUP_SIZE = 500

# "Re:", "Fw:", "Fwd:", plus the German and French forms that appear in this
# corpus, optionally with an Outlook counter such as "Re[2]:".
_REPLY_PREFIX = re.compile(
    r"^\s*(?:re|fw|fwd|aw|antw|tr|rv|sv|vs)\s*(?:\[\d+\])?\s*:\s*", re.IGNORECASE
)
_WHITESPACE = re.compile(r"\s+")

THREAD_SCHEMA = pa.schema(
    [
        pa.field("message_uid", pa.string(), nullable=False),
        pa.field("thread_id", pa.string(), nullable=False),
        pa.field("position_in_thread", pa.int32(), nullable=False),
        pa.field("thread_size", pa.int32(), nullable=False),
        pa.field("is_root", pa.bool_(), nullable=False),
        pa.field("n_distinct_senders", pa.int32(), nullable=False),
        pa.field("is_conversation", pa.bool_(), nullable=False),
        pa.field("reply_latency_seconds", pa.int64()),
    ]
)


def normalize_subject(subject: str) -> str:
    """Strip reply/forward prefixes and normalize for comparison."""
    text = subject
    while True:
        stripped = _REPLY_PREFIX.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    return _WHITESPACE.sub(" ", text).strip().lower()


def is_linkable_subject(subject_norm: str) -> bool:
    """Whether a subject is distinctive enough to justify linking on."""
    return len(subject_norm) >= MIN_SUBJECT_CHARS


class UnionFind:
    """Disjoint-set over integer ids, with path compression by rank."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self._rank[a] < self._rank[b]:
            a, b = b, a
        self._parent[b] = a
        if self._rank[a] == self._rank[b]:
            self._rank[a] += 1


@dataclass(frozen=True, slots=True)
class ThreadCandidate:
    """One message reduced to just what threading needs."""

    message_uid: str
    date: datetime
    subject_norm: str
    participants: frozenset[int]
    sender: int | None = None


def load_candidates(
    messages_glob: str = MESSAGES_PARQUET_GLOB,
    recipients_path: Path = RECIPIENTS_PARQUET,
) -> list[ThreadCandidate]:
    """Load messages and their participant sets, ready for linking.

    Addresses are interned to integers: at corpus scale the participant sets
    hold millions of references, and comparing small int sets is both faster
    and far lighter than comparing strings.
    """
    con = duckdb.connect()
    rows = con.execute(
        """
        SELECT m.message_uid, m.date, m.subject, m.from_addr,
               coalesce(list(r.address) FILTER (WHERE r.address IS NOT NULL), []) AS recipients
        FROM read_parquet(?) AS m
        LEFT JOIN read_parquet(?) AS r ON r.message_uid = m.message_uid
        WHERE m.date IS NOT NULL
        GROUP BY m.message_uid, m.date, m.subject, m.from_addr
        """,
        [messages_glob, str(recipients_path)],
    ).fetchall()
    con.close()

    interned: dict[str, int] = {}

    def intern(address: str) -> int:
        return interned.setdefault(address, len(interned))

    candidates: list[ThreadCandidate] = []
    for uid, date, subject, from_addr, recipients in rows:
        people = {intern(a) for a in recipients if a}
        sender = intern(from_addr) if from_addr else None
        if sender is not None:
            people.add(sender)
        candidates.append(
            ThreadCandidate(
                message_uid=uid,
                date=date,
                subject_norm=normalize_subject(subject or ""),
                participants=frozenset(people),
                sender=sender,
            )
        )
    return candidates


def _group_by_subject(candidates: Sequence[ThreadCandidate]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        if is_linkable_subject(candidate.subject_norm):
            groups.setdefault(candidate.subject_norm, []).append(index)
    return groups


def link_candidates(candidates: Sequence[ThreadCandidate]) -> UnionFind:
    """Union messages that satisfy subject, participant and time conditions."""
    union = UnionFind(len(candidates))
    window = timedelta(days=WINDOW_DAYS)

    for subject, indices in _group_by_subject(candidates).items():
        if len(indices) > MAX_GROUP_SIZE:
            log.debug("skipping template subject %r (%d messages)", subject, len(indices))
            continue

        ordered = sorted(indices, key=lambda i: candidates[i].date)
        for position, index in enumerate(ordered):
            current = candidates[index]
            for other_index in ordered[position + 1 :]:
                other = candidates[other_index]
                if other.date - current.date > window:
                    break  # ordered by date, so nothing later can qualify
                shared = current.participants & other.participants
                if len(shared) >= MIN_SHARED_PARTICIPANTS:
                    union.union(index, other_index)
    return union


def assemble(candidates: Sequence[ThreadCandidate], union: UnionFind) -> pa.Table:
    """Turn linked components into per-message thread positions."""
    components: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        components.setdefault(union.find(index), []).append(index)

    rows: list[dict[str, object]] = []
    for root, members in components.items():
        ordered = sorted(members, key=lambda i: candidates[i].date)
        thread_id = f"t{root:08d}"
        first_date = candidates[ordered[0]].date
        n_senders = len({candidates[i].sender for i in ordered if candidates[i].sender is not None})
        # A run of messages from a single sender under a recurring subject is a
        # newsletter or a system alert, not a conversation. The largest such
        # group in this corpus is 251 messages from one address over 127 days
        # ("Williams Energy News Live"). Flagging rather than dropping keeps
        # the rows available for descriptive work while stopping them reaching
        # the simulator as though they were dialogue.
        is_conversation = len(ordered) > 1 and n_senders > 1
        for position, index in enumerate(ordered):
            candidate = candidates[index]
            rows.append(
                {
                    "message_uid": candidate.message_uid,
                    "thread_id": thread_id,
                    "position_in_thread": position,
                    "thread_size": len(ordered),
                    "is_root": position == 0,
                    "n_distinct_senders": n_senders,
                    "is_conversation": is_conversation,
                    "reply_latency_seconds": (
                        None
                        if position == 0
                        else int((candidate.date - first_date).total_seconds())
                    ),
                }
            )
    return pa.Table.from_pylist(rows, schema=THREAD_SCHEMA)


def summarize(table: pa.Table) -> dict[str, object]:
    con = duckdb.connect()
    con.register("threads", table)
    row = con.execute("""
        SELECT count(*),
               count(DISTINCT thread_id),
               count(DISTINCT thread_id) FILTER (WHERE thread_size = 1),
               max(thread_size),
               avg(thread_size),
               count(DISTINCT thread_id) FILTER (WHERE is_conversation),
               count(DISTINCT thread_id) FILTER (WHERE is_conversation AND thread_size >= 3),
               count(DISTINCT thread_id) FILTER (WHERE NOT is_conversation AND thread_size > 1)
        FROM threads
        """).fetchone()
    sizes = con.execute("""
        SELECT thread_size, count(DISTINCT thread_id) AS n
        FROM threads WHERE thread_size > 1
        GROUP BY thread_size ORDER BY thread_size LIMIT 10
        """).fetchall()
    con.close()
    assert row is not None
    return {
        "messages": int(row[0]),
        "threads": int(row[1]),
        "singleton_threads": int(row[2]),
        "multi_message_threads": int(row[1]) - int(row[2]),
        "largest_thread": int(row[3]),
        "mean_thread_size": round(float(row[4]), 3),
        "conversations": int(row[5]),
        "conversations_with_3_or_more": int(row[6]),
        "single_sender_groups": int(row[7]),
        "size_distribution": {int(size): int(n) for size, n in sizes},
    }


def sample_for_review(
    candidates: Sequence[ThreadCandidate],
    table: pa.Table,
    out_path: Path,
    *,
    n_threads: int = 50,
    seed: int = 20260807,
) -> None:
    """Write multi-message threads as plain text for manual precision checking.

    Reconstruction quality cannot be asserted, only measured. Read this file,
    mark each thread correct or not, and report the resulting precision.
    """
    import random

    by_uid = {c.message_uid: c for c in candidates}
    con = duckdb.connect()
    con.register("threads", table)
    rows = con.execute("""
        SELECT thread_id, message_uid, position_in_thread, thread_size
        FROM threads WHERE is_conversation ORDER BY thread_id, position_in_thread
        """).fetchall()
    con.close()

    grouped: dict[str, list[tuple[str, int, int]]] = {}
    for thread_id, uid, position, size in rows:
        grouped.setdefault(thread_id, []).append((uid, int(position), int(size)))

    rng = random.Random(seed)
    chosen = rng.sample(sorted(grouped), min(n_threads, len(grouped)))

    lines: list[str] = [
        "Thread reconstruction -- manual review sample",
        "",
        "Mark each thread CORRECT if every message plausibly belongs to one",
        "conversation, or WRONG if unrelated conversations were merged.",
        "Report the resulting precision in the thesis.",
        "",
        "=" * 78,
    ]
    for thread_id in chosen:
        members = grouped[thread_id]
        lines.append(f"\nTHREAD {thread_id}  ({len(members)} messages)   [ ] CORRECT  [ ] WRONG")
        for uid, position, _ in members:
            candidate = by_uid[uid]
            people = len(candidate.participants)
            lines.append(
                f"  {position:>2}. {candidate.date:%Y-%m-%d %H:%M}  "
                f"({people} participants)  {candidate.subject_norm[:70]}"
            )
        lines.append("-" * 78)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_thread_ids(table: pa.Table) -> Iterator[str]:
    yield from table.column("thread_id").to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", default=MESSAGES_PARQUET_GLOB)
    parser.add_argument("--recipients", type=Path, default=RECIPIENTS_PARQUET)
    parser.add_argument("--out", type=Path, default=INTERIM_DIR / "threads.parquet")
    parser.add_argument("--review-sample", type=int, default=50)
    args = parser.parse_args()

    configure_logging()
    ensure_dirs()

    log.info("loading candidates")
    candidates = load_candidates(args.messages, args.recipients)
    log.info("loaded %d messages with a usable date", len(candidates))

    log.info("linking")
    union = link_candidates(candidates)
    table = assemble(candidates, union)
    pq.write_table(table, args.out, compression="zstd")

    stats = summarize(table)
    (INTERIM_DIR / "threads_report.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    review_path = INTERIM_DIR / "threads_review_sample.txt"
    sample_for_review(candidates, table, review_path, n_threads=args.review_sample)

    log.info("threads                %8d", stats["threads"])
    log.info("  singletons           %8d", stats["singleton_threads"])
    log.info("  multi-message        %8d", stats["multi_message_threads"])
    log.info("  largest              %8d", stats["largest_thread"])
    log.info("mean thread size       %8.3f", stats["mean_thread_size"])
    log.info("conversations          %8d  (>1 message AND >1 sender)", stats["conversations"])
    log.info(
        "  with >=3 messages    %8d  <- candidate simulator stimuli",
        stats["conversations_with_3_or_more"],
    )
    log.info(
        "single-sender groups   %8d  (newsletters/alerts, not dialogue)",
        stats["single_sender_groups"],
    )
    log.info("review sample -> %s  (hand-check this before trusting precision)", review_path)


if __name__ == "__main__":
    main()
