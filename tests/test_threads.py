"""Thread reconstruction tests.

Emphasis is on the failure this module is designed to avoid: merging two
unrelated conversations. A merged thread becomes an incoherent prompt for the
agent simulator, which is worse than a conversation split in two.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pyarrow as pa
import pytest

from thesis.data.threads import (
    MAX_GROUP_SIZE,
    ThreadCandidate,
    UnionFind,
    assemble,
    is_linkable_subject,
    link_candidates,
    normalize_subject,
    summarize,
)

BASE = datetime(2001, 5, 14, 9, 0, 0)


def make(
    uid: str,
    subject: str,
    people: set[int],
    *,
    offset_days: float = 0.0,
) -> ThreadCandidate:
    return ThreadCandidate(
        message_uid=uid,
        date=BASE + timedelta(days=offset_days),
        subject_norm=normalize_subject(subject),
        participants=frozenset(people),
    )


def thread_of(table: pa.Table, uid: str) -> str:
    cols = table.to_pydict()
    return str(cols["thread_id"][cols["message_uid"].index(uid)])


# --------------------------------------------------------------- subjects


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Re: Gas storage", "gas storage"),
        ("RE: RE: Gas storage", "gas storage"),
        ("Fwd: Re: Gas storage", "gas storage"),
        ("Re[2]: Gas storage", "gas storage"),
        ("AW: Gas storage", "gas storage"),
        ("  Gas   storage  ", "gas storage"),
        ("Gas storage", "gas storage"),
    ],
)
def test_normalize_subject_strips_prefixes(raw: str, expected: str) -> None:
    assert normalize_subject(raw) == expected


def test_normalize_subject_keeps_re_inside_words() -> None:
    """'Report' must not lose its leading 'Re'."""
    assert normalize_subject("Report on storage") == "report on storage"


def test_short_and_empty_subjects_are_not_linkable() -> None:
    assert not is_linkable_subject("")
    assert not is_linkable_subject("fyi")
    assert is_linkable_subject("gas storage")


# --------------------------------------------------------------- union-find


def test_union_find_merges_transitively() -> None:
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(0) != uf.find(3)


# ------------------------------------------------------------------ linking


def test_links_reply_sharing_participants_within_window() -> None:
    candidates = [
        make("a", "Gas storage schedule", {1, 2}),
        make("b", "Re: Gas storage schedule", {1, 2}, offset_days=1),
    ]
    table = assemble(candidates, link_candidates(candidates))
    assert thread_of(table, "a") == thread_of(table, "b")


def test_does_not_link_across_the_time_window() -> None:
    candidates = [
        make("a", "Gas storage schedule", {1, 2}),
        make("b", "Re: Gas storage schedule", {1, 2}, offset_days=45),
    ]
    table = assemble(candidates, link_candidates(candidates))
    assert thread_of(table, "a") != thread_of(table, "b")


def test_does_not_link_on_a_single_shared_participant() -> None:
    """One person in common is coincidence, not a conversation."""
    candidates = [
        make("a", "Gas storage schedule", {1, 2}),
        make("b", "Re: Gas storage schedule", {1, 9}, offset_days=1),
    ]
    table = assemble(candidates, link_candidates(candidates))
    assert thread_of(table, "a") != thread_of(table, "b")


def test_generic_subject_never_merges_unrelated_conversations() -> None:
    """'fyi' recurs across hundreds of unrelated conversations."""
    candidates = [
        make("a", "FYI", {1, 2}),
        make("b", "FYI", {1, 2}, offset_days=1),
    ]
    table = assemble(candidates, link_candidates(candidates))
    assert thread_of(table, "a") != thread_of(table, "b")


def test_template_subjects_are_not_linked() -> None:
    """A subject appearing thousands of times is a template, not a thread."""
    candidates = [
        make(f"m{i}", "Daily position report", {1, 2}, offset_days=i * 0.1)
        for i in range(MAX_GROUP_SIZE + 5)
    ]
    table = assemble(candidates, link_candidates(candidates))
    assert len({thread_of(table, c.message_uid) for c in candidates}) == len(candidates)


def test_transitive_linking_builds_one_thread() -> None:
    candidates = [
        make("a", "Storage schedule", {1, 2}),
        make("b", "Re: Storage schedule", {1, 2}, offset_days=5),
        make("c", "Re: Storage schedule", {1, 2}, offset_days=25),
    ]
    table = assemble(candidates, link_candidates(candidates))
    ids = {thread_of(table, uid) for uid in ("a", "b", "c")}
    assert len(ids) == 1


def test_two_distinct_conversations_stay_separate() -> None:
    candidates = [
        make("a1", "Storage schedule", {1, 2}),
        make("a2", "Re: Storage schedule", {1, 2}, offset_days=1),
        make("b1", "Risk review", {3, 4}, offset_days=2),
        make("b2", "Re: Risk review", {3, 4}, offset_days=3),
    ]
    table = assemble(candidates, link_candidates(candidates))
    assert thread_of(table, "a1") == thread_of(table, "a2")
    assert thread_of(table, "b1") == thread_of(table, "b2")
    assert thread_of(table, "a1") != thread_of(table, "b1")


# ---------------------------------------------------------------- assembly


def test_positions_ordering_and_latency() -> None:
    candidates = [
        make("second", "Re: Storage schedule", {1, 2}, offset_days=2),
        make("first", "Storage schedule", {1, 2}),
    ]
    cols = assemble(candidates, link_candidates(candidates)).to_pydict()
    order = {uid: i for i, uid in enumerate(cols["message_uid"])}

    assert cols["position_in_thread"][order["first"]] == 0
    assert cols["is_root"][order["first"]] is True
    assert cols["reply_latency_seconds"][order["first"]] is None

    assert cols["position_in_thread"][order["second"]] == 1
    assert cols["is_root"][order["second"]] is False
    assert cols["reply_latency_seconds"][order["second"]] == 2 * 86400
    assert cols["thread_size"][order["second"]] == 2


def test_isolated_message_becomes_a_singleton_thread() -> None:
    candidates = [make("lonely", "Unique subject line", {1, 2})]
    cols = assemble(candidates, link_candidates(candidates)).to_pydict()
    assert cols["thread_size"] == [1]
    assert cols["is_root"] == [True]


def test_summarize_counts_threads() -> None:
    candidates = [
        make("a", "Storage schedule", {1, 2}),
        make("b", "Re: Storage schedule", {1, 2}, offset_days=1),
        make("c", "Unrelated topic here", {5, 6}, offset_days=1),
    ]
    stats = summarize(assemble(candidates, link_candidates(candidates)))
    assert stats["messages"] == 3
    assert stats["threads"] == 2
    assert stats["singleton_threads"] == 1
    assert stats["multi_message_threads"] == 1
    assert stats["largest_thread"] == 2


# ------------------------------------------------- conversation vs newsletter


def sender_make(
    uid: str, subject: str, sender: int, others: set[int], days: float
) -> ThreadCandidate:
    return ThreadCandidate(
        message_uid=uid,
        date=BASE + timedelta(days=days),
        subject_norm=normalize_subject(subject),
        participants=frozenset({sender, *others}),
        sender=sender,
    )


def test_single_sender_run_is_not_a_conversation() -> None:
    """251 messages from one address under a recurring subject is a newsletter.

    Real case from this corpus: "Williams Energy News Live", 251 messages from
    a single sender over 127 days. It must not reach the simulator as dialogue.
    """
    candidates = [
        sender_make(f"n{i}", "Energy news live daily", 1, {2, 3}, i * 2.0) for i in range(6)
    ]
    cols = assemble(candidates, link_candidates(candidates)).to_pydict()
    assert cols["thread_size"][0] > 1
    assert set(cols["n_distinct_senders"]) == {1}
    assert not any(cols["is_conversation"])


def test_two_way_exchange_is_a_conversation() -> None:
    candidates = [
        sender_make("a", "Storage schedule", 1, {2}, 0),
        sender_make("b", "Re: Storage schedule", 2, {1}, 1),
    ]
    cols = assemble(candidates, link_candidates(candidates)).to_pydict()
    assert set(cols["n_distinct_senders"]) == {2}
    assert all(cols["is_conversation"])


def test_singleton_is_never_a_conversation() -> None:
    candidates = [sender_make("only", "Unique subject line", 1, {2}, 0)]
    cols = assemble(candidates, link_candidates(candidates)).to_pydict()
    assert cols["is_conversation"] == [False]


def test_summarize_separates_conversations_from_newsletters() -> None:
    candidates = [
        sender_make("a", "Storage schedule", 1, {2}, 0),
        sender_make("b", "Re: Storage schedule", 2, {1}, 1),
        sender_make("n1", "Daily energy digest", 9, {2, 3}, 0),
        sender_make("n2", "Daily energy digest", 9, {2, 3}, 1),
    ]
    stats = summarize(assemble(candidates, link_candidates(candidates)))
    assert stats["conversations"] == 1
    assert stats["single_sender_groups"] == 1
