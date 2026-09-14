"""Tests for the deterministic parts of q1_real: direction from ranks, the
strict and loose selections, the self-email rule, the reply flag, and the
helpers that summarize fits. Hand-built fixtures only, no real data. The
models themselves are tested in test_hierarchy.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from thesis.analysis.q1_real import (
    RANK_COL,
    ContrastRow,
    assign_direction,
    build_analysis_frames,
    compare_with_simulator,
    direction_counts,
    drop_self_recipients,
    imperative_agreement,
    mark_replies,
    select_loose,
    select_strict,
    standardized_levels,
)


def _row(
    uid: str,
    to_addr: str,
    recipient_rank: float,
    recipient_id: float,
    *,
    from_addr: str = "boss@enron.com",
    sender_id: int = 1,
    sender_rank: int = 3,
    n_to: int = 1,
    n_cc: int = 0,
    n_bcc: int = 0,
    in_frame: bool = True,
) -> dict[str, object]:
    """One To row, shaped like load_to_rows' output. NaN rank and id mean the
    recipient is not on the employee list."""
    return {
        "message_uid": uid,
        "from_addr": from_addr,
        "n_to": n_to,
        "n_cc": n_cc,
        "n_bcc": n_bcc,
        "sender_rank": sender_rank,
        "sender_id": sender_id,
        "to_addr": to_addr,
        "recipient_rank": recipient_rank,
        "recipient_id": recipient_id,
        "in_frame": in_frame,
    }


NAN = float("nan")


def test_direction_follows_the_rank_difference() -> None:
    sender = pd.Series([3, 3, 3])
    recipient = pd.Series([5, 3, 1])
    assert assign_direction(sender, recipient).tolist() == ["up", "lateral", "down"]


def test_strict_keeps_one_ranked_to_recipient_and_nothing_else() -> None:
    rows = pd.DataFrame(
        [
            _row("up", "ceo@enron.com", 6, 10),
            _row("peer", "peer@enron.com", 3, 11),
            _row("down", "junior@enron.com", 1, 12),
            _row("unranked", "someone@enron.com", NAN, NAN),
            _row("cc", "peer@enron.com", 3, 11, n_cc=1),
            _row("bcc", "peer@enron.com", 3, 11, n_bcc=2),
            _row("two_to", "peer@enron.com", 3, 11, n_to=2),
            _row("two_to", "ceo@enron.com", 6, 10, n_to=2),
        ]
    )
    strict = select_strict(rows).set_index("message_uid")
    assert sorted(strict.index) == ["down", "peer", "up"]
    assert strict.loc["up", "direction"] == "up"
    assert strict.loc["peer", "direction"] == "lateral"
    assert strict.loc["down", "direction"] == "down"


def test_loose_needs_every_to_ranked_and_one_shared_direction() -> None:
    rows = pd.DataFrame(
        [
            _row("both_up", "ceo@enron.com", 6, 10, n_to=2),
            _row("both_up", "vp@enron.com", 5, 13, n_to=2),
            _row("mixed", "ceo@enron.com", 6, 10, n_to=2),
            _row("mixed", "junior@enron.com", 1, 12, n_to=2),
            _row("one_unranked", "ceo@enron.com", 6, 10, n_to=2),
            _row("one_unranked", "someone@enron.com", NAN, NAN, n_to=2),
            _row("with_cc", "peer@enron.com", 3, 11, n_cc=3),
        ]
    )
    loose = select_loose(rows).set_index("message_uid")
    assert sorted(loose.index) == ["both_up", "with_cc"]
    assert loose.loc["both_up", "direction"] == "up"
    assert loose.loc["with_cc", "direction"] == "lateral"


def test_self_email_to_a_second_address_is_dropped_only_by_person() -> None:
    """The sender writing to their own other address looks lateral by
    address. It must go once identity is checked by person."""
    rows = pd.DataFrame(
        [
            _row("literal_self", "boss@enron.com", 3, 1),
            _row("other_alias", "boss.two@enron.com", 3, 1),
            _row("real_peer", "peer@enron.com", 3, 11),
        ]
    )
    by_address = select_strict(drop_self_recipients(rows, by_person=False))
    by_person = select_strict(drop_self_recipients(rows, by_person=True))
    assert sorted(by_address["message_uid"]) == ["other_alias", "real_peer"]
    assert by_person["message_uid"].tolist() == ["real_peer"]


def test_a_self_alias_among_other_recipients_does_not_block_loose() -> None:
    """Only the self row goes. The email still counts if the rest agree."""
    rows = pd.DataFrame(
        [
            _row("m", "boss.two@enron.com", 3, 1, n_to=2),
            _row("m", "ceo@enron.com", 6, 10, n_to=2),
        ]
    )
    assert select_loose(rows).empty
    loose = select_loose(drop_self_recipients(rows))
    assert loose["direction"].tolist() == ["up"]


def test_unranked_recipient_survives_the_self_rule_with_nullable_ids() -> None:
    """load_to_rows gives recipient_id as nullable Int32. An unranked
    recipient's id is <NA>, and the self rule must keep that row. If it is
    dropped, an email with one unranked To looks fully ranked."""
    rows = pd.DataFrame(
        [
            _row("m", "ceo@enron.com", 6, 10, n_to=2),
            _row("m", "someone@enron.com", NAN, NAN, n_to=2),
        ]
    ).astype({"recipient_rank": "Int32", "recipient_id": "Int32", "sender_id": "int32"})
    kept = drop_self_recipients(rows)
    assert len(kept) == 2
    assert select_loose(kept).empty


def test_in_frame_flag_is_carried_through_selection() -> None:
    rows = pd.DataFrame(
        [
            _row("inside", "peer@enron.com", 3, 11),
            _row("outside", "peer@enron.com", 3, 11, in_frame=False),
        ]
    )
    strict = select_strict(rows).set_index("message_uid")
    assert bool(strict.loc["inside", "in_frame"])
    assert not bool(strict.loc["outside", "in_frame"])


def test_reply_flag_is_true_only_for_non_first_messages() -> None:
    selected = pd.DataFrame({"message_uid": ["first", "reply", "unthreaded"]})
    threads = pd.DataFrame({"message_uid": ["first", "reply"], "is_root": [True, False]})
    marked = mark_replies(selected, threads)
    assert marked["is_reply"].tolist() == [False, True, False]


def test_direction_counts_report_senders_across_directions() -> None:
    selected = pd.DataFrame(
        {
            "sender_id": [1, 1, 1, 2, 2, 3],
            "sender_rank": [3, 3, 3, 1, 1, 6],
            "direction": ["up", "lateral", "down", "up", "up", "down"],
        }
    )
    counts = direction_counts(selected)
    assert counts["n_emails"] == 6
    assert counts["emails_by_direction"] == {"down": 2, "lateral": 1, "up": 3}
    assert counts["n_senders"] == 3
    assert counts["n_senders_2plus_directions"] == 1
    assert counts["n_senders_all_3_directions"] == 1
    assert counts["emails_by_sender_rank_and_direction"]["1"] == {"down": 0, "lateral": 0, "up": 2}


def test_standardized_levels_average_over_each_rows_own_rank() -> None:
    coefficients = {
        "Intercept": 0.0,
        "direction[T.up]": 1.0,
        "direction[T.down]": -1.0,
        f"{RANK_COL}[T.2]": 2.0,
    }
    frame = pd.DataFrame({RANK_COL: ["1", "2"]})
    linear = standardized_levels(coefficients, frame, covariate=RANK_COL, logistic=False)
    assert linear == {"down": 0.0, "lateral": 1.0, "up": 2.0}

    logistic = standardized_levels(coefficients, frame, covariate=RANK_COL, logistic=True)

    def expit(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    assert logistic["up"] == pytest.approx((expit(1.0) + expit(3.0)) / 2)
    assert logistic["lateral"] == pytest.approx((expit(0.0) + expit(2.0)) / 2)


def test_standardized_levels_without_covariate_is_intercept_plus_contrast() -> None:
    coefficients = {"Intercept": -1.0, "direction[T.up]": 0.4, "direction[T.down]": 0.2}
    levels = standardized_levels(
        coefficients, pd.DataFrame({"x": [0, 0, 0]}), covariate=None, logistic=True
    )
    assert levels["up"] == pytest.approx(1.0 / (1.0 + np.exp(0.6)))


def test_imperative_agreement_counts_matches_and_missing() -> None:
    recomputed = pd.DataFrame(
        {"message_uid": ["a", "b", "c"], "imperative_ratio": [0.5, 0.25, 0.0]}
    )
    stored = pd.DataFrame({"message_uid": ["a", "b"], "imperative_ratio": [0.5, 0.75]})
    agreement = imperative_agreement(recomputed, stored)
    assert agreement["n_compared"] == 2
    assert agreement["n_missing_from_stored"] == 1
    assert agreement["share_within_0.001"] == 0.5
    assert agreement["max_abs_diff"] == 0.5


def test_analysis_frames_join_direction_and_rank_onto_sentences() -> None:
    selected = pd.DataFrame(
        {
            "message_uid": ["m1", "m2"],
            "from_addr": ["a@enron.com", "b@enron.com"],
            "sender_id": [1, 2],
            "sender_rank": [2, 4],
            "direction": ["up", "down"],
            "in_frame": [True, True],
            "is_reply": [False, True],
        }
    )
    emails = pd.DataFrame(
        {"message_uid": ["m1", "m2"], "imperative_ratio": [0.5, 0.0], "hedge_rate": [0.0, 0.1]}
    )
    sentences = pd.DataFrame(
        {"message_uid": ["m1", "m1", "m2"], "sentence_index": [0, 1, 0], "is_imperative": [1, 0, 0]}
    )
    email_frame, sentence_frame = build_analysis_frames(selected, emails, sentences)
    assert email_frame[RANK_COL].tolist() == ["2", "4"]
    assert sentence_frame["direction"].tolist() == ["up", "up", "down"]
    assert sentence_frame["is_reply"].tolist() == [False, False, True]


def test_simulator_comparison_gives_no_difference_for_an_identical_contrast() -> None:
    same = ContrastRow("strict", "is_imperative", "up", 0.395, 0.046, 100, 10)
    bigger = ContrastRow("strict", "is_imperative", "down", 2.0, 1e-6, 100, 10)
    comparison = compare_with_simulator([same, bigger])
    assert comparison["is_imperative:up"]["difference"] == 0.0
    assert comparison["is_imperative:up"]["difference_p"] == pytest.approx(1.0)
    assert comparison["is_imperative:down"]["difference"] == pytest.approx(2.0 - 0.163)
    assert comparison["is_imperative:down"]["difference_p"] < 0.001
