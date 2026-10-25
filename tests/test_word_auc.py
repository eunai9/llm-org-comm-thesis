import pandas as pd

from thesis.analysis.word_auc import clean_lines, keep_line, word_auc_summary


def test_header_and_routing_lines_are_removed() -> None:
    assert not keep_line("To: Kim Ward/HOU/ECT@ECT")
    assert not keep_line("cc: someone")
    assert not keep_line("Subject: Re: deal")
    assert not keep_line("Mark Taylor/HOU/ECT on 12/01/2000")


def test_signature_lines_are_removed() -> None:
    assert not keep_line("Enron North America Corp.")
    assert not keep_line("1400 Smith Street")
    assert not keep_line("Houston, Texas 77002")
    assert not keep_line("Phone: 713-853-1234")
    assert not keep_line("(713) 853-5678")
    assert not keep_line("jane.doe@enron.com, john.roe@enron.com")


def test_ordinary_writing_is_kept() -> None:
    assert keep_line("Please review the attached letter.")
    assert keep_line("This covers Enron's Domestic Affiliates.")
    assert keep_line("")


def test_clean_lines_counts_removed_lines() -> None:
    text, removed = clean_lines(
        "Thanks, will do.\nMark\nEnron North America Corp.\n1400 Smith Street"
    )
    assert text == "Thanks, will do.\nMark"
    assert removed == 2


def test_summary_separates_obviously_different_texts() -> None:
    real = [f"Real note {i} about the gas contract and pipeline volumes." for i in range(20)]
    generated = [f"I'll review and let you know {i}." for i in range(20)]
    frame = pd.DataFrame({"real_reply_body_recleaned": real, "generated_reply": generated})
    summary = word_auc_summary(frame)
    assert summary["n_pairs"] == 20
    assert summary["auc_full_text"] > 0.9
    assert summary["auc_same_length"] > 0.9
    assert 0.0 <= summary["auc_length_only"] <= 1.0


def test_summary_skips_pairs_without_a_generated_reply() -> None:
    frame = pd.DataFrame(
        {
            "real_reply_body_recleaned": [f"Real note {i} about gas." for i in range(12)],
            "generated_reply": [f"I'll check {i}." for i in range(11)] + [None],
        }
    )
    assert word_auc_summary(frame)["n_pairs"] == 11
