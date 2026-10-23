from thesis.analysis.judge_blindness import DEFAULT_JUDGE_MODEL, output_suffix, plot_subtitle


def test_default_judge_keeps_original_file_names() -> None:
    assert output_suffix(DEFAULT_JUDGE_MODEL) == ""


def test_other_judges_get_their_own_file_names() -> None:
    assert output_suffix("openai/gpt-oss-20b@low") == "_gpt_oss_20b"
    assert output_suffix("openai/gpt-oss-120b@low") == "_gpt_oss_120b"
    assert output_suffix("qwen/qwen3.8-27b") == "_qwen3_8_27b"


def test_subtitle_reports_both_gaps_and_n() -> None:
    summary = {
        "reply only": {"n": 100, "p_value": 0.321},
        "with context": {"n": 100, "p_value": 0.384},
    }
    subtitle = plot_subtitle(summary, judge="gpt-oss-20b")
    assert "gpt-oss-20b" in subtitle
    assert "p=0.321" in subtitle
    assert "p=0.384" in subtitle
    assert "n=100" in subtitle
