from minicpm_slack_asr.paper_table import build_latex_table, build_markdown_table


def _summary():
    return {
        "conditions": {
            "baseline": {"samples": 100, "micro_wer": 0.0200},
            "slack_2pass": {"samples": 100, "micro_wer": 0.0150},
        },
        "relative_wer_reduction_pct": 25.0,
    }


def test_markdown_table_contains_official_and_ours():
    table = build_markdown_table(_summary())
    assert "MiniCPM-o 4.5 official" in table
    assert "1.40" in table
    assert "2.00" in table
    assert "1.50" in table
    assert "25.00" in table
    assert "≥10 s" in table


def test_latex_table_contains_paper_rows():
    table = build_latex_table(_summary())
    assert "\\begin{table}" in table
    assert "\\toprule" in table
    assert "MiniCPM-o 4.5 official" in table
    assert "test-clean ($\\geq 10$ s)" in table
    assert "25.00" in table
