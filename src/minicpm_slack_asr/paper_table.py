from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OFFICIAL_MINICPMO45_TEST_CLEAN_WER_PCT = 1.40
OFFICIAL_TEST_CLEAN_SAMPLES = 2620
OFFICIAL_REFERENCE_LABEL = "MiniCPM-o 4.5 official"
LATEX_ROW_END = r"\\"


def _wer_pct(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value) * 100.0:.2f}"


def _pct(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):.2f}"


def _samples(condition: dict[str, Any] | None) -> str:
    if not condition:
        return "—"
    value = condition.get("samples")
    return "—" if value is None else str(value)


def build_markdown_table(summary: dict[str, Any]) -> str:
    conditions = summary.get("conditions", {})
    baseline = conditions.get("baseline", {})
    slack = conditions.get("slack_2pass", {})
    rel = summary.get("relative_wer_reduction_pct")

    lines = [
        "# ASR Results Table",
        "",
        "| Method | Evaluation set | Samples | WER (%) ↓ | Rel. WER Reduction (%) ↑ |",
        "|---|---|---:|---:|---:|",
        (
            f"| {OFFICIAL_REFERENCE_LABEL} | LibriSpeech test-clean | "
            f"{OFFICIAL_TEST_CLEAN_SAMPLES} | {OFFICIAL_MINICPMO45_TEST_CLEAN_WER_PCT:.2f} | — |"
        ),
        (
            "| Baseline (ours) | LibriSpeech test-clean | "
            f"{_samples(baseline)} | {_wer_pct(baseline.get('micro_wer'))} | — |"
        ),
        (
            "| Slack 2-pass (ours) | LibriSpeech test-clean | "
            f"{_samples(slack)} | {_wer_pct(slack.get('micro_wer'))} | {_pct(rel)} |"
        ),
        "",
        (
            "**Note.** All rows use the standard LibriSpeech test-clean split. The official "
            "MiniCPM-o 4.5 value (1.40% WER) is included as a reference; the primary controlled "
            "comparison is still Baseline (ours) vs. Slack 2-pass (ours), because they use the "
            "same inference implementation and differ only in the listening-slack two-pass method. "
            "Relative WER reduction is computed against Baseline (ours)."
        ),
        "",
    ]
    return "\n".join(lines)


def build_latex_table(summary: dict[str, Any]) -> str:
    conditions = summary.get("conditions", {})
    baseline = conditions.get("baseline", {})
    slack = conditions.get("slack_2pass", {})
    rel = summary.get("relative_wer_reduction_pct")

    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{ASR performance on the full LibriSpeech test-clean split.}",
            r"\label{tab:slack_asr}",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            f"Method & Evaluation set & Samples & WER (\\%) $\\downarrow$ & Rel. WER Red. (\\%) $\\uparrow$ {LATEX_ROW_END}",
            r"\midrule",
            (
                f"{OFFICIAL_REFERENCE_LABEL} & test-clean & {OFFICIAL_TEST_CLEAN_SAMPLES} & "
                f"{OFFICIAL_MINICPMO45_TEST_CLEAN_WER_PCT:.2f} & -- {LATEX_ROW_END}"
            ),
            (
                "Baseline (ours) & test-clean & "
                f"{_samples(baseline)} & {_wer_pct(baseline.get('micro_wer'))} & -- {LATEX_ROW_END}"
            ),
            (
                "Slack 2-pass (ours) & test-clean & "
                f"{_samples(slack)} & {_wer_pct(slack.get('micro_wer'))} & {_pct(rel)} {LATEX_ROW_END}"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize \textit{Note:} All rows use the standard LibriSpeech test-clean split. The official 1.40\% WER is shown as a reference. The controlled comparison is Baseline (ours) versus Slack 2-pass (ours), which share the same inference implementation. Relative WER reduction is measured against Baseline (ours).",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )


def write_paper_tables(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "paper_table.md"
    tex_path = output_dir / "paper_table.tex"
    md_path.write_text(build_markdown_table(summary), encoding="utf-8")
    tex_path.write_text(build_latex_table(summary), encoding="utf-8")
    return md_path, tex_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-style ASR result tables from summary.json.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.summary.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    md_path, tex_path = write_paper_tables(summary, args.output_dir)
    print(f"[paper-table] markdown={md_path}")
    print(f"[paper-table] latex={tex_path}")


if __name__ == "__main__":
    main()
