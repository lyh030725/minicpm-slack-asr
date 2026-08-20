from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OFFICIAL_MINICPMO45_TEST_CLEAN_WER_PCT = 1.40
OFFICIAL_REFERENCE_LABEL = "MiniCPM-o 4.5 official"


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
            f"| {OFFICIAL_REFERENCE_LABEL} | LibriSpeech test-clean (all) | — | "
            f"{OFFICIAL_MINICPMO45_TEST_CLEAN_WER_PCT:.2f} | — |"
        ),
        (
            "| Baseline (ours) | LibriSpeech test-clean, ≥10 s | "
            f"{_samples(baseline)} | {_wer_pct(baseline.get('micro_wer'))} | — |"
        ),
        (
            "| Slack 2-pass (ours) | LibriSpeech test-clean, ≥10 s | "
            f"{_samples(slack)} | {_wer_pct(slack.get('micro_wer'))} | {_pct(rel)} |"
        ),
        "",
        (
            "**Note.** The official MiniCPM-o 4.5 WER (1.40%) is reported on the full "
            "LibriSpeech test-clean set, whereas our baseline and Slack 2-pass results use "
            "only utterances with duration ≥10 s. Therefore, the primary apples-to-apples "
            "comparison is Baseline (ours) vs. Slack 2-pass (ours); the official number is "
            "included as a reference sanity check. Relative WER reduction is computed against "
            "Baseline (ours)."
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
            r"\caption{ASR performance on LibriSpeech test-clean. The official MiniCPM-o 4.5 result is shown for reference; our two methods are evaluated on the identical subset of utterances with duration $\geq 10$ s.}",
            r"\label{tab:slack_asr}",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Method & Evaluation set & Samples & WER (\%) $\downarrow$ & Rel. WER Red. (\%) $\uparrow$ \\",
            r"\midrule",
            (
                f"{OFFICIAL_REFERENCE_LABEL} & test-clean (all) & -- & "
                f"{OFFICIAL_MINICPMO45_TEST_CLEAN_WER_PCT:.2f} & -- \\\\"
            ),
            (
                "Baseline (ours) & test-clean ($\\geq 10$ s) & "
                f"{_samples(baseline)} & {_wer_pct(baseline.get('micro_wer'))} & -- \\\\"
            ),
            (
                "Slack 2-pass (ours) & test-clean ($\\geq 10$ s) & "
                f"{_samples(slack)} & {_wer_pct(slack.get('micro_wer'))} & {_pct(rel)} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize \textit{Note:} The official 1.40\% WER is reported on the full LibriSpeech test-clean set. Our baseline and Slack 2-pass use only utterances with duration $\geq 10$ s, so the primary controlled comparison is between those two rows. Relative WER reduction is measured against Baseline (ours).",
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
