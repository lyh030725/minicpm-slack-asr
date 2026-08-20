from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .dataset import discover_librispeech
from .model import MiniCPMSlackASR, ModelConfig
from .run import CHUNK_FIELDS, CONDITIONS, SAMPLE_FIELDS, _build_summary, _run_condition, _warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small end-to-end MiniCPM-o ASR sanity set after enabling the TTS-template "
            "<|tts_bos|> response prefix for both slack drafts and final text-only generation."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/LibriSpeech/test-clean"))
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tts-draft-sanity"),
        help="Directory for samples.csv, chunks.csv, diagnostics.jsonl, and summary.json.",
    )
    return parser.parse_args()


def _text_anomaly_flags(text: str) -> dict[str, bool]:
    lowered = text.lower().strip()
    return {
        "contains_soa_or_eoa": "<|soa" in lowered or "<|eoa" in lowered,
        "contains_nooutput": "<nooutput>" in lowered,
        "contains_missing_speech_preamble": "text of the given speech" in lowered,
        "is_got_it": lowered in {"got it", "got it."},
    }


def main() -> None:
    args = parse_args()
    samples = discover_librispeech(
        args.dataset_root,
        min_duration_s=0.0,
        max_samples=args.max_samples,
        shuffle=False,
        seed=args.seed,
    )
    if not samples:
        raise SystemExit(f"No LibriSpeech samples found under {args.dataset_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "samples.csv"
    chunks_path = args.output_dir / "chunks.csv"
    diagnostics_path = args.output_dir / "diagnostics.jsonl"
    summary_path = args.output_dir / "summary.json"

    runner = MiniCPMSlackASR(
        ModelConfig(
            model_id=args.model_id,
            revision=args.model_revision,
            seed=args.seed,
            attn_implementation=args.attn_implementation,
        )
    )
    _warmup(runner, samples[0])

    settings = runner.model_settings()
    print(
        "[sanity] slack/final response prefix uses <|tts_bos|>: "
        f"{settings.get('use_tts_template')}"
    )
    print("[sanity] final generate_audio=False; init_tts=False")

    rows: list[dict[str, Any]] = []
    anomaly_counts = {
        "draft_contains_soa_or_eoa": 0,
        "draft_contains_missing_speech_preamble": 0,
        "final_contains_soa_or_eoa": 0,
        "final_contains_missing_speech_preamble": 0,
        "final_contains_nooutput": 0,
        "final_is_got_it": 0,
    }

    # Sanity runs are always fresh. Do not resume old decoder/prefix results.
    with samples_path.open("w", encoding="utf-8", newline="") as sf, chunks_path.open(
        "w", encoding="utf-8", newline=""
    ) as cf, diagnostics_path.open("w", encoding="utf-8") as df:
        sample_writer = csv.DictWriter(sf, fieldnames=SAMPLE_FIELDS, extrasaction="ignore")
        chunk_writer = csv.DictWriter(cf, fieldnames=CHUNK_FIELDS, extrasaction="ignore")
        sample_writer.writeheader()
        chunk_writer.writeheader()

        for sample in samples:
            for condition in args.conditions:
                row = _run_condition(
                    runner,
                    sample,
                    condition,
                    realtime=args.realtime,
                    chunk_writer=chunk_writer,
                    compare_finalizers=False,
                )
                rows.append(row)
                sample_writer.writerow(row)

                draft = str(row.get("first_pass_draft") or "")
                final_text = str(row.get("final_transcript") or "")
                draft_flags = _text_anomaly_flags(draft)
                final_flags = _text_anomaly_flags(final_text)

                if condition == "slack_2pass":
                    anomaly_counts["draft_contains_soa_or_eoa"] += int(draft_flags["contains_soa_or_eoa"])
                    anomaly_counts["draft_contains_missing_speech_preamble"] += int(
                        draft_flags["contains_missing_speech_preamble"]
                    )
                anomaly_counts["final_contains_soa_or_eoa"] += int(final_flags["contains_soa_or_eoa"])
                anomaly_counts["final_contains_missing_speech_preamble"] += int(
                    final_flags["contains_missing_speech_preamble"]
                )
                anomaly_counts["final_contains_nooutput"] += int(final_flags["contains_nooutput"])
                anomaly_counts["final_is_got_it"] += int(final_flags["is_got_it"])

                diagnostic_record = {
                    "sample_id": sample.sample_id,
                    "condition": condition,
                    "reference": sample.reference,
                    "first_pass_draft": draft,
                    "draft_tokens": int(row["draft_tokens"]),
                    "draft_anomalies": draft_flags,
                    "final_transcript": final_text,
                    "wer": float(row["wer"]),
                    "final_anomalies": final_flags,
                    "finalizer": runner.final_diagnostics(),
                }
                df.write(json.dumps(diagnostic_record, ensure_ascii=False) + "\n")

                sf.flush()
                cf.flush()
                df.flush()

                print(
                    f"\n[{condition}] {sample.sample_id} WER={float(row['wer']):.4f} "
                    f"draft_tokens={row['draft_tokens']}"
                )
                if condition == "slack_2pass":
                    print(f"  draft={draft!r}")
                    print(f"  draft_anomalies={draft_flags}")
                print(f"  final={final_text!r}")
                print(f"  final_anomalies={final_flags}")

    summary = _build_summary(rows, min_duration=0.0)
    summary["selected_samples"] = len(samples)
    summary["requested_conditions"] = args.conditions
    summary["anomaly_counts"] = anomaly_counts
    summary["response_mode"] = {
        "slack_prefix": "non-thinking assistant + <|tts_bos|>",
        "final_decoder": "MiniCPM-o streaming_generate",
        "use_tts_template": True,
        "generate_audio": False,
        "init_tts": False,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"\n[saved] samples: {samples_path}")
    print(f"[saved] chunks: {chunks_path}")
    print(f"[saved] diagnostics: {diagnostics_path}")
    print(f"[saved] summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
