from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .dataset import discover_librispeech
from .model import MiniCPMSlackASR, ModelConfig
from .run import CHUNK_FIELDS, CONDITIONS, _run_condition, _warmup
from .wer import compute_wer


COMPARISON_FIELDS = [
    "sample_id",
    "condition",
    "decoder",
    "reference",
    "first_pass_draft",
    "final_transcript",
    "wer",
    "substitutions",
    "deletions",
    "insertions",
    "reference_words",
    "draft_tokens",
    "audio_prefill_ms",
    "draft_compute_ms",
    "final_prompt_prefix_ms",
    "final_decode_ms",
    "error",
    "same_pre_final_state",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream each MiniCPM-o ASR sample once, snapshot the exact pre-final state, then compare "
            "the custom final decoder against upstream streaming_generate(generate_audio=False)."
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
    parser.add_argument("--diagnostic-top-k", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/decoder-ab-sanity"),
        help="Directory for samples.csv, chunks.csv, and diagnostics.jsonl.",
    )
    return parser.parse_args()


def _comparison_row(
    *,
    base_row: dict[str, Any],
    decoder: str,
    result: dict[str, Any],
    same_pre_final_state: bool,
) -> dict[str, Any]:
    transcript = str(result.get("text") or "")
    counts = compute_wer(str(base_row["reference"]), transcript)
    return {
        "sample_id": base_row["sample_id"],
        "condition": base_row["condition"],
        "decoder": decoder,
        "reference": base_row["reference"],
        "first_pass_draft": base_row["first_pass_draft"],
        "final_transcript": transcript,
        "wer": counts.wer,
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "reference_words": counts.reference_words,
        "draft_tokens": base_row["draft_tokens"],
        "audio_prefill_ms": base_row["audio_prefill_ms"],
        "draft_compute_ms": base_row["draft_compute_ms"],
        "final_prompt_prefix_ms": result.get("prompt_prefix_ms") or 0.0,
        "final_decode_ms": result.get("decode_ms") or 0.0,
        "error": result.get("error") or "",
        "same_pre_final_state": same_pre_final_state,
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

    runner = MiniCPMSlackASR(
        ModelConfig(
            model_id=args.model_id,
            revision=args.model_revision,
            seed=args.seed,
            attn_implementation=args.attn_implementation,
            diagnostic_top_k=args.diagnostic_top_k,
        )
    )
    _warmup(runner, samples[0])

    # Each condition is streamed only once. Immediately before final decoding the
    # runner snapshots MiniCPM-o state, runs the custom finalizer, restores the exact
    # state, and then runs upstream text-only streaming_generate. This keeps audio KV,
    # draft, instruction, and streaming timing identical across the A/B finalizers.
    with samples_path.open("w", encoding="utf-8", newline="") as sf, chunks_path.open(
        "w", encoding="utf-8", newline=""
    ) as cf, diagnostics_path.open("w", encoding="utf-8") as df:
        sample_writer = csv.DictWriter(sf, fieldnames=COMPARISON_FIELDS, extrasaction="ignore")
        chunk_writer = csv.DictWriter(cf, fieldnames=CHUNK_FIELDS, extrasaction="ignore")
        sample_writer.writeheader()
        chunk_writer.writeheader()

        for sample in samples:
            for condition in args.conditions:
                base_row = _run_condition(
                    runner,
                    sample,
                    condition,
                    realtime=args.realtime,
                    chunk_writer=chunk_writer,
                    compare_finalizers=True,
                )
                comparison = base_row.pop("_finalizer_comparison")
                same_state = bool(comparison.get("same_pre_final_state"))

                custom_row = _comparison_row(
                    base_row=base_row,
                    decoder="custom",
                    result=comparison["custom"],
                    same_pre_final_state=same_state,
                )
                official_row = _comparison_row(
                    base_row=base_row,
                    decoder="official_text_only",
                    result=comparison["official_text_only"],
                    same_pre_final_state=same_state,
                )
                sample_writer.writerow(custom_row)
                sample_writer.writerow(official_row)

                diagnostic_record = {
                    "sample_id": sample.sample_id,
                    "condition": condition,
                    "reference": sample.reference,
                    "first_pass_draft": base_row["first_pass_draft"],
                    "draft_tokens": int(base_row["draft_tokens"]),
                    "pre_final_cache_len": comparison.get("pre_final_cache_len"),
                    "restored_cache_len": comparison.get("restored_cache_len"),
                    "same_pre_final_state": same_state,
                    "custom": comparison["custom"],
                    "official_text_only": comparison["official_text_only"],
                    "custom_wer": custom_row["wer"],
                    "official_text_only_wer": official_row["wer"],
                }
                df.write(json.dumps(diagnostic_record, ensure_ascii=False) + "\n")

                sf.flush()
                cf.flush()
                df.flush()

                print(f"\n[{condition}] {sample.sample_id} same_pre_final_state={same_state}")
                print(
                    f"  A custom:   WER={float(custom_row['wer']):.4f} "
                    f"final={custom_row['final_transcript']!r} error={custom_row['error']!r}"
                )
                print(
                    f"  B official: WER={float(official_row['wer']):.4f} "
                    f"final={official_row['final_transcript']!r} error={official_row['error']!r}"
                )
                print("  custom first-final-token diagnostics:")
                print(json.dumps(comparison["custom"].get("diagnostics", {}), indent=2, ensure_ascii=False))

    print(f"\n[saved] A/B samples: {samples_path}")
    print(f"[saved] shared streaming chunks: {chunks_path}")
    print(f"[saved] full A/B diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
