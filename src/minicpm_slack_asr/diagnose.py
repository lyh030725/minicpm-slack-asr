from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .dataset import discover_librispeech
from .model import MiniCPMSlackASR, ModelConfig
from .run import CHUNK_FIELDS, CONDITIONS, SAMPLE_FIELDS, _run_condition, _warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small MiniCPM-o ASR sanity set and save first-final-token top-k diagnostics."
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
        default=Path("results/decoder-sanity"),
        help="Directory for samples.csv, chunks.csv, and diagnostics.jsonl.",
    )
    return parser.parse_args()


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

    # Sanity runs are intentionally fresh: overwrite prior files so decoder versions
    # and top-k diagnostics cannot be mixed across code revisions.
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
                )
                diag = runner.final_diagnostics()

                sample_writer.writerow(row)
                diagnostic_record = {
                    "sample_id": sample.sample_id,
                    "condition": condition,
                    "reference": sample.reference,
                    "final_transcript": row["final_transcript"],
                    "wer": float(row["wer"]),
                    "draft_tokens": int(row["draft_tokens"]),
                    "first_pass_draft": row["first_pass_draft"],
                    "first_final_token": diag,
                }
                df.write(json.dumps(diagnostic_record, ensure_ascii=False) + "\n")

                sf.flush()
                cf.flush()
                df.flush()

                print(
                    f"\n[{condition}] {sample.sample_id} WER={float(row['wer']):.4f} "
                    f"draft_tokens={row['draft_tokens']} final={row['final_transcript']!r}"
                )
                print("  first-final-token diagnostics:")
                print(json.dumps(diag, indent=2, ensure_ascii=False))

    print(f"\n[saved] samples: {samples_path}")
    print(f"[saved] chunks: {chunks_path}")
    print(f"[saved] diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
