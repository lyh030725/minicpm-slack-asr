from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from .dataset import discover_librispeech
from .model import MiniCPMSlackASR, ModelConfig
from .run import CHUNK_FIELDS, CONDITIONS, _run_condition, _warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small MiniCPM-o ASR sanity set and print first-final-token top-k diagnostics."
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

    sink = io.StringIO()
    chunk_writer = csv.DictWriter(sink, fieldnames=CHUNK_FIELDS, extrasaction="ignore")

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
            print(
                f"\n[{condition}] {sample.sample_id} WER={float(row['wer']):.4f} "
                f"draft_tokens={row['draft_tokens']} final={row['final_transcript']!r}"
            )
            print("  first-final-token diagnostics:")
            print(json.dumps(diag, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
