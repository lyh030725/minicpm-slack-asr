from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .dataset import discover_librispeech


def write_manifest(samples, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "path", "duration_s", "sample_rate", "frames", "reference"],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "path": str(sample.path),
                    "duration_s": f"{sample.duration_s:.6f}",
                    "sample_rate": sample.sample_rate,
                    "frames": sample.frames,
                    "reference": sample.reference,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a LibriSpeech test-clean manifest.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--min-duration", type=float, default=0.0, help="0 selects the complete split.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all qualifying samples")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = discover_librispeech(
        args.dataset_root,
        min_duration_s=args.min_duration,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    write_manifest(samples, args.output)
    print(f"[manifest] selected={len(samples)} duration>={args.min_duration}s output={args.output}")


if __name__ == "__main__":
    main()
