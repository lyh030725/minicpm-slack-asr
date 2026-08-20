from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .dataset import AudioSample, discover_librispeech, load_audio, streaming_chunks
from .manifest import write_manifest
from .model import MiniCPMSlackASR, ModelConfig
from .wer import WERCounts, aggregate_counts, compute_wer


UNIT_BUDGET_S = 1.0
CONDITIONS = ("baseline", "slack_2pass")

SAMPLE_FIELDS = [
    "sample_id",
    "duration_s",
    "condition",
    "reference",
    "first_pass_draft",
    "final_transcript",
    "first_pass_wer",
    "wer",
    "substitutions",
    "deletions",
    "insertions",
    "reference_words",
    "audio_prefill_ms",
    "draft_compute_ms",
    "final_prompt_prefix_ms",
    "final_decode_ms",
    "deadline_miss_chunks",
    "draft_tokens",
]

CHUNK_FIELDS = [
    "sample_id",
    "condition",
    "unit_idx",
    "is_last_chunk",
    "valid_audio_samples",
    "audio_start_s",
    "audio_end_s",
    "start_lag_ms",
    "prefill_ms",
    "slack_before_draft_ms",
    "draft_started",
    "draft_prefix_tokens",
    "draft_prefix_ms",
    "draft_decode_ms",
    "draft_total_ms",
    "draft_new_tokens",
    "draft_new_text",
    "draft_so_far",
    "deadline_remaining_ms",
    "deadline_miss",
    "kv_cache_before_draft",
    "kv_cache_after_restore",
]


def _sleep_until(deadline_s: float) -> float:
    while True:
        now = time.perf_counter()
        remaining = deadline_s - now
        if remaining <= 0:
            return now
        if remaining > 0.004:
            time.sleep(remaining - 0.002)
        else:
            time.sleep(0)


def _wer_to_row(counts: WERCounts) -> dict[str, Any]:
    return {
        "wer": counts.wer,
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "reference_words": counts.reference_words,
    }


def _collect_environment(args: argparse.Namespace, runner: MiniCPMSlackASR) -> dict[str, Any]:
    env: dict[str, Any] = {
        "created_unix_s": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
        "target_runpod_image": "runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404",
        "args": {},
        "model_settings": runner.model_settings(),
        "speech_generation": "disabled: init_tts=False; project calls model.llm only for decoding",
    }
    for key, value in vars(args).items():
        if isinstance(value, Path):
            env["args"][key] = str(value)
        else:
            env["args"][key] = value

    try:
        import transformers

        env["transformers"] = transformers.__version__
    except Exception:
        pass

    try:
        from huggingface_hub import model_info

        env["model_resolved_sha"] = model_info(args.model_id, revision=args.model_revision).sha
    except Exception as exc:
        env["model_resolved_sha_error"] = repr(exc)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        }
    return env


def _warmup(runner: MiniCPMSlackASR, sample: AudioSample) -> None:
    print(f"[warmup] {sample.sample_id}")
    audio = load_audio(sample.path)
    chunks = list(streaming_chunks(audio))
    runner.reset_for_sample(f"warmup-{sample.sample_id}")
    draft_ids: list[int] = []
    for unit_idx, chunk, _valid, is_last in chunks[:2]:
        runner.prefill_audio(chunk, is_last_chunk=is_last and len(chunks[:2]) == len(chunks))
        if not is_last:
            step = runner.draft_in_slack(draft_ids, deadline_s=time.perf_counter() + 0.5)
            draft_ids.extend(step.new_token_ids)
    torch.cuda.synchronize()
    runner.model.reset_session(reset_token2wav_cache=False)
    torch.cuda.empty_cache()


def _run_condition(
    runner: MiniCPMSlackASR,
    sample: AudioSample,
    condition: str,
    *,
    realtime: bool,
    chunk_writer: csv.DictWriter,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(condition)

    audio = load_audio(sample.path)
    chunks = list(streaming_chunks(audio))
    runner.reset_for_sample(f"{condition}-{sample.sample_id}")

    draft_ids: list[int] = []
    total_prefill_ms = 0.0
    total_draft_ms = 0.0
    deadline_miss_chunks = 0

    origin = time.perf_counter()
    for unit_idx, chunk, valid_samples, is_last_chunk in chunks:
        scheduled_start = origin + unit_idx * UNIT_BUDGET_S
        if condition == "slack_2pass" and realtime:
            actual_start = _sleep_until(scheduled_start)
            deadline = origin + (unit_idx + 1) * UNIT_BUDGET_S
            start_lag_ms = (actual_start - scheduled_start) * 1000.0
        else:
            actual_start = time.perf_counter()
            deadline = actual_start + UNIT_BUDGET_S
            start_lag_ms = 0.0

        prefill_ms = runner.prefill_audio(chunk, is_last_chunk=is_last_chunk)
        total_prefill_ms += prefill_ms
        after_prefill = time.perf_counter()
        slack_before_ms = (deadline - after_prefill) * 1000.0

        draft_started = False
        draft_prefix_tokens = 0
        draft_prefix_ms = 0.0
        draft_decode_ms = 0.0
        draft_total_ms = 0.0
        draft_new_ids: list[int] = []
        draft_new_text = ""
        kv_before = runner.cache_length()
        kv_after = kv_before

        # There is no next-audio deadline after the last chunk. The second pass starts
        # immediately, so the last chunk is intentionally not used for draft decoding.
        if condition == "slack_2pass" and not is_last_chunk:
            draft_started = slack_before_ms >= runner.config.min_slack_ms
            step = runner.draft_in_slack(draft_ids, deadline_s=deadline)
            draft_prefix_tokens = step.prefix_tokens
            draft_prefix_ms = step.prefix_ms
            draft_decode_ms = step.decode_ms
            draft_total_ms = step.total_ms
            draft_new_ids = step.new_token_ids
            draft_new_text = step.new_text
            kv_before = step.cache_len_before
            kv_after = step.cache_len_after_restore
            draft_ids.extend(draft_new_ids)
            total_draft_ms += draft_total_ms
            deadline_remaining_ms = step.deadline_remaining_ms
            deadline_miss = step.deadline_miss
        else:
            deadline_remaining_ms = (deadline - time.perf_counter()) * 1000.0
            deadline_miss = deadline_remaining_ms < 0.0

        if deadline_miss:
            deadline_miss_chunks += 1

        draft_so_far = runner.tokenizer.decode(draft_ids, skip_special_tokens=True).strip()
        chunk_writer.writerow(
            {
                "sample_id": sample.sample_id,
                "condition": condition,
                "unit_idx": unit_idx,
                "is_last_chunk": is_last_chunk,
                "valid_audio_samples": valid_samples,
                "audio_start_s": unit_idx,
                "audio_end_s": min((unit_idx + 1), sample.duration_s),
                "start_lag_ms": start_lag_ms,
                "prefill_ms": prefill_ms,
                "slack_before_draft_ms": slack_before_ms,
                "draft_started": draft_started,
                "draft_prefix_tokens": draft_prefix_tokens,
                "draft_prefix_ms": draft_prefix_ms,
                "draft_decode_ms": draft_decode_ms,
                "draft_total_ms": draft_total_ms,
                "draft_new_tokens": len(draft_new_ids),
                "draft_new_text": draft_new_text,
                "draft_so_far": draft_so_far,
                "deadline_remaining_ms": deadline_remaining_ms,
                "deadline_miss": deadline_miss,
                "kv_cache_before_draft": kv_before,
                "kv_cache_after_restore": kv_after,
            }
        )

    first_pass_draft = runner.tokenizer.decode(draft_ids, skip_special_tokens=True).strip()
    final_text, final_prefix_ms, final_decode_ms = runner.finalize(
        draft_text=first_pass_draft if condition == "slack_2pass" else None
    )
    counts = compute_wer(sample.reference, final_text)
    first_pass_counts = compute_wer(sample.reference, first_pass_draft) if condition == "slack_2pass" else None

    row = {
        "sample_id": sample.sample_id,
        "duration_s": sample.duration_s,
        "condition": condition,
        "reference": sample.reference,
        "first_pass_draft": first_pass_draft if condition == "slack_2pass" else "",
        "final_transcript": final_text,
        "first_pass_wer": first_pass_counts.wer if first_pass_counts is not None else "",
        **_wer_to_row(counts),
        "audio_prefill_ms": total_prefill_ms,
        "draft_compute_ms": total_draft_ms,
        "final_prompt_prefix_ms": final_prefix_ms,
        "final_decode_ms": final_decode_ms,
        "deadline_miss_chunks": deadline_miss_chunks,
        "draft_tokens": len(draft_ids),
    }
    return row


def _load_existing_samples(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not path.exists():
        return [], set()
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    completed = {(row["sample_id"], row["condition"]) for row in rows}
    return rows, completed


def _counts_from_row(row: dict[str, Any]) -> WERCounts:
    return WERCounts(
        substitutions=int(float(row["substitutions"])),
        deletions=int(float(row["deletions"])),
        insertions=int(float(row["insertions"])),
        reference_words=int(float(row["reference_words"])),
    )


def _build_summary(rows: list[dict[str, Any]], *, min_duration: float) -> dict[str, Any]:
    if min_duration <= 0:
        selection = "LibriSpeech ASR test-clean, complete standard split"
    else:
        selection = f"LibriSpeech ASR test-clean, duration >= {min_duration:g} seconds"

    summary: dict[str, Any] = {
        "selection": selection,
        "min_duration_s": min_duration,
        "conditions": {},
    }
    condition_counts: dict[str, WERCounts] = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row.get("condition") == condition]
        counts = aggregate_counts([_counts_from_row(row) for row in subset]) if subset else WERCounts(0, 0, 0, 0)
        condition_counts[condition] = counts
        sample_wers = [float(row["wer"]) for row in subset if row.get("wer") not in (None, "")]
        summary["conditions"][condition] = {
            "samples": len(subset),
            "micro_wer": counts.wer if counts.reference_words else None,
            "mean_sample_wer": sum(sample_wers) / len(sample_wers) if sample_wers else None,
            "substitutions": counts.substitutions,
            "deletions": counts.deletions,
            "insertions": counts.insertions,
            "reference_words": counts.reference_words,
            "deadline_miss_chunks": sum(int(float(row.get("deadline_miss_chunks", 0) or 0)) for row in subset),
            "draft_tokens": sum(int(float(row.get("draft_tokens", 0) or 0)) for row in subset),
        }

    baseline = condition_counts["baseline"]
    slack = condition_counts["slack_2pass"]
    if baseline.reference_words and slack.reference_words and baseline.wer > 0 and baseline.reference_words == slack.reference_words:
        summary["relative_wer_reduction_pct"] = (baseline.wer - slack.wer) / baseline.wer * 100.0
        summary["absolute_wer_change"] = slack.wer - baseline.wer
    else:
        summary["relative_wer_reduction_pct"] = None
        summary["absolute_wer_change"] = None
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-free two-pass ASR using MiniCPM-o 4.5 LLM listening slack."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/LibriSpeech/test-clean"))
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Inclusive lower bound in seconds. Default 0 evaluates the complete test-clean split.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means every qualifying sample.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/test-clean-all-nonthinking"))
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
        help="Default runs both baseline and slack_2pass for WER comparison.",
    )
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--model-id", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--max-final-tokens", type=int, default=512)
    parser.add_argument("--max-draft-tokens-per-chunk", type=int, default=12)
    parser.add_argument("--max-draft-prefix-tokens", type=int, default=256)
    parser.add_argument("--min-slack-ms", type=float, default=50.0)
    parser.add_argument("--initial-decode-guard-ms", type=float, default=25.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_librispeech(
        args.dataset_root,
        min_duration_s=args.min_duration,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not samples:
        raise SystemExit(
            f"No LibriSpeech FLAC files with duration >= {args.min_duration}s under {args.dataset_root}. "
            "Run scripts/download_librispeech.sh first."
        )
    write_manifest(samples, args.output_dir / "manifest.csv")

    if args.min_duration <= 0:
        print(f"[data] selected={len(samples)} utterances from the complete LibriSpeech test-clean split")
    else:
        print(f"[data] selected={len(samples)} test-clean utterances with duration>={args.min_duration}s")
    print(f"[data] max_samples={args.max_samples} (0 means all)")
    print(f"[run] conditions={args.conditions} realtime={args.realtime}")
    print("[model] speech decoding disabled: init_tts=False, LLM text tokens only")

    runner = MiniCPMSlackASR(
        ModelConfig(
            model_id=args.model_id,
            revision=args.model_revision,
            seed=args.seed,
            attn_implementation=args.attn_implementation,
            max_final_tokens=args.max_final_tokens,
            max_draft_tokens_per_chunk=args.max_draft_tokens_per_chunk,
            max_draft_prefix_tokens=args.max_draft_prefix_tokens,
            min_slack_ms=args.min_slack_ms,
            initial_decode_guard_ms=args.initial_decode_guard_ms,
        )
    )

    with (args.output_dir / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(_collect_environment(args, runner), f, indent=2, ensure_ascii=False)

    if args.warmup:
        _warmup(runner, samples[0])

    samples_csv = args.output_dir / "samples.csv"
    chunks_csv = args.output_dir / "chunks.csv"
    if args.resume:
        rows, completed = _load_existing_samples(samples_csv)
        sample_mode = "a" if samples_csv.exists() else "w"
        chunk_mode = "a" if chunks_csv.exists() else "w"
    else:
        rows, completed = [], set()
        sample_mode = "w"
        chunk_mode = "w"

    with samples_csv.open(sample_mode, encoding="utf-8", newline="") as sf, chunks_csv.open(
        chunk_mode, encoding="utf-8", newline=""
    ) as cf:
        sample_writer = csv.DictWriter(sf, fieldnames=SAMPLE_FIELDS, extrasaction="ignore")
        chunk_writer = csv.DictWriter(cf, fieldnames=CHUNK_FIELDS, extrasaction="ignore")
        if sample_mode == "w":
            sample_writer.writeheader()
        if chunk_mode == "w":
            chunk_writer.writeheader()

        progress_desc = "LibriSpeech test-clean" if args.min_duration <= 0 else f"LibriSpeech >={args.min_duration:g}s"
        for sample in tqdm(samples, desc=progress_desc):
            for condition in args.conditions:
                key = (sample.sample_id, condition)
                if key in completed:
                    continue
                try:
                    row = _run_condition(
                        runner,
                        sample,
                        condition,
                        realtime=args.realtime,
                        chunk_writer=chunk_writer,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    raise
                sample_writer.writerow(row)
                sf.flush()
                cf.flush()
                rows.append(row)
                completed.add(key)
                print(
                    f"\n[{condition}] {sample.sample_id} WER={float(row['wer']):.4f} "
                    f"draft_tokens={row['draft_tokens']} final={row['final_transcript']!r}"
                )

    # Reload so resumed rows and numeric text have one consistent representation.
    final_rows, _ = _load_existing_samples(samples_csv)
    summary = _build_summary(final_rows, min_duration=args.min_duration)
    summary["selected_samples"] = len(samples)
    summary["requested_conditions"] = args.conditions
    summary["max_samples"] = args.max_samples
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
