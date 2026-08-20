from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from .dataset import discover_librispeech
from .model import MiniCPMSlackASR, ModelConfig, build_final_prompt
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
            "upstream text-only generation with use_tts_template=False versus True. Both variants "
            "keep generate_audio=False, so this isolates only the <|tts_bos|> assistant-prefix effect."
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
        default=Path("results/tts-prefix-ab-sanity"),
        help="Directory for samples.csv, chunks.csv, and diagnostics.jsonl.",
    )
    return parser.parse_args()


def _run_official_text_only(
    runner: MiniCPMSlackASR,
    *,
    draft_text: str | None,
    use_tts_template: bool,
) -> dict[str, Any]:
    """Run MiniCPM-o upstream text-only generation from the runner's current pre-final state."""
    result: dict[str, Any] = {
        "text": "",
        "prompt_prefix_ms": None,
        "decode_ms": None,
        "error": None,
        "generate_audio": False,
        "use_tts_template": use_tts_template,
        "enable_thinking": False,
        "do_sample": False,
    }

    try:
        final_prompt = build_final_prompt(draft_text)
        final_prompt_ms = runner._prefill_raw_text(final_prompt) if final_prompt else 0.0

        torch.cuda.synchronize()
        start = time.perf_counter()
        pieces: list[str] = []
        iterator = runner.model.streaming_generate(
            session_id=runner.session_id,
            generate_audio=False,
            max_new_tokens=runner.config.max_final_tokens,
            enable_thinking=False,
            use_tts_template=use_tts_template,
            do_sample=False,
        )
        for item in iterator:
            if not isinstance(item, tuple) or not item:
                continue
            text_chunk = item[0]
            if text_chunk:
                pieces.append(str(text_chunk))
        torch.cuda.synchronize()

        result["text"] = "".join(pieces).strip()
        result["prompt_prefix_ms"] = final_prompt_ms
        result["decode_ms"] = (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        result["error"] = repr(exc)

    return result


def _compare_tts_prefix_finalizers(runner: MiniCPMSlackASR, *, draft_text: str | None) -> dict[str, Any]:
    """A/B only the upstream assistant prefix's <|tts_bos|> token from identical model state."""
    required = ["save_speculative_snapshot", "restore_speculative_snapshot", "streaming_generate"]
    missing = [name for name in required if not hasattr(runner.model, name)]
    if missing:
        raise RuntimeError(f"MiniCPM-o revision lacks TTS-prefix A/B APIs: {missing}")
    if runner.session_id is None:
        raise RuntimeError("reset_for_sample() must be called before TTS-prefix A/B comparison.")

    pre_final_cache_len = runner.cache_length()
    snapshot = runner.model.save_speculative_snapshot()

    no_tts = _run_official_text_only(
        runner,
        draft_text=draft_text,
        use_tts_template=False,
    )

    restored = runner.model.restore_speculative_snapshot(snapshot)
    if not restored:
        raise RuntimeError("Failed to restore exact pre-final MiniCPM-o state for TTS-prefix A/B comparison.")
    restored_cache_len = runner.cache_length()
    if restored_cache_len != pre_final_cache_len:
        raise RuntimeError(
            "Pre-final KV restore length mismatch during TTS-prefix A/B comparison: "
            f"before={pre_final_cache_len}, restored={restored_cache_len}"
        )

    with_tts = _run_official_text_only(
        runner,
        draft_text=draft_text,
        use_tts_template=True,
    )

    comparison = {
        "pre_final_cache_len": pre_final_cache_len,
        "restored_cache_len": restored_cache_len,
        "same_pre_final_state": pre_final_cache_len == restored_cache_len,
        "only_changed_variable": "use_tts_template / <|tts_bos|> assistant-prefix token",
        "generate_audio": False,
        "official_no_tts_template": no_tts,
        "official_with_tts_template": with_tts,
        # _run_condition expects a `custom` entry when compare_finalizers=True.
        # Point it at variant A only as a transport shim; diagnose.py writes both
        # real A/B rows below and does not interpret this alias as a custom decoder.
        "custom": no_tts,
    }
    return comparison


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
        )
    )
    _warmup(runner, samples[0])

    # _run_condition already owns the streaming/audio/slack path. Override only its
    # finalizer callback for this diagnostic so the streaming implementation itself
    # remains untouched. Each condition is streamed once; A and B are then generated
    # from the exact same snapshotted pre-final state.
    runner.compare_finalizers = lambda *, draft_text: _compare_tts_prefix_finalizers(
        runner,
        draft_text=draft_text,
    )

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

                no_tts_row = _comparison_row(
                    base_row=base_row,
                    decoder="official_no_tts_template",
                    result=comparison["official_no_tts_template"],
                    same_pre_final_state=same_state,
                )
                with_tts_row = _comparison_row(
                    base_row=base_row,
                    decoder="official_with_tts_template",
                    result=comparison["official_with_tts_template"],
                    same_pre_final_state=same_state,
                )
                sample_writer.writerow(no_tts_row)
                sample_writer.writerow(with_tts_row)

                diagnostic_record = {
                    "sample_id": sample.sample_id,
                    "condition": condition,
                    "reference": sample.reference,
                    "first_pass_draft": base_row["first_pass_draft"],
                    "draft_tokens": int(base_row["draft_tokens"]),
                    "pre_final_cache_len": comparison.get("pre_final_cache_len"),
                    "restored_cache_len": comparison.get("restored_cache_len"),
                    "same_pre_final_state": same_state,
                    "only_changed_variable": comparison.get("only_changed_variable"),
                    "generate_audio": False,
                    "official_no_tts_template": comparison["official_no_tts_template"],
                    "official_with_tts_template": comparison["official_with_tts_template"],
                    "official_no_tts_template_wer": no_tts_row["wer"],
                    "official_with_tts_template_wer": with_tts_row["wer"],
                }
                df.write(json.dumps(diagnostic_record, ensure_ascii=False) + "\n")

                sf.flush()
                cf.flush()
                df.flush()

                print(f"\n[{condition}] {sample.sample_id} same_pre_final_state={same_state}")
                print(
                    f"  A no <|tts_bos|>: WER={float(no_tts_row['wer']):.4f} "
                    f"final={no_tts_row['final_transcript']!r} error={no_tts_row['error']!r}"
                )
                print(
                    f"  B + <|tts_bos|>:  WER={float(with_tts_row['wer']):.4f} "
                    f"final={with_tts_row['final_transcript']!r} error={with_tts_row['error']!r}"
                )

    print(f"\n[saved] TTS-prefix A/B samples: {samples_path}")
    print(f"[saved] shared streaming chunks: {chunks_path}")
    print(f"[saved] full TTS-prefix A/B diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
