from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModel


SYSTEM_PROMPT = (
    "You are a precise automatic speech recognition engine. Return transcript text only. "
    "Never explain, reason aloud, summarize, answer the speaker, or add labels or preambles."
)
# Keep the main ASR request close to the official MiniCPM-o ASR example instead of
# layering experiment-specific XML-like formatting into the streaming user turn.
STREAMING_ASR_PROMPT = "Please listen to the audio snippet carefully and transcribe the content."

# Baseline already has the ASR instruction before the streamed audio, so no extra
# text is appended at utterance end. This is the closest path to normal generation
# from the accumulated full-audio KV cache.
BASELINE_FINAL_PROMPT = ""
SLACK_FINAL_PROMPT_TEMPLATE = (
    "\nTentative transcript (it may contain recognition errors or omissions):\n{draft}\n"
    "Use the audio above as the source of truth. Correct the tentative transcript if needed and "
    "output only the verbatim transcript."
)

# Mirrored from MiniCPM-o 4.5 ChunkPrefillChunkGenerate. The upstream decoder also
# suppresses tokenizer.bad_token_ids; both sets are used below before greedy argmax.
OFFICIAL_FORBIDDEN_TOKENS = (
    ":",
    "：",
    "；",
    "#",
    "“",
    "”",
    "‘",
    "’",
    "@",
    "*",
    "【",
    "】",
    "「",
    "」",
    "(",
    ")",
    "（",
    "）",
    "[",
    "]",
    "&",
    "/",
    "$",
)


def build_final_prompt(draft_text: str | None) -> str:
    """Return no extra baseline prompt; add only a short draft-correction hint for slack."""
    if draft_text is None:
        return BASELINE_FINAL_PROMPT
    draft = draft_text.strip()
    if not draft:
        # If slack produced no usable draft, make the final path identical to baseline.
        return BASELINE_FINAL_PROMPT
    return SLACK_FINAL_PROMPT_TEMPLATE.format(draft=draft)


def build_non_thinking_assistant_prefix(model: Any) -> str:
    """Match MiniCPM-o 4.5's streaming_generate non-thinking BOS prefix."""
    think_str = getattr(model, "think_str", None)
    if not isinstance(think_str, str) or not think_str:
        raise RuntimeError(
            "MiniCPM-o model.think_str is unavailable. Cannot safely reproduce the upstream "
            "Qwen3 non-thinking generation prefix; pin a compatible MiniCPM-o 4.5 revision."
        )
    think_str = think_str.replace("\\n", "\n")
    prefix = "<|im_end|>\n<|im_start|>assistant\n" + think_str
    if "<think>" not in prefix or "</think>" not in prefix:
        raise RuntimeError(
            "Unexpected MiniCPM-o think_str. Refusing to run because non-thinking decoding "
            f"cannot be verified: {think_str!r}"
        )
    return prefix


def resolve_thinking_token_ids(tokenizer: Any) -> dict[str, int]:
    """Resolve Qwen3 thinking markers to single token IDs for hard masking."""
    result: dict[str, int] = {}
    for marker in ("<think>", "</think>"):
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"Cannot hard-mask Qwen3 thinking marker {marker!r}: expected one token, got {ids}. "
                "Pin a compatible MiniCPM-o 4.5 tokenizer revision."
            )
        result[marker] = int(ids[0])
    if len(set(result.values())) != len(result):
        raise RuntimeError(f"Unexpected shared token ID for Qwen3 thinking markers: {result}")
    return result


def resolve_official_forbidden_token_ids(tokenizer: Any) -> set[int]:
    """Mirror MiniCPM-o's text decoder suppression list."""
    result = {int(tid) for tid in getattr(tokenizer, "bad_token_ids", []) if tid is not None and int(tid) >= 0}
    for token in OFFICIAL_FORBIDDEN_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and int(tid) >= 0:
            result.add(int(tid))
    return result


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "openbmb/MiniCPM-o-4_5"
    revision: str = "main"
    seed: int = 42
    attn_implementation: str = "sdpa"
    max_final_tokens: int = 512
    max_draft_tokens_per_chunk: int = 12
    max_draft_prefix_tokens: int = 256
    min_slack_ms: float = 50.0
    initial_decode_guard_ms: float = 25.0
    diagnostic_top_k: int = 5


@dataclass
class DraftStepResult:
    new_token_ids: list[int]
    new_text: str
    prefix_tokens: int
    prefix_ms: float
    decode_ms: float
    total_ms: float
    deadline_remaining_ms: float
    deadline_miss: bool
    cache_len_before: int
    cache_len_after_restore: int


class MiniCPMSlackASR:
    """Audio-in / LLM-text-out runner for MiniCPM-o 4.5.

    TTS is not initialized. All generation calls the Qwen3 LLM backbone directly.
    Intermediate draft decoding temporarily extends the LLM KV cache and rolls it
    back to the exact post-audio-prefill length before the next audio chunk.
    """

    def __init__(self, config: ModelConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MiniCPM-o 4.5 inference.")
        self.config = config
        self._seed_all(config.seed)

        self.model = AutoModel.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation=config.attn_implementation,
            init_vision=False,
            init_audio=True,
            init_tts=False,
        ).eval().cuda()
        self.model.prepare_processor()
        self.tokenizer = self.model.processor.tokenizer
        self.device = self.model.llm.device

        required = ["streaming_prefill", "_get_kv_cache_length", "_truncate_llm_cache"]
        missing = [name for name in required if not hasattr(self.model, name)]
        if missing:
            raise RuntimeError(
                "The selected MiniCPM-o revision no longer exposes the streaming/cache APIs "
                f"required by this experiment: {missing}. Pin a compatible model revision."
            )

        self._assistant_prefix_text = build_non_thinking_assistant_prefix(self.model)
        self._assistant_prefix_ids = self.tokenizer.encode(
            self._assistant_prefix_text,
            add_special_tokens=False,
        )
        self._thinking_token_ids = resolve_thinking_token_ids(self.tokenizer)

        # Match upstream streaming text-generation terminators exactly.
        self._terminator_ids = {
            int(tid)
            for tok in ("<|tts_eos|>", "<|im_end|>", "</s>")
            for tid in [self.tokenizer.convert_tokens_to_ids(tok)]
            if tid is not None and int(tid) >= 0
        }
        self._official_forbidden_token_ids = resolve_official_forbidden_token_ids(self.tokenizer)

        # Text-only ASR should never emit modality/chat-control special tokens. Keep
        # terminators separate so they can still end decoding after a real text token.
        all_special_ids = {int(tid) for tid in getattr(self.tokenizer, "all_special_ids", [])}
        self._special_control_token_ids = all_special_ids - self._terminator_ids

        self._forbidden_generation_token_ids = (
            set(self._thinking_token_ids.values())
            | self._official_forbidden_token_ids
            | self._special_control_token_ids
        )
        self._draft_raw_stop_token_ids = set(self._thinking_token_ids.values()) | self._terminator_ids

        self._decode_ms_ema = float(config.initial_decode_guard_ms)
        self._last_final_diagnostics: dict[str, Any] = {}
        self.session_id: str | None = None

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _decode_text_checked(self, token_ids: list[int], *, phase: str) -> str:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if "<think>" in text or "</think>" in text:
            raise RuntimeError(
                f"Qwen3 thinking content escaped the decoder-level mask during {phase}. "
                f"Generated text starts with: {text[:200]!r}"
            )
        return text

    def reset_for_sample(self, session_id: str) -> None:
        self._seed_all(self.config.seed)
        self.session_id = session_id
        self.model.reset_session(reset_token2wav_cache=False)
        self._decode_ms_ema = float(self.config.initial_decode_guard_ms)
        self._last_final_diagnostics = {}

        self.model.streaming_prefill(
            session_id=session_id,
            msgs=[{"role": "system", "content": [SYSTEM_PROMPT]}],
            omni_mode=False,
            use_tts_template=False,
            enable_thinking=False,
            is_last_chunk=False,
        )
        self.model.streaming_prefill(
            session_id=session_id,
            msgs=[{"role": "user", "content": [STREAMING_ASR_PROMPT]}],
            omni_mode=False,
            use_tts_template=False,
            enable_thinking=False,
            is_last_chunk=False,
        )
        torch.cuda.synchronize()

    def cache_length(self) -> int:
        return int(self.model._get_kv_cache_length())

    def prefill_audio(self, audio_chunk: np.ndarray, *, is_last_chunk: bool) -> float:
        if self.session_id is None:
            raise RuntimeError("reset_for_sample() must be called first.")
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.model.streaming_prefill(
            session_id=self.session_id,
            msgs=[{"role": "user", "content": [audio_chunk]}],
            omni_mode=False,
            use_tts_template=False,
            enable_thinking=False,
            is_last_chunk=is_last_chunk,
        )
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0

    def _prefill_raw_text(self, text: str) -> float:
        """Append a short final draft hint directly to the current full-audio LLM KV."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return 0.0
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        cache_length = self.cache_length()
        attention_mask = torch.ones(
            (1, cache_length + input_ids.shape[1]),
            dtype=torch.bool,
            device=self.device,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.llm(
                input_ids=input_ids,
                past_key_values=self.model.llm_past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        self.model.llm_past_key_values = out.past_key_values
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0

    def _apply_generation_masks(
        self,
        logits: torch.Tensor,
        *,
        generated_count: int,
        min_new_tokens: int,
    ) -> torch.Tensor:
        masked_logits = logits.clone()
        vocab_size = masked_logits.shape[-1]
        forbidden_ids = sorted(tid for tid in self._forbidden_generation_token_ids if 0 <= tid < vocab_size)
        if forbidden_ids:
            masked_logits[..., forbidden_ids] = float("-inf")

        # LibriSpeech utterances are non-empty. For final decoding only, min_new_tokens=1
        # prevents a masked <think> from simply exposing <|im_end|> as the new argmax and
        # producing final=''. Slack decoding keeps min_new_tokens=0 and may legitimately
        # choose to produce no tentative text in a short slack window.
        if generated_count < min_new_tokens:
            terminator_ids = sorted(tid for tid in self._terminator_ids if 0 <= tid < vocab_size)
            if terminator_ids:
                masked_logits[..., terminator_ids] = float("-inf")

        if not torch.isfinite(masked_logits).any():
            raise RuntimeError("All decoder logits were masked; tokenizer/model constraints are inconsistent.")
        return masked_logits

    def _argmax_token(
        self,
        logits: torch.Tensor,
        *,
        generated_count: int = 0,
        min_new_tokens: int = 0,
    ) -> int:
        masked_logits = self._apply_generation_masks(
            logits,
            generated_count=generated_count,
            min_new_tokens=min_new_tokens,
        )
        next_id = int(torch.argmax(masked_logits, dim=-1).item())
        if next_id in self._forbidden_generation_token_ids:
            raise RuntimeError(f"Decoder selected forbidden token id={next_id} after masking.")
        if generated_count < min_new_tokens and next_id in self._terminator_ids:
            raise RuntimeError("Decoder selected a terminator before min_new_tokens after masking.")
        return next_id

    def _topk_snapshot(self, logits: torch.Tensor, *, k: int) -> list[dict[str, Any]]:
        k = max(1, min(int(k), int(logits.shape[-1])))
        values, indices = torch.topk(logits.float(), k=k, dim=-1)
        rows: list[dict[str, Any]] = []
        for value, index in zip(values[0].tolist(), indices[0].tolist()):
            token_id = int(index)
            try:
                token = self.tokenizer.convert_ids_to_tokens(token_id)
            except Exception:
                token = None
            try:
                decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                decoded = None
            rows.append(
                {
                    "id": token_id,
                    "token": token,
                    "decoded": decoded,
                    "logit": float(value),
                    "is_terminator": token_id in self._terminator_ids,
                    "is_forbidden": token_id in self._forbidden_generation_token_ids,
                }
            )
        return rows

    def _decode_with_prefix(
        self,
        prefix_ids: list[int],
        *,
        max_new_tokens: int,
        deadline_s: float | None = None,
        min_new_tokens: int = 0,
        stop_on_raw_draft_control: bool = False,
        capture_first_token_diagnostics: bool = False,
    ) -> tuple[list[int], float, float, dict[str, Any]]:
        """Greedy LLM-only decode with MiniCPM token suppression and optional diagnostics."""
        if not prefix_ids:
            raise ValueError("prefix_ids must not be empty")

        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        cache_length = self.cache_length()
        attention_mask = torch.ones(
            (1, cache_length + input_ids.shape[1]),
            dtype=torch.bool,
            device=self.device,
        )
        torch.cuda.synchronize()
        prefix_start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.llm(
                input_ids=input_ids,
                past_key_values=self.model.llm_past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        self.model.llm_past_key_values = out.past_key_values
        torch.cuda.synchronize()
        prefix_end = time.perf_counter()
        prefix_ms = (prefix_end - prefix_start) * 1000.0

        diagnostics: dict[str, Any] = {}
        if deadline_s is not None and prefix_end >= deadline_s:
            return [], prefix_ms, 0.0, diagnostics

        generated: list[int] = []
        logits = out.logits[:, -1, :]
        decode_start = time.perf_counter()
        first_selection = True

        while len(generated) < max_new_tokens:
            raw_argmax_id = int(torch.argmax(logits, dim=-1).item())
            masked_logits = self._apply_generation_masks(
                logits,
                generated_count=len(generated),
                min_new_tokens=min_new_tokens,
            )
            selected_id = int(torch.argmax(masked_logits, dim=-1).item())

            if first_selection and capture_first_token_diagnostics:
                diagnostics = {
                    "raw_argmax_id": raw_argmax_id,
                    "selected_id": selected_id,
                    "raw_argmax_token": self.tokenizer.convert_ids_to_tokens(raw_argmax_id),
                    "selected_token": self.tokenizer.convert_ids_to_tokens(selected_id),
                    "raw_topk": self._topk_snapshot(logits, k=self.config.diagnostic_top_k),
                    "masked_topk": self._topk_snapshot(masked_logits, k=self.config.diagnostic_top_k),
                }

            # For tentative slack work, do not force a transcript when the model's raw
            # preference is to stop or re-enter thinking. Simply spend zero additional
            # draft tokens and rollback the temporary branch.
            if stop_on_raw_draft_control and raw_argmax_id in self._draft_raw_stop_token_ids:
                if first_selection and capture_first_token_diagnostics:
                    diagnostics["stopped_on_raw_control"] = True
                break

            if selected_id in self._terminator_ids:
                break
            if selected_id in self._forbidden_generation_token_ids:
                raise RuntimeError(f"Decoder selected forbidden token id={selected_id} after masking.")

            generated.append(selected_id)
            first_selection = False
            if len(generated) >= max_new_tokens:
                break

            if deadline_s is not None:
                remaining_ms = (deadline_s - time.perf_counter()) * 1000.0
                guard_ms = max(2.0, self._decode_ms_ema * 1.20)
                if remaining_ms <= guard_ms:
                    break

            step_input = torch.tensor([[generated[-1]]], dtype=torch.long, device=self.device)
            step_cache_length = self.cache_length()
            step_attention_mask = torch.ones(
                (1, step_cache_length + 1),
                dtype=torch.bool,
                device=self.device,
            )
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            with torch.inference_mode():
                out = self.model.llm(
                    input_ids=step_input,
                    past_key_values=self.model.llm_past_key_values,
                    attention_mask=step_attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            self.model.llm_past_key_values = out.past_key_values
            torch.cuda.synchronize()
            step_end = time.perf_counter()
            step_ms = (step_end - step_start) * 1000.0
            self._decode_ms_ema = 0.8 * self._decode_ms_ema + 0.2 * step_ms

            if deadline_s is not None and step_end > deadline_s:
                break
            logits = out.logits[:, -1, :]

        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        return generated, prefix_ms, decode_ms, diagnostics

    def draft_in_slack(
        self,
        draft_token_ids: list[int],
        *,
        deadline_s: float,
    ) -> DraftStepResult:
        start = time.perf_counter()
        cache_len_before = self.cache_length()
        remaining_ms = (deadline_s - start) * 1000.0
        if remaining_ms < self.config.min_slack_ms:
            return DraftStepResult(
                new_token_ids=[],
                new_text="",
                prefix_tokens=0,
                prefix_ms=0.0,
                decode_ms=0.0,
                total_ms=0.0,
                deadline_remaining_ms=remaining_ms,
                deadline_miss=remaining_ms < 0.0,
                cache_len_before=cache_len_before,
                cache_len_after_restore=cache_len_before,
            )

        draft_tail = draft_token_ids[-self.config.max_draft_prefix_tokens :]
        prefix_ids = [*self._assistant_prefix_ids, *draft_tail]
        try:
            new_ids, prefix_ms, decode_ms, _ = self._decode_with_prefix(
                prefix_ids,
                max_new_tokens=self.config.max_draft_tokens_per_chunk,
                deadline_s=deadline_s,
                min_new_tokens=0,
                stop_on_raw_draft_control=True,
            )
        finally:
            self.model._truncate_llm_cache(cache_len_before)
            torch.cuda.synchronize()

        finish = time.perf_counter()
        cache_len_after = self.cache_length()
        if cache_len_after != cache_len_before:
            raise RuntimeError(
                f"Draft KV rollback failed: before={cache_len_before}, after={cache_len_after}"
            )
        new_text = self._decode_text_checked(new_ids, phase="slack draft")
        return DraftStepResult(
            new_token_ids=new_ids,
            new_text=new_text,
            prefix_tokens=len(prefix_ids),
            prefix_ms=prefix_ms,
            decode_ms=decode_ms,
            total_ms=(finish - start) * 1000.0,
            deadline_remaining_ms=(deadline_s - finish) * 1000.0,
            deadline_miss=finish > deadline_s,
            cache_len_before=cache_len_before,
            cache_len_after_restore=cache_len_after,
        )

    def finalize(self, *, draft_text: str | None) -> tuple[str, float, float]:
        final_prompt = build_final_prompt(draft_text)
        final_prompt_ms = self._prefill_raw_text(final_prompt) if final_prompt else 0.0
        generated_ids, prefix_ms, decode_ms, diagnostics = self._decode_with_prefix(
            self._assistant_prefix_ids,
            max_new_tokens=self.config.max_final_tokens,
            deadline_s=None,
            min_new_tokens=1,
            stop_on_raw_draft_control=False,
            capture_first_token_diagnostics=True,
        )
        self._last_final_diagnostics = diagnostics
        text = self._decode_text_checked(generated_ids, phase="final ASR").strip()
        if not text:
            raise RuntimeError(
                "Final ASR decoded to an empty transcript after min_new_tokens=1. "
                f"First-token diagnostics: {diagnostics}"
            )
        return text, final_prompt_ms + prefix_ms, decode_ms

    def final_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_final_diagnostics)

    def model_settings(self) -> dict[str, Any]:
        return {
            "model_id": self.config.model_id,
            "revision": self.config.revision,
            "attn_implementation": self.config.attn_implementation,
            "init_vision": False,
            "init_audio": True,
            "init_tts": False,
            "decode": "greedy argmax on model.llm only",
            "enable_thinking": False,
            "assistant_generation_prefix": self._assistant_prefix_text.replace("\n", "\\n"),
            "assistant_prefix_source": "matches upstream streaming_generate(enable_thinking=False, use_tts_template=False)",
            "thinking_token_mask": dict(self._thinking_token_ids),
            "official_bad_token_suppression_count": len(self._official_forbidden_token_ids),
            "special_control_token_suppression_count": len(self._special_control_token_ids),
            "terminator_ids": sorted(self._terminator_ids),
            "terminators": ["<|tts_eos|>", "<|im_end|>", "</s>"],
            "final_min_new_tokens": 1,
            "draft_raw_think_or_eos_policy": "stop draft without forcing another token",
            "final_prompt_prefill": "baseline: none; slack: short direct model.llm draft-correction hint",
            "system_prompt": SYSTEM_PROMPT,
            "streaming_asr_prompt": STREAMING_ASR_PROMPT,
            "baseline_final_prompt": BASELINE_FINAL_PROMPT,
            "slack_final_prompt_template": SLACK_FINAL_PROMPT_TEMPLATE,
            "max_final_tokens": self.config.max_final_tokens,
            "max_draft_tokens_per_chunk": self.config.max_draft_tokens_per_chunk,
            "max_draft_prefix_tokens": self.config.max_draft_prefix_tokens,
            "min_slack_ms": self.config.min_slack_ms,
            "diagnostic_top_k": self.config.diagnostic_top_k,
        }
