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
STREAMING_ASR_PROMPT = (
    "Please listen to the English audio carefully and transcribe the speech verbatim. "
    "During streaming, if a tentative transcript is already present as the assistant prefix, "
    "continue it only with words that have already been spoken. Do not answer the speaker, "
    "summarize, explain, reason aloud, or predict future words. Output only spoken words; do not "
    "write labels such as 'Transcription:' or any preamble. When the audio is complete, review the "
    "entire utterance and output only one corrected final transcript."
)
BASELINE_FINAL_PROMPT = (
    "\n<FINAL_ASR>\n"
    "Use the full audio above as the source of truth. Output only its verbatim transcript.\n"
    "</FINAL_ASR>"
)
SLACK_FINAL_PROMPT_TEMPLATE = (
    "\n<DRAFT>\n{draft}\n</DRAFT>\n"
    "<FINAL_ASR>\n"
    "Use the full audio above as the source of truth. Correct the draft only where needed and "
    "output only the verbatim transcript.\n"
    "</FINAL_ASR>"
)


def build_final_prompt(draft_text: str | None) -> str:
    """Build a compact final-ASR instruction with the draft clearly delimited.

    The previous verbose prompt could occasionally be echoed as assistant output. Keeping
    the instruction short and isolating the fallible draft reduces that copy/continuation
    failure mode while preserving the same information available to the final decoder.
    """
    if draft_text is None:
        return BASELINE_FINAL_PROMPT
    return SLACK_FINAL_PROMPT_TEMPLATE.format(draft=draft_text.strip())


def build_non_thinking_assistant_prefix(model: Any) -> str:
    """Match MiniCPM-o 4.5's own streaming_generate non-thinking BOS prefix.

    Upstream MiniCPM-o constructs text generation with:
      <|im_end|>\n<|im_start|>assistant\n + model.think_str
    when ``enable_thinking=False`` and ``use_tts_template=False``.

    ``think_str`` is the Qwen3 non-thinking marker (an empty, already-closed
    <think> block). The prefix strongly steers Qwen3 away from reasoning, while
    decoder-level masking below makes that constraint explicit.
    """
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
    """Resolve Qwen3 thinking markers to single token IDs for hard masking.

    Qwen3 exposes <think> and </think> as dedicated tokens. Requiring each marker
    to map to exactly one token makes the mask explicit and fail-closed: if a future
    tokenizer revision changes that assumption, the experiment stops instead of
    silently allowing reasoning text back into ASR output.
    """
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

    TTS is not initialized. All generation in this project calls the Qwen3 LLM backbone
    directly. Intermediate draft decoding temporarily extends the model's LLM KV cache,
    then truncates it back to the exact pre-draft length before the next audio chunk.
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

        # Match upstream streaming_generate(enable_thinking=False, use_tts_template=False).
        # The upstream non-thinking prefix is retained, and the actual <think>/</think>
        # token IDs are also hard-masked at every greedy decoding step below.
        self._assistant_prefix_text = build_non_thinking_assistant_prefix(self.model)
        self._assistant_prefix_ids = self.tokenizer.encode(
            self._assistant_prefix_text,
            add_special_tokens=False,
        )
        self._thinking_token_ids = resolve_thinking_token_ids(self.tokenizer)
        self._forbidden_generation_token_ids = set(self._thinking_token_ids.values())
        self._terminator_ids = {
            int(tid)
            for tok in ("<|im_end|>", "<|endoftext|>", "</s>")
            for tid in [self.tokenizer.convert_tokens_to_ids(tok)]
            if tid is not None and int(tid) >= 0
        }
        self._decode_ms_ema = float(config.initial_decode_guard_ms)
        self.session_id: str | None = None

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _decode_text_checked(self, token_ids: list[int], *, phase: str) -> str:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        # Decoder-level masking should make these markers impossible. Keep this check as
        # a final fail-closed guard against tokenizer/model revision drift.
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

        self.model.streaming_prefill(
            session_id=session_id,
            msgs=[{"role": "system", "content": [SYSTEM_PROMPT]}],
            omni_mode=False,
            use_tts_template=False,
            enable_thinking=False,
            is_last_chunk=False,
        )
        # Start one user turn with the ASR task prompt; all following audio chunks are
        # appended to this same turn.
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

    def _prefill_streaming_text(self, text: str) -> float:
        """Append final text through MiniCPM-o's own streaming prefill path.

        This keeps the model's streaming/chat state and position handling aligned with the
        audio-prefill path instead of mutating only the Qwen3 KV cache with a raw LLM call.
        """
        if self.session_id is None:
            raise RuntimeError("reset_for_sample() must be called first.")
        if not text:
            return 0.0
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.model.streaming_prefill(
            session_id=self.session_id,
            msgs=[{"role": "user", "content": [text]}],
            omni_mode=False,
            use_tts_template=False,
            enable_thinking=False,
            is_last_chunk=False,
        )
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0

    def _argmax_token(self, logits: torch.Tensor) -> int:
        # Prefix steering alone is not a hard constraint: Qwen3 can occasionally
        # re-enter reasoning and emit <think>. Mask both dedicated marker token IDs
        # before argmax so neither slack drafts nor final ASR can enter thinking mode.
        masked_logits = logits.clone()
        forbidden_ids = sorted(self._forbidden_generation_token_ids)
        masked_logits[..., forbidden_ids] = float("-inf")
        next_id = int(torch.argmax(masked_logits, dim=-1).item())
        if next_id in self._forbidden_generation_token_ids:
            raise RuntimeError("Decoder selected a forbidden Qwen3 thinking token after masking.")
        return next_id

    def _decode_with_prefix(
        self,
        prefix_ids: list[int],
        *,
        max_new_tokens: int,
        deadline_s: float | None = None,
    ) -> tuple[list[int], float, float]:
        """Greedy LLM-only decode. Returns (new ids, prefix_ms, decode_ms)."""
        if not prefix_ids:
            raise ValueError("prefix_ids must not be empty")

        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        torch.cuda.synchronize()
        prefix_start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.llm(
                input_ids=input_ids,
                past_key_values=self.model.llm_past_key_values,
                use_cache=True,
                return_dict=True,
            )
        self.model.llm_past_key_values = out.past_key_values
        torch.cuda.synchronize()
        prefix_end = time.perf_counter()
        prefix_ms = (prefix_end - prefix_start) * 1000.0

        if deadline_s is not None and prefix_end >= deadline_s:
            return [], prefix_ms, 0.0

        generated: list[int] = []
        logits = out.logits[:, -1, :]
        decode_start = time.perf_counter()

        # The first output token is available from the prefix forward pass.
        next_id = self._argmax_token(logits)
        if next_id not in self._terminator_ids and max_new_tokens > 0:
            generated.append(next_id)

        while len(generated) < max_new_tokens:
            if not generated:
                break
            if deadline_s is not None:
                remaining_ms = (deadline_s - time.perf_counter()) * 1000.0
                guard_ms = max(2.0, self._decode_ms_ema * 1.20)
                if remaining_ms <= guard_ms:
                    break

            step_input = torch.tensor([[generated[-1]]], dtype=torch.long, device=self.device)
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            with torch.inference_mode():
                out = self.model.llm(
                    input_ids=step_input,
                    past_key_values=self.model.llm_past_key_values,
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

            next_id = self._argmax_token(out.logits[:, -1, :])
            if next_id in self._terminator_ids:
                break
            generated.append(next_id)

        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        return generated, prefix_ms, decode_ms

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
            new_ids, prefix_ms, decode_ms = self._decode_with_prefix(
                prefix_ids,
                max_new_tokens=self.config.max_draft_tokens_per_chunk,
                deadline_s=deadline_s,
            )
        finally:
            # Draft text is an external tentative state. Never let it contaminate the
            # audio-stream KV cache consumed by the next chunk.
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
        final_prompt_ms = self._prefill_streaming_text(final_prompt)
        generated_ids, prefix_ms, decode_ms = self._decode_with_prefix(
            self._assistant_prefix_ids,
            max_new_tokens=self.config.max_final_tokens,
            deadline_s=None,
        )
        text = self._decode_text_checked(generated_ids, phase="final ASR").strip()
        return text, final_prompt_ms + prefix_ms, decode_ms

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
            "thinking_token_mask_scope": "slack draft and final ASR decoding",
            "final_prompt_prefill": "MiniCPM-o streaming_prefill, text-only, same user turn",
            "system_prompt": SYSTEM_PROMPT,
            "streaming_asr_prompt": STREAMING_ASR_PROMPT,
            "baseline_final_prompt": BASELINE_FINAL_PROMPT,
            "slack_final_prompt_template": SLACK_FINAL_PROMPT_TEMPLATE,
            "max_final_tokens": self.config.max_final_tokens,
            "max_draft_tokens_per_chunk": self.config.max_draft_tokens_per_chunk,
            "max_draft_prefix_tokens": self.config.max_draft_prefix_tokens,
            "min_slack_ms": self.config.min_slack_ms,
        }
