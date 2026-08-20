from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModel


SYSTEM_PROMPT = "You are a precise automatic speech recognition assistant."
STREAMING_ASR_PROMPT = (
    "Please listen to the English audio carefully and transcribe the speech verbatim. "
    "During streaming, if a tentative transcript is already present as the assistant prefix, "
    "continue it only with words that have already been spoken. Do not answer the speaker, "
    "summarize, explain, or predict future words. When the audio is complete, review the entire "
    "utterance and output only one corrected final transcript."
)
BASELINE_FINAL_PROMPT = (
    "\nThe audio is complete. Review the entire utterance and output only the final verbatim transcript."
)
SLACK_FINAL_PROMPT_TEMPLATE = (
    "\nThe audio is complete. The following is a tentative first-pass transcript produced during "
    "listening and it may contain recognition errors or omissions:\n{draft}\n"
    "Re-check the full audio, correct any misheard or missing words, and output only the final "
    "verbatim transcript."
)
ASSISTANT_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"


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

        self._assistant_prefix_ids = self.tokenizer.encode(ASSISTANT_PREFIX, add_special_tokens=False)
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

    def _prefill_raw_text(self, text: str) -> float:
        """Append plain text to the current user turn without invoking audio/TTS code."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return 0.0
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.llm(
                input_ids=input_ids,
                past_key_values=self.model.llm_past_key_values,
                use_cache=True,
                return_dict=True,
            )
        self.model.llm_past_key_values = out.past_key_values
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0

    def _argmax_token(self, logits: torch.Tensor) -> int:
        return int(torch.argmax(logits, dim=-1).item())

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
        new_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
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
        if draft_text is None:
            final_prompt = BASELINE_FINAL_PROMPT
        else:
            final_prompt = SLACK_FINAL_PROMPT_TEMPLATE.format(draft=draft_text.strip())

        final_prompt_ms = self._prefill_raw_text(final_prompt)
        generated_ids, prefix_ms, decode_ms = self._decode_with_prefix(
            self._assistant_prefix_ids,
            max_new_tokens=self.config.max_final_tokens,
            deadline_s=None,
        )
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
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
            "system_prompt": SYSTEM_PROMPT,
            "streaming_asr_prompt": STREAMING_ASR_PROMPT,
            "baseline_final_prompt": BASELINE_FINAL_PROMPT,
            "slack_final_prompt_template": SLACK_FINAL_PROMPT_TEMPLATE,
            "max_final_tokens": self.config.max_final_tokens,
            "max_draft_tokens_per_chunk": self.config.max_draft_tokens_per_chunk,
            "max_draft_prefix_tokens": self.config.max_draft_prefix_tokens,
            "min_slack_ms": self.config.min_slack_ms,
        }
