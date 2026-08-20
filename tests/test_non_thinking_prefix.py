import torch

from minicpm_slack_asr.model import (
    MiniCPMSlackASR,
    build_final_prompt,
    build_non_thinking_assistant_prefix,
    resolve_thinking_token_ids,
)


class DummyMiniCPM:
    think_str = "<think>\\n\\n</think>\\n\\n"


class DummyTokenizer:
    token_map = {"<think>": 101, "</think>": 102}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text in self.token_map:
            return [self.token_map[text]]
        return [999]


def test_non_thinking_prefix_matches_upstream_shape():
    prefix = build_non_thinking_assistant_prefix(DummyMiniCPM())
    assert prefix == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert prefix.endswith("</think>\n\n")


def test_thinking_markers_resolve_to_dedicated_token_ids():
    assert resolve_thinking_token_ids(DummyTokenizer()) == {
        "<think>": 101,
        "</think>": 102,
    }


def test_argmax_hard_masks_thinking_token_ids():
    runner = object.__new__(MiniCPMSlackASR)
    runner._forbidden_generation_token_ids = {1, 3}

    # Thinking tokens have the largest raw logits, but token 2 must win after masking.
    logits = torch.tensor([[0.1, 100.0, 2.0, 99.0]], dtype=torch.float32)
    assert runner._argmax_token(logits) == 2


def test_baseline_final_prompt_is_compact():
    prompt = build_final_prompt(None)
    assert "<FINAL_ASR>" in prompt
    assert "<DRAFT>" not in prompt
    assert "The audio is complete" not in prompt
    assert "verbatim transcript" in prompt


def test_slack_final_prompt_delimits_full_draft():
    draft = "SHE RAN TO HER HUSBAND'S SIDE"
    prompt = build_final_prompt(draft)
    assert f"<DRAFT>\n{draft}\n</DRAFT>" in prompt
    assert "Use the full audio above as the source of truth." in prompt
    assert "The audio is complete" not in prompt
