import torch

from minicpm_slack_asr.model import (
    MiniCPMSlackASR,
    build_final_prompt,
    build_non_thinking_assistant_prefix,
    resolve_official_forbidden_token_ids,
    resolve_thinking_token_ids,
)


class DummyMiniCPM:
    think_str = "<think>\\n\\n</think>\\n\\n"


class DummyTokenizer:
    token_map = {
        "<think>": 101,
        "</think>": 102,
        ":": 201,
        "#": 202,
    }
    bad_token_ids = [301, 302]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text in self.token_map:
            return [self.token_map[text]]
        return [999]

    def convert_tokens_to_ids(self, token):
        return self.token_map.get(token, 999)


def test_non_thinking_prefix_matches_upstream_shape():
    prefix = build_non_thinking_assistant_prefix(DummyMiniCPM())
    assert prefix == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert prefix.endswith("</think>\n\n")


def test_thinking_markers_resolve_to_dedicated_token_ids():
    assert resolve_thinking_token_ids(DummyTokenizer()) == {
        "<think>": 101,
        "</think>": 102,
    }


def test_official_forbidden_ids_include_bad_ids_and_upstream_tokens():
    forbidden = resolve_official_forbidden_token_ids(DummyTokenizer())
    assert 301 in forbidden
    assert 302 in forbidden
    assert 201 in forbidden
    assert 202 in forbidden


def test_argmax_masks_forbidden_ids():
    runner = object.__new__(MiniCPMSlackASR)
    runner._forbidden_generation_token_ids = {1, 3}

    logits = torch.tensor([[0.1, 100.0, 2.0, 99.0]], dtype=torch.float32)
    assert runner._argmax_token(logits) == 2


def test_final_min_token_masks_terminator_until_text_exists():
    runner = object.__new__(MiniCPMSlackASR)
    runner._forbidden_generation_token_ids = {1}
    runner._terminator_ids = {2}

    # token 1 is forbidden, token 2 is EOS, token 3 is real text.
    logits = torch.tensor([[0.0, 100.0, 90.0, 80.0]], dtype=torch.float32)
    assert runner._argmax_token(logits, generated_count=0, min_new_tokens=1) == 3
    assert runner._argmax_token(logits, generated_count=1, min_new_tokens=1) == 2


def test_baseline_final_prompt_adds_no_extra_text():
    assert build_final_prompt(None) == ""


def test_empty_slack_draft_uses_baseline_path():
    assert build_final_prompt("") == ""
    assert build_final_prompt("   ") == ""


def test_slack_final_prompt_is_short_and_has_no_xml_tags():
    draft = "SHE RAN TO HER HUSBAND'S SIDE"
    prompt = build_final_prompt(draft)
    assert draft in prompt
    assert "Use the audio above as the source of truth." in prompt
    assert "<DRAFT>" not in prompt
    assert "<FINAL_ASR>" not in prompt
    assert "The audio is complete" not in prompt
