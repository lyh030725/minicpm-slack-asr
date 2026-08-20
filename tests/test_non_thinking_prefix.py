from minicpm_slack_asr.model import (
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
