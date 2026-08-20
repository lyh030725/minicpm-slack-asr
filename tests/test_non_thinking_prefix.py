from minicpm_slack_asr.model import build_non_thinking_assistant_prefix


class DummyMiniCPM:
    think_str = "<think>\\n\\n</think>\\n\\n"


def test_non_thinking_prefix_matches_upstream_shape():
    prefix = build_non_thinking_assistant_prefix(DummyMiniCPM())
    assert prefix == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert prefix.endswith("</think>\n\n")
