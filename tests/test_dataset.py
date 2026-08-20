import numpy as np

from minicpm_slack_asr.dataset import TARGET_SAMPLE_RATE, streaming_chunks


def test_streaming_chunks_keeps_partial_tail():
    audio = np.arange(TARGET_SAMPLE_RATE + 123, dtype=np.float32)
    chunks = list(streaming_chunks(audio))
    assert len(chunks) == 2
    _, first, valid_first, last_first = chunks[0]
    _, second, valid_second, last_second = chunks[1]
    assert len(first) == TARGET_SAMPLE_RATE
    assert valid_first == TARGET_SAMPLE_RATE
    assert last_first is False
    assert len(second) == TARGET_SAMPLE_RATE
    assert valid_second == 123
    assert last_second is True
    assert np.all(second[123:] == 0)
