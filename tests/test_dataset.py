import numpy as np
import soundfile as sf

from minicpm_slack_asr.dataset import TARGET_SAMPLE_RATE, discover_librispeech, streaming_chunks


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


def test_discover_includes_exactly_ten_seconds(tmp_path):
    chapter = tmp_path / "1" / "2"
    chapter.mkdir(parents=True)
    exact_id = "1-2-0000"
    short_id = "1-2-0001"
    sf.write(chapter / f"{exact_id}.flac", np.zeros(10 * TARGET_SAMPLE_RATE, dtype=np.float32), TARGET_SAMPLE_RATE)
    sf.write(chapter / f"{short_id}.flac", np.zeros(10 * TARGET_SAMPLE_RATE - 1, dtype=np.float32), TARGET_SAMPLE_RATE)
    (chapter / "1-2.trans.txt").write_text(
        f"{exact_id} EXACTLY TEN SECONDS\n{short_id} JUST UNDER TEN SECONDS\n",
        encoding="utf-8",
    )

    samples = discover_librispeech(tmp_path, min_duration_s=10.0, max_samples=0)
    assert [sample.sample_id for sample in samples] == [exact_id]
