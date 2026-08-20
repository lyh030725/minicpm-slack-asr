from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import math
import random

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000
UNIT_SECONDS = 1.0
UNIT_SAMPLES = TARGET_SAMPLE_RATE


@dataclass(frozen=True)
class AudioSample:
    sample_id: str
    path: Path
    duration_s: float
    sample_rate: int
    frames: int
    reference: str


def _load_librispeech_transcripts(root: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for transcript_file in sorted(root.rglob("*.trans.txt")):
        with transcript_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                sample_id, text = parts
                transcripts[sample_id] = text.strip()
    return transcripts


def discover_librispeech(
    root: Path,
    min_duration_s: float = 10.0,
    max_samples: int = 0,
    shuffle: bool = False,
    seed: int = 42,
) -> list[AudioSample]:
    """Discover LibriSpeech FLAC files with duration greater than or equal to min_duration_s.

    ``max_samples=0`` means all qualifying utterances. References are read from the
    standard ``*.trans.txt`` files shipped with LibriSpeech.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    transcripts = _load_librispeech_transcripts(root)
    samples: list[AudioSample] = []
    for path in sorted(root.rglob("*.flac")):
        info = sf.info(str(path))
        duration_s = float(info.frames) / float(info.samplerate)
        if duration_s < min_duration_s:
            continue
        sample_id = path.stem
        reference = transcripts.get(sample_id)
        if reference is None:
            raise RuntimeError(f"Missing LibriSpeech transcript for {sample_id}: {path}")
        samples.append(
            AudioSample(
                sample_id=sample_id,
                path=path,
                duration_s=duration_s,
                sample_rate=int(info.samplerate),
                frames=int(info.frames),
                reference=reference,
            )
        )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)

    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def load_audio(path: Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    elif audio.ndim != 1:
        raise ValueError(f"Unexpected audio shape for {path}: {audio.shape}")

    if int(sr) != target_sample_rate:
        gcd = math.gcd(int(sr), target_sample_rate)
        up = target_sample_rate // gcd
        down = int(sr) // gcd
        audio = resample_poly(audio, up, down).astype(np.float32, copy=False)

    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def streaming_chunks(
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Iterator[tuple[int, np.ndarray, int, bool]]:
    """Yield every utterance sample exactly once in 1-second streaming chunks.

    The final partial chunk is zero padded to one second because MiniCPM-o's upstream
    streaming example also pads the final audio chunk. ``valid_samples`` records how
    much speech is real so the padding is explicit in the logs.
    """
    audio = np.asarray(audio, dtype=np.float32)
    chunk_size = int(sample_rate)
    if chunk_size <= 0:
        raise ValueError("sample_rate must be positive")

    n_chunks = int(math.ceil(len(audio) / chunk_size)) if len(audio) else 0
    for unit_idx in range(n_chunks):
        start = unit_idx * chunk_size
        end = min(start + chunk_size, len(audio))
        valid_samples = end - start
        is_last = unit_idx == n_chunks - 1
        if valid_samples == chunk_size:
            chunk = np.ascontiguousarray(audio[start:end], dtype=np.float32)
        else:
            chunk = np.zeros(chunk_size, dtype=np.float32)
            if valid_samples > 0:
                chunk[:valid_samples] = audio[start:end]
        yield unit_idx, chunk, valid_samples, is_last
