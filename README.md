# MiniCPM-o 4.5 Slack ASR

Training-free experiment for using **MiniCPM-o 4.5's listening-time LLM compute slack** to improve ASR.

The experiment is intentionally **audio-in / text-token-out only**. MiniCPM-o's TTS/speech-generation path is never initialized or called:

```python
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-4_5",
    trust_remote_code=True,
    init_vision=False,
    init_audio=True,
    init_tts=False,
)
```

All intermediate and final decoding directly uses the Qwen3 LLM backbone (`model.llm`). No speech tokens, audio-token decoder, Token2Wav, or waveform generation are used.

## Research question

Can otherwise-idle compute between 1-second streaming audio arrivals be used for tentative text ASR, then reused as a draft when the complete utterance is available?

```text
1-second audio chunk
        |
        v
MiniCPM-o streaming audio prefill
        |
        +---- remaining time before next 1 s deadline ----+
        |                                                  |
        |                              LLM-only tentative ASR text decoding
        |                                                  |
next audio chunk <-----------------------------------------+

... repeat while speech is arriving ...

last audio chunk
        |
        v
full audio context + tentative first-pass transcript
        |
        v
LLM-only second-pass correction
        |
        v
final ASR text
```

## Dataset

The default experiment uses **every LibriSpeech ASR `test-clean` utterance whose duration is greater than or equal to 10 seconds**.

There is no sample cap in the default run:

```text
min_duration >= 10.0 s
max_samples = 0   # all qualifying utterances
```

The final partial second of an utterance is zero-padded to one second so no real speech samples are dropped. The number of real samples is recorded in `chunks.csv`.

## Two conditions

The default command evaluates both conditions on the same selected utterances.

### 1. `baseline`

1. Stream the complete audio through MiniCPM-o in 1-second chunks.
2. Do **no** intermediate decoding.
3. After the complete audio is available, ask the LLM backbone for one final verbatim transcript.

### 2. `slack_2pass`

1. Stream the same audio in 1-second chunks.
2. After each non-final chunk, compute the wall-clock time remaining until the next 1-second deadline.
3. If enough time remains, use only that budget for tentative LLM text decoding.
4. Restore the LLM KV cache to the exact state it had immediately after the audio prefill.
5. Keep the generated text externally as the tentative first-pass transcript.
6. After the complete utterance, give that tentative transcript back as an explicitly fallible draft and ask the LLM to re-check the full audio and produce the final transcript.

This gives a clean comparison of final WER with and without listening-slack work.

## Why the tentative draft does not remain in the main KV cache

Intermediate decoding is speculative. If a wrong word is permanently appended to the same KV cache used by later audio chunks, the model can self-condition on its own mistake.

This implementation therefore temporarily extends the audio-stream LLM KV cache and then truncates it back:

```text
                         temporary branch
                              |
audio KV after chunk ----------+---- assistant prefix + tentative text
       |
       |                             decode inside slack
       |                                   |
       +<--------- truncate KV ------------+
       |
next audio chunk
```

Only the text string/token IDs are retained externally. The second pass receives that text as a draft that may be corrected.

The current upstream MiniCPM-o model already exposes cache-length/truncation helpers used by its own speculative streaming logic; this project relies on those helpers. For final reproducible experiments, pin the model revision and keep the resolved Hugging Face SHA recorded in `environment.json`.

## Prompting

No training or fine-tuning is used. The prompt tells the model to:

- perform verbatim English ASR,
- continue a tentative assistant transcript only with speech already heard,
- avoid answering, summarizing, or predicting future words,
- treat the first-pass transcript as fallible,
- re-check the full utterance before the final transcript.

The exact prompts used in a run are written into `environment.json`.

## Slack policy

For each non-final 1-second chunk:

```text
slack = next_audio_deadline - audio_prefill_finish_time
```

Tentative decoding begins only when at least `--min-slack-ms` remains (50 ms by default).

Generation is greedy and token-by-token. The implementation keeps an EMA of observed token latency and refuses to intentionally begin another token forward pass when the remaining time is smaller than a 1.2x latency guard.

The first-pass generation also has a semantic cap of 12 new text tokens per audio chunk by default. This avoids turning unused compute into aggressive future-word prediction. Change it with:

```bash
--max-draft-tokens-per-chunk N
```

The complete tentative transcript is used as the second-pass draft. Up to 256 of its most recent tokens are re-prefilled as the assistant prefix during each intermediate branch; for typical LibriSpeech >=10 s utterances this generally covers the whole draft.

## RunPod environment

Target image:

```text
runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404
```

The setup script intentionally preserves the image's PyTorch/CUDA installation.

### 1. Install

```bash
bash scripts/setup_runpod.sh
```

Recommended cache variables:

```bash
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 2. Download LibriSpeech test-clean

```bash
bash scripts/download_librispeech.sh
```

This also creates:

```text
data/manifests/test-clean-ge10.csv
```

containing every utterance with duration greater than or equal to 10 seconds.

### 3. Run the complete experiment

```bash
bash scripts/run_test_clean.sh
```

Equivalent command:

```bash
python -m minicpm_slack_asr.run \
  --dataset-root data/LibriSpeech/test-clean \
  --min-duration 10 \
  --max-samples 0 \
  --conditions baseline slack_2pass \
  --realtime \
  --output-dir results/test-clean-ge10
```

`--max-samples 0` is important: it means **all** qualifying >=10-second samples.

For a quick code check before the full run, you can temporarily use e.g. `--max-samples 2 --no-realtime`. Do not use that for the final reported experiment.

## Outputs

```text
results/test-clean-ge10/
├── manifest.csv
├── samples.csv
├── chunks.csv
├── summary.json
└── environment.json
```

### `samples.csv`

One row per sample/condition, including:

- LibriSpeech reference text
- tentative first-pass draft (`slack_2pass` only)
- final transcript
- first-pass WER
- final WER
- substitutions / deletions / insertions
- total audio prefill time
- total draft compute time
- final decode time
- number of chunks that missed the 1-second deadline

### `chunks.csv`

Per streaming chunk diagnostics, including:

- audio prefill time
- slack available before draft decoding
- tentative tokens generated in that slack
- tentative text so far
- draft-prefix prefill/decode time
- remaining time at the deadline
- deadline miss flag
- LLM KV length before speculative decoding and after rollback

For every successful rollback:

```text
kv_cache_before_draft == kv_cache_after_restore
```

The run raises an error if this invariant fails.

### `summary.json`

Reports micro-WER for `baseline` and `slack_2pass`, plus:

```text
relative_wer_reduction_pct
absolute_wer_change
```

A positive `relative_wer_reduction_pct` means the slack-assisted two-pass condition improved WER.

## Resume behavior

The full >=10-second set can take a while. `--resume` is enabled by default. If `samples.csv` already contains a completed `(sample_id, condition)`, that condition is skipped on the next invocation.

Disable this with:

```bash
--no-resume
```

## Important implementation detail

This repository does **not** call MiniCPM-o's speech-generation path and does **not** merely generate audio and then hide/playback-disable it. The TTS module is not initialized at model load time, and text decoding calls the LLM backbone directly.

That distinction is central to the experiment: the available compute is spent on text-token ASR rather than speech-token decoding.

## Tests

CPU-only utility tests:

```bash
pytest -q
```

They cover WER computation and preservation of the final partial LibriSpeech audio chunk.

## Upstream references

- MiniCPM-o 4.5: https://huggingface.co/openbmb/MiniCPM-o-4_5
- LibriSpeech / OpenSLR 12: https://www.openslr.org/12
