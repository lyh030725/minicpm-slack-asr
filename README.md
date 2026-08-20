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

## Important: Qwen3 non-thinking decoding

Both the slack draft and final ASR use MiniCPM-o's upstream **non-thinking** generation prefix. MiniCPM-o's own `streaming_generate(enable_thinking=False, use_tts_template=False)` starts generation with:

```text
<|im_end|>
<|im_start|>assistant
<think>

</think>

```

The empty already-closed `<think>` block is required by Qwen3's hard non-thinking mode. Omitting it makes raw `model.llm` decoding enter `<think>...</think>` reasoning even when earlier `streaming_prefill()` calls used `enable_thinking=False`.

This project therefore builds the prefix from `model.think_str`, matching the upstream MiniCPM-o implementation. If generated draft/final token IDs unexpectedly contain `<think>` or `</think>`, the run raises an error rather than silently contaminating WER.

The prompt also explicitly requires transcript-only output with no `Transcription:` label, preamble, explanation, or reasoning.

## Dataset

The default experiment uses **every LibriSpeech ASR `test-clean` utterance whose duration is greater than or equal to 10 seconds**.

```text
min_duration >= 10.0 s
max_samples = 0   # all qualifying utterances
```

The final partial second is zero-padded so no real speech samples are dropped.

## Conditions

### `baseline`

1. Stream the complete audio in 1-second chunks.
2. Do no intermediate decoding.
3. After the complete audio is available, generate one final verbatim transcript with the LLM backbone only.

### `slack_2pass`

1. Stream the same audio in 1-second chunks.
2. After each non-final chunk, compute the time remaining until the next 1-second deadline.
3. Use only that slack for tentative LLM text decoding.
4. Roll the LLM KV cache back to the exact post-audio-prefill state.
5. Keep only the tentative transcript externally.
6. After the complete utterance, give the fallible draft back to the LLM and produce a corrected final transcript.

The main comparison is final WER of `baseline` vs. `slack_2pass` on the identical >=10 s subset.

## Draft KV isolation

```text
                         temporary branch
                              |
audio KV after chunk ----------+---- non-thinking assistant prefix + tentative text
       |
       |                             decode inside slack
       |                                   |
       +<--------- truncate KV ------------+
       |
next audio chunk
```

For every successful speculative step:

```text
kv_cache_before_draft == kv_cache_after_restore
```

The run raises an error if this invariant fails.

## Slack policy

For each non-final 1-second chunk:

```text
slack = next_audio_deadline - audio_prefill_finish_time
```

Tentative decoding begins only when at least `--min-slack-ms` remains (50 ms by default). Generation is token-by-token greedy decoding. A latency EMA with a 1.2x guard prevents intentionally starting a token that is expected to cross the next audio deadline.

The first-pass semantic cap is 12 new tokens per audio chunk by default:

```bash
--max-draft-tokens-per-chunk N
```

Up to 256 recent draft tokens are re-prefilled as the tentative assistant prefix:

```bash
--max-draft-prefix-tokens N
```

## RunPod environment

Target image:

```text
runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404
```

Install:

```bash
bash scripts/setup_runpod.sh
```

Recommended cache variables:

```bash
export HF_HOME=/workspace/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Download LibriSpeech:

```bash
bash scripts/download_librispeech.sh
```

Run the complete experiment:

```bash
bash scripts/run_test_clean.sh
```

The fixed run script deliberately writes to a fresh directory:

```text
results/test-clean-ge10-nonthinking/
```

This prevents pre-fix rows containing Qwen3 reasoning output from being resumed into the corrected experiment.

Equivalent command:

```bash
python -m minicpm_slack_asr.run \
  --dataset-root data/LibriSpeech/test-clean \
  --min-duration 10 \
  --max-samples 0 \
  --conditions baseline slack_2pass \
  --realtime \
  --output-dir results/test-clean-ge10-nonthinking
```

For a short GPU sanity check before the full run:

```bash
python -m minicpm_slack_asr.run \
  --dataset-root data/LibriSpeech/test-clean \
  --min-duration 10 \
  --max-samples 3 \
  --conditions baseline slack_2pass \
  --no-realtime \
  --no-resume \
  --output-dir results/sanity-nonthinking
```

A healthy console result should contain only transcript text in `final=...`; it should not contain `<think>`, `Transcription is:`, or reasoning prose.

## Outputs

```text
results/test-clean-ge10-nonthinking/
├── manifest.csv
├── samples.csv
├── chunks.csv
├── summary.json
├── environment.json
├── paper_table.md
└── paper_table.tex
```

### `samples.csv`

One row per sample/condition, including:

- LibriSpeech reference text
- tentative first-pass draft (`slack_2pass` only)
- final transcript
- first-pass WER
- final WER
- substitutions / deletions / insertions
- audio-prefill time
- draft-compute time
- final decode time
- deadline-miss chunks
- total draft tokens

### `chunks.csv`

Per streaming chunk diagnostics include:

- audio prefill time
- slack before draft decoding
- `draft_new_text`
- `draft_so_far`
- draft prefix/decode time
- deadline remaining time
- deadline miss
- KV length before draft and after rollback

### `summary.json`

Reports micro-WER for `baseline` and `slack_2pass` plus relative/absolute WER changes.

### Paper table

After `scripts/run_test_clean.sh` finishes, `paper_table.md` and `paper_table.tex` compare:

1. MiniCPM-o 4.5 official LibriSpeech test-clean WER (1.40%, full test-clean, reference only),
2. our baseline on test-clean >=10 s,
3. our Slack 2-pass result on the same >=10 s subset.

The official 1.40% number is not an apples-to-apples subset comparison; the controlled comparison is our baseline vs. Slack 2-pass.

## Resume behavior

`--resume` is enabled by default. Completed `(sample_id, condition)` pairs in the selected output directory are skipped. Use `--no-resume` to restart that output directory from scratch.

## Tests

CPU-only utility tests:

```bash
pytest -q
```

Tests cover WER, LibriSpeech duration selection/tail preservation, paper-table generation, and the upstream-compatible Qwen3 non-thinking assistant prefix.

## Upstream references

- MiniCPM-o 4.5: https://huggingface.co/openbmb/MiniCPM-o-4_5
- Qwen3-8B: https://huggingface.co/Qwen/Qwen3-8B
- LibriSpeech / OpenSLR 12: https://www.openslr.org/12
