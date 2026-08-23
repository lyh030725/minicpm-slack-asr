# ASR Results Re-scored with OpenBMB UltraEval-Audio WER

Pinned scorer: `OpenBMB/UltraEval-Audio@bbc07b1effc03a85006c36dc765b8f2b8eae8d36`

| Method | Samples | WER (%) ↓ | S | D | I | Total edit distance | Rel. WER Reduction (%) ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (ours) | 2620 | 1.718 | 656 | 132 | 123 | 911 | — |
| Slack 2-pass (ours) | 2620 | 2.399 | 875 | 211 | 186 | 1272 | -39.63 |

**Scoring pipeline.** OpenBMB UltraEval-Audio `PracticeWER`: lowercase → `EnglishTextNormalizer` → `EvaluationTokenizer(13a, lowercase=True, punctuation_removal=False, character_tokenization=False)` → corpus edit distance / normalized reference-token count.

**Caveat.** This is an exact reproduction of the pinned public UltraEval-Audio English WER implementation, not a claim that MiniCPM-o 4.5 model-card WER 1.40 used this exact scorer commit.
