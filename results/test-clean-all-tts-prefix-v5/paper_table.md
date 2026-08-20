# ASR Results Table

| Method | Evaluation set | Samples | WER (%) ↓ | Rel. WER Reduction (%) ↑ |
|---|---|---:|---:|---:|
| MiniCPM-o 4.5 official | LibriSpeech test-clean | 2620 | 1.40 | — |
| Baseline (ours) | LibriSpeech test-clean | 2620 | 2.10 | — |
| Slack 2-pass (ours) | LibriSpeech test-clean | 2620 | 2.75 | -30.95 |

**Note.** All rows use the standard LibriSpeech test-clean split. The official MiniCPM-o 4.5 value (1.40% WER) is included as a reference; the primary controlled comparison is still Baseline (ours) vs. Slack 2-pass (ours), because they use the same inference implementation and differ only in the listening-slack two-pass method. Relative WER reduction is computed against Baseline (ours).
