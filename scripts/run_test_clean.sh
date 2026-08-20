#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_CUDA_ALLOC_CONF || true

# Fresh directory so rows from the streaming-text final-prefill regression are not resumed.
OUTPUT_DIR="results/test-clean-all-finalprefill-v3"

python -m minicpm_slack_asr.run \
  --dataset-root data/LibriSpeech/test-clean \
  --min-duration 0 \
  --max-samples 0 \
  --conditions baseline slack_2pass \
  --realtime \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

python -m minicpm_slack_asr.paper_table \
  --summary "${OUTPUT_DIR}/summary.json" \
  --output-dir "${OUTPUT_DIR}"
