#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Use a fresh directory so pre-fix runs that contained Qwen3 <think> output are never
# accidentally resumed into the corrected experiment.
OUTPUT_DIR="results/test-clean-ge10-nonthinking"

python -m minicpm_slack_asr.run \
  --dataset-root data/LibriSpeech/test-clean \
  --min-duration 10 \
  --max-samples 0 \
  --conditions baseline slack_2pass \
  --realtime \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

python -m minicpm_slack_asr.paper_table \
  --summary "${OUTPUT_DIR}/summary.json" \
  --output-dir "${OUTPUT_DIR}"
