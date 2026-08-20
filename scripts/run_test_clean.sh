#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m minicpm_slack_asr.run \
  --dataset-root data/LibriSpeech/test-clean \
  --min-duration 10 \
  --max-samples 0 \
  --conditions baseline slack_2pass \
  --realtime \
  --output-dir results/test-clean-gt10 \
  "$@"
