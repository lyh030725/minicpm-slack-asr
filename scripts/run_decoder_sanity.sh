#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_CUDA_ALLOC_CONF || true

python -m minicpm_slack_asr.diagnose \
  --dataset-root data/LibriSpeech/test-clean \
  --max-samples 20 \
  --conditions baseline slack_2pass \
  --realtime \
  --output-dir results/tts-prefix-ab-sanity \
  "$@"
