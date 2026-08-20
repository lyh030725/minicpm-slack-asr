#!/usr/bin/env bash
set -euo pipefail

TARGET_IMAGE="runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404"

echo "[setup] Target RunPod image: ${TARGET_IMAGE}"
echo "[setup] Preserving the image-provided PyTorch/CUDA stack."

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  ffmpeg \
  git \
  libsndfile1 \
  wget
rm -rf /var/lib/apt/lists/*

python scripts/check_env.py --pre-install
python -m pip install --upgrade pip setuptools wheel
python -m pip install -c constraints-runpod.txt -r requirements.txt
python -m pip install --no-deps -e .

mkdir -p /workspace/.cache/huggingface
cat <<'ENV_HINT'

[setup] Recommended environment:
  export HF_HOME=/workspace/.cache/huggingface
  export HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENV_HINT

python scripts/check_env.py
python - <<'PY'
import torch
if not str(torch.__version__).startswith("2.9.1"):
    raise SystemExit(f"[setup] ERROR: expected image PyTorch 2.9.1, got {torch.__version__}")
print(f"[setup] PyTorch preserved: {torch.__version__}; CUDA={torch.version.cuda}")
PY

echo "[setup] Done."
