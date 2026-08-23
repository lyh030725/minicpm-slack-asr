#!/usr/bin/env bash
set -euo pipefail

TARGET_IMAGE="runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404"
OPENBMB_ULTRAEVAL_COMMIT="bbc07b1effc03a85006c36dc765b8f2b8eae8d36"

echo "[setup] Target RunPod image: ${TARGET_IMAGE}"
echo "[setup] Preserving the image-provided PyTorch/CUDA stack."
echo "[setup] OpenBMB UltraEval-Audio scorer: ${OPENBMB_ULTRAEVAL_COMMIT}"

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

# Install only the pinned OpenBMB scorer source. Its minimal runtime dependencies
# are already pinned in requirements.txt; --no-deps avoids UltraEval's unrelated
# benchmark/model dependencies. The archive route also avoids submodule issues.
python -m pip install --no-deps \
  "https://github.com/OpenBMB/UltraEval-Audio/archive/${OPENBMB_ULTRAEVAL_COMMIT}.tar.gz"

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

# Fail setup immediately if the exact scorer imports are unavailable.
python - <<'PY'
from minicpm_slack_asr.wer import OPENBMB_ULTRAEVAL_COMMIT, normalize_for_wer

assert normalize_for_wer("twenty") == normalize_for_wer("20")
print(f"[setup] OpenBMB WER scorer ready: {OPENBMB_ULTRAEVAL_COMMIT}")
PY

echo "[setup] Done."
