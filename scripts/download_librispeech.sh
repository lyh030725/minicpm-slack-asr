#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data}"
ARCHIVE="${ROOT}/test-clean.tar.gz"
DATASET_DIR="${ROOT}/LibriSpeech/test-clean"
URL="https://www.openslr.org/resources/12/test-clean.tar.gz"

mkdir -p "${ROOT}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  if [[ ! -f "${ARCHIVE}" ]]; then
    echo "[data] Downloading LibriSpeech test-clean..."
    wget -c "${URL}" -O "${ARCHIVE}"
  fi
  echo "[data] Extracting ${ARCHIVE}..."
  tar -xzf "${ARCHIVE}" -C "${ROOT}"
else
  echo "[data] ${DATASET_DIR} already exists; skipping download/extract."
fi

python -m minicpm_slack_asr.manifest \
  --dataset-root "${DATASET_DIR}" \
  --min-duration 10 \
  --max-samples 0 \
  --output data/manifests/test-clean-ge10.csv
