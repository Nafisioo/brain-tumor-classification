#!/usr/bin/env bash

set -Eeuo pipefail

####################################################
# Brain Tumor API
# Model Downloader
####################################################

MODEL_VERSION="v1.0.0"
MODEL_NAME="resnet18_finetuned_best.pt"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/artifacts"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
TEMP_MODEL="${MODEL_PATH}.download"

# Wrapped in a proxy mirror to prevent unexpected EOF errors and throttling
MODEL_URL="https://mirror.ghproxy.com/https://github.com/Nafisioo/brain-tumor-classification/releases/download/${MODEL_VERSION}/${MODEL_NAME}"

EXPECTED_SHA256="a1d3a6efb01c6dd1a997aa9f0ac117a0e50c8d09d8e851e13221c5e85ef9d73d"

####################################################

echo "========================================="
echo " Brain Tumor API"
echo " Model Downloader"
echo "========================================="
echo "Version : ${MODEL_VERSION}"
echo "Model   : ${MODEL_NAME}"
echo

####################################################
# Dependencies
####################################################

command -v curl >/dev/null || {
    echo "ERROR: curl is not installed."
    exit 1
}

command -v sha256sum >/dev/null || {
    echo "ERROR: sha256sum is not installed."
    exit 1
}

####################################################
# Prepare directory
####################################################

mkdir -p "${MODEL_DIR}"

####################################################
# Checksum helper
####################################################

verify_checksum() {
    local file="$1"
    echo "${EXPECTED_SHA256}  ${file}" | sha256sum -c - >/dev/null 2>&1
}

####################################################
# Cached model
####################################################

if [[ -f "${MODEL_PATH}" ]]; then
    echo "Existing model detected."
    if verify_checksum "${MODEL_PATH}"; then
        echo "✓ SHA256 verified."
        echo "Using cached model."
        exit 0
    fi
    echo "Cached model failed checksum."
    rm -f "${MODEL_PATH}"
fi

####################################################
# Download
####################################################

echo
echo "Downloading model..."
echo "${MODEL_URL}"
echo

curl \
    --fail \
    --location \
    --continue-at - \
    --retry 10 \
    --retry-delay 5 \
    --retry-all-errors \
    --output "${TEMP_MODEL}" \
    "${MODEL_URL}"

####################################################
# Verify downloaded file
####################################################

echo
echo "Verifying SHA256..."

if ! verify_checksum "${TEMP_MODEL}"; then
    echo "ERROR: checksum verification failed."
    rm -f "${TEMP_MODEL}"
    exit 1
fi

####################################################
# Atomic move
####################################################

mv "${TEMP_MODEL}" "${MODEL_PATH}"

echo
echo "✓ Model downloaded successfully."
echo "✓ SHA256 verified."

ls -lh "${MODEL_PATH}"

echo
echo "Startup can continue."