#!/usr/bin/env bash

set -Eeuo pipefail

echo "=========================================="
echo " Starting Brain Tumor API Container"
echo "=========================================="

echo
echo "[1/2] Preparing model..."

/app/scripts/download_model.sh

echo
echo "[2/2] Starting FastAPI..."

exec uvicorn \
    api.main:app \
    --host 0.0.0.0 \
    --port 8000