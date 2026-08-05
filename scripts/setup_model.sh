#!/usr/bin/env bash

set -Eeuo pipefail


echo "Preparing local model..."

./scripts/download_model.sh

echo "Local model ready."