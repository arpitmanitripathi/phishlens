#!/usr/bin/env bash
# Publish the app to Hugging Face Spaces.
#
#   scripts/deploy_space.sh <hf-username>/phishlens
#
# Copies only what the Docker image needs (code, models, Dockerfile, lock file) into a temporary
# folder, puts the Space card (deploy/huggingface/README.md) in as README.md, and uploads it.
# Hugging Face then builds the Docker image and starts the app.
set -euo pipefail

SPACE="${1:?usage: scripts/deploy_space.sh <hf-username>/phishlens}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$ROOT"/Dockerfile "$ROOT"/.dockerignore "$ROOT"/pyproject.toml "$ROOT"/uv.lock "$STAGE"/
cp -R "$ROOT"/src "$ROOT"/app "$ROOT"/models "$STAGE"/
cp "$ROOT"/deploy/huggingface/README.md "$STAGE"/README.md
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} +
rm -f "$STAGE"/models/.gitkeep

HF="uvx --from huggingface_hub hf"
$HF repos create "$SPACE" --repo-type space --space-sdk docker --exist-ok
$HF upload "$SPACE" "$STAGE" . --repo-type space \
    --commit-message "Deploy $(git -C "$ROOT" rev-parse --short HEAD)"
echo "Deployed: https://huggingface.co/spaces/$SPACE"
