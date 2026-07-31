#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/third_party/llama.cpp"

if [[ -d "$TARGET_DIR/.git" ]]; then
  git -C "$TARGET_DIR" pull --ff-only
else
  git clone https://github.com/ggml-org/llama.cpp "$TARGET_DIR"
fi

echo "llama.cpp is ready at $TARGET_DIR"
