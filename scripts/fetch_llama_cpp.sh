#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/third_party/llama.cpp"
REVISION="$(tr -d '[:space:]' < "$ROOT_DIR/scripts/llama_cpp_revision.txt")"

if [[ -d "$TARGET_DIR/.git" ]]; then
  git -C "$TARGET_DIR" remote set-url origin https://github.com/ggml-org/llama.cpp.git
else
  if [[ -e "$TARGET_DIR" ]]; then
    echo "Error: $TARGET_DIR exists but is not a Git checkout" >&2
    exit 1
  fi
  mkdir -p "$TARGET_DIR"
  git -C "$TARGET_DIR" init
  git -C "$TARGET_DIR" remote add origin https://github.com/ggml-org/llama.cpp.git
fi

git -C "$TARGET_DIR" fetch --depth 1 origin "$REVISION"
git -C "$TARGET_DIR" checkout --detach FETCH_HEAD

ACTUAL_REVISION="$(git -C "$TARGET_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_REVISION" != "$REVISION" ]]; then
  echo "Error: expected llama.cpp $REVISION but checked out $ACTUAL_REVISION" >&2
  exit 1
fi

echo "llama.cpp $ACTUAL_REVISION is ready at $TARGET_DIR"
