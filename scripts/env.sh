#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  SCRIPT_SOURCE="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  SCRIPT_SOURCE="$(eval 'print -r -- ${(%):-%x}')"
else
  SCRIPT_SOURCE="$0"
fi
ROOT="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
export QWEN_TTS_IGPU_ROOT="$ROOT"
export VIRTUAL_ENV="$ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export UV_CACHE_DIR="$ROOT/.uv-cache"
