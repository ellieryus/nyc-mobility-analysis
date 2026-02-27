#!/usr/bin/env zsh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN="/Users/heliamahmoodzadeh/Documents/GitHub/nyc-mobility-analysis/.venv/bin/python"
MPLCONFIGDIR="/Users/heliamahmoodzadeh/Documents/New project/.mplconfig" "$PY_BIN" pipelines/run_pipeline.py
