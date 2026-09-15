#!/usr/bin/env bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/backend"

if [ ! -f ".venv/bin/python" ]; then
    echo "Error: backend/.venv not found. Please create it first."
    exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT_DIR/models/hf}"
exec .venv/bin/python run.py
