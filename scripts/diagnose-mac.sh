#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "===================================================="
echo "         BRAINOS 3.0 — MAC ENVIRONMENT DIAGNOSTIC"
echo "===================================================="

# OS & Hardware
echo "[1] System & Hardware:"
echo "    macOS:        $(sw_vers -productVersion 2>/dev/null || uname -s)"
echo "    Architecture: $(uname -m)"
echo "    CPU Brand:    $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -p)"
echo "    Kernel:       $(uname -r)"

# Node environment
echo ""
echo "[2] Node & Frontend:"
if command -v node >/dev/null 2>&1; then
    echo "    Node:         $(node --version)"
else
    echo "    Node:         NOT FOUND"
fi
if command -v npm >/dev/null 2>&1; then
    echo "    npm:          $(npm --version)"
else
    echo "    npm:          NOT FOUND"
fi
if [ -d "$ROOT_DIR/frontend/node_modules" ]; then
    echo "    node_modules: Installed ($(find "$ROOT_DIR/frontend/node_modules/.bin" -maxdepth 1 -type l 2>/dev/null | wc -l | tr -d ' ') binary symlinks)"
else
    echo "    node_modules: NOT INSTALLED"
fi
if [ -d "$ROOT_DIR/frontend/dist" ]; then
    echo "    frontend dist: Built (SPA bundle present)"
else
    echo "    frontend dist: Not built"
fi

# Python environment
echo ""
echo "[3] Python & Virtual Environment:"
VENV_PYTHON="$ROOT_DIR/backend/.venv/bin/python"
if [ -f "$VENV_PYTHON" ]; then
    echo "    venv path:    $ROOT_DIR/backend/.venv"
    echo "    venv Python:  $($VENV_PYTHON --version 2>&1)"
    echo "    venv Arch:    $($VENV_PYTHON -c 'import platform; print(platform.machine())' 2>&1)"
else
    echo "    venv Python:  NOT FOUND at $VENV_PYTHON"
fi

# PyTorch & Apple Silicon MPS
echo ""
echo "[4] PyTorch & Apple Silicon MPS:"
if [ -f "$VENV_PYTHON" ]; then
    $VENV_PYTHON - <<'PY'
import sys
try:
    import torch
    print(f"    PyTorch:      {torch.__version__}")
    print(f"    MPS built:    {torch.backends.mps.is_built()}")
    print(f"    MPS available:{torch.backends.mps.is_available()}")
    print(f"    CUDA avail:   {torch.cuda.is_available()}")
    if torch.backends.mps.is_available():
        x = torch.randn((64, 64), device="mps")
        y = x @ x
        print(f"    MPS compute:  PASS (tensor shape: {list(y.shape)})")
    else:
        print("    MPS compute:  SKIPPED (MPS not available)")
except Exception as e:
    print(f"    PyTorch error:{e}")
PY
fi

# Hugging Face & Model Cache
echo ""
echo "[5] Model & Hugging Face Cache:"
if [ -f "$ROOT_DIR/.env" ]; then
    echo "    .env file:    Present ($ROOT_DIR/.env)"
else
    echo "    .env file:    NOT FOUND (using defaults)"
fi
if command -v hf >/dev/null 2>&1; then
    echo "    HF CLI:       $(which hf) (whoami: $(hf auth whoami 2>&1 | tr '\n' ' '))"
elif [ -f "$ROOT_DIR/backend/.venv/bin/hf" ]; then
    echo "    HF CLI (venv):$ROOT_DIR/backend/.venv/bin/hf"
else
    echo "    HF CLI:       NOT FOUND"
fi

MODEL_DIR="$ROOT_DIR/models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
if [ -d "$MODEL_DIR" ]; then
    echo "    Model Cache:  Present ($MODEL_DIR)"
    SNAP="$MODEL_DIR/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    if [ -f "$SNAP/config.json" ] && [ -s "$SNAP/config.json" ] && [ -f "$SNAP/model.safetensors" ] && [ -s "$SNAP/model.safetensors" ]; then
        echo "    Model Files:  Valid (safetensors size: $(ls -lh "$SNAP/model.safetensors" | awk '{print $5}'))"
    else
        echo "    Model Files:  INVALID or empty!"
    fi
else
    echo "    Model Cache:  Not found in $ROOT_DIR/models/hf"
fi

# Ports & Connectivity
echo ""
echo "[6] Ports & Services:"
if lsof -Pi :8765 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "    Port 8765:    IN USE by process $(lsof -Pi :8765 -sTCP:LISTEN -t | head -n 1)"
else
    echo "    Port 8765:    AVAILABLE"
fi
if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "    Port 5173:    IN USE by process $(lsof -Pi :5173 -sTCP:LISTEN -t | head -n 1)"
else
    echo "    Port 5173:    AVAILABLE"
fi

echo "===================================================="
