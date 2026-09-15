#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "===================================================="
echo "         Starting BrainOS 3.0 on Apple Silicon"
echo "===================================================="

echo "Project Root: $ROOT_DIR"

if [ ! -f "$ROOT_DIR/backend/.venv/bin/python" ]; then
    echo "ERROR: backend/.venv not found!"
    exit 1
fi
PYTHON="$ROOT_DIR/backend/.venv/bin/python"

if [ -f "$ROOT_DIR/.env" ]; then
    echo "Environment file: Loaded (.env)"
fi

export HF_HOME="${HF_HOME:-$ROOT_DIR/models/hf}"
echo "HF_HOME: $HF_HOME"

DEVICE=$($PYTHON -c "import torch; print('MPS (Apple Silicon GPU)' if torch.backends.mps.is_available() else ('CUDA' if torch.cuda.is_available() else 'CPU'))" 2>/dev/null || echo "CPU")
echo "Compute Device: $DEVICE"

if [ ! -d "$ROOT_DIR/frontend/dist" ]; then
    echo "Building frontend..."
    (cd "$ROOT_DIR/frontend" && npm run build)
fi

cleanup() {
    echo ""
    echo "Shutting down BrainOS..."
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "BrainOS stopped cleanly."
}
trap cleanup SIGINT SIGTERM EXIT

# If port 8765 is already occupied by a previous backend, stop it
OLD_PID=$(lsof -Pi :8765 -sTCP:LISTEN -t 2>/dev/null || true)
if [ -n "$OLD_PID" ]; then
    echo "Stopping existing process on port 8765 (PID: $OLD_PID)..."
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
fi

echo ""
echo "Starting BrainOS backend server..."
(cd "$ROOT_DIR/backend" && exec "$PYTHON" run.py) &
BACKEND_PID=$!

echo "Waiting for backend to be ready on port 8765..."
READY=0
for i in {1..30}; do
    if curl -s http://127.0.0.1:8765/api/ready 2>/dev/null | grep -q "ready"; then
        READY=1
        break
    fi
    sleep 1
done

if [ $READY -eq 1 ]; then
    echo "Backend is READY! (PID: $BACKEND_PID)"
else
    echo "Backend started (PID: $BACKEND_PID). Model loading in progress..."
fi

echo ""
echo "Starting Vite frontend dev server..."
(cd "$ROOT_DIR/frontend" && exec npm run dev) &
FRONTEND_PID=$!

echo ""
echo "===================================================="
echo "BrainOS 3.0 is running!"
echo "   Observatory UI (Integrated): http://127.0.0.1:8765"
echo "   Frontend Dev Server (Vite):  http://localhost:5173"
echo "   Backend REST API:           http://127.0.0.1:8765/api"
echo "   WebSocket Stream:           ws://127.0.0.1:8765/ws"
echo ""
echo "Press Ctrl+C to stop all services."
echo "===================================================="

wait
