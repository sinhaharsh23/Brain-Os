# BrainOS 3.0 — Apple Silicon Setup

This document provides complete instructions for running BrainOS 3.0 on Apple Silicon (M-series Macs, including M5 / M5 Pro, M4, M3, M2, M1).

## Architecture Overview

BrainOS 3.0 runs natively on macOS `arm64`:
- **Inference Engine**: PyTorch with Apple Silicon Metal Performance Shaders (`mps`) backend for hardware acceleration, falling back to optimized CPU inference when needed.
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct` stored locally in `models/hf/` (no internet connection required once cached).
- **Backend**: FastAPI + Uvicorn with WebSocket event streaming on port `8765`.
- **Frontend**: React + Three.js / React Three Fiber WebGL 3D transformer stack visualization.
  - Can be accessed via the integrated backend build at `http://127.0.0.1:8765/`
  - Or via Vite hot-reloading dev server at `http://localhost:5173/`

## Prerequisites

1. **macOS**: macOS 12+ (tested on macOS 26 / Darwin 25.6.0 arm64).
2. **Python**: Python 3.9+ (`arm64`).
3. **Node.js**: Node v18+ and `npm` (tested on Node v26, npm 11).
4. **Hugging Face CLI** (optional): `hf` or `huggingface-cli` authenticated via `hf auth whoami`.

## Environment Setup

### 1. Configuration (`.env`)

A project-relative `.env` file is loaded by the backend:

```env
BRAINOS_HOST=127.0.0.1
BRAINOS_PORT=8765
BRAINOS_DEFAULT_MODEL=Qwen/Qwen2.5-0.5B-Instruct
BRAINOS_MAX_NEW_TOKENS_DEFAULT=128
BRAINOS_MAX_PROMPT_TOKENS=1024
BRAINOS_DEVICE=auto
BRAINOS_DTYPE=auto
BRAINOS_CAPTURE_INTERVAL_MS=500
BRAINOS_REPLAY_DIR=./backend/replay_sessions
BRAINOS_CORS_ORIGINS=http://127.0.0.1:8765,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8765
BRAINOS_MAX_PROMPT_CHARS=12000
BRAINOS_MAX_NEW_TOKENS_LIMIT=512
HF_HOME=./models/hf
```

`BRAINOS_DEVICE=auto` automatically detects Apple Silicon MPS acceleration.

### 2. Python Virtual Environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm ci
npm run build
```

## Current MPS Status

The primary development machine is a physical MacBook Pro with Apple M5 Pro (`arm64`). Apple MPS support is implemented and verified for the default Qwen/Qwen2.5-0.5B-Instruct inference and tensor-capture path. The checks below are useful for diagnosing a new checkout or a changed Python environment; they are not a substitute for the current machine baseline.

## Checking Apple Silicon MPS Acceleration

Run the diagnostic script:

```bash
./scripts/diagnose-mac.sh
```

Or verify PyTorch MPS directly:

```bash
backend/.venv/bin/python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## Running BrainOS

### Option A: One-Command Startup (Recommended)

To start both the backend and frontend dev server with a single command:

```bash
cd ~/Documents/BrainOS
./scripts/start-brainos.sh
```

Press `Ctrl+C` to cleanly stop all services.

### Option B: Start Separately

**Terminal 1 (Backend):**
```bash
cd ~/Documents/BrainOS
./scripts/start-backend.sh
```

**Terminal 2 (Frontend Dev Server):**
```bash
cd ~/Documents/BrainOS
./scripts/start-frontend.sh
```

## Verification & Testing

1. **Smoke Test (WebSocket + Introspection APIs):**
   ```bash
   backend/.venv/bin/python scripts/local_smoke.py
   ```

2. **Backend Unit & Integration Test Suite:**
   ```bash
   backend/.venv/bin/python -m pytest backend/tests -q
   ```

3. **Browser End-to-End Test Suite:**
   ```bash
   cd frontend
   npx playwright test
   ```

## Common Issues & Troubleshooting

- **Port 8765 already in use**:
  ```bash
  lsof -ti :8765 | xargs kill -9
  ```
- **Hugging Face Cache Symlinks**:
  If copying the repository from another machine, ensure symlinks in `models/hf/hub/.../snapshots` point to valid blob files rather than zero-byte regular files. Run `./scripts/diagnose-mac.sh` to verify cache integrity.
- **Vite Proxy**:
  Vite proxies `/api` and `/ws` to `http://127.0.0.1:8765`. Ensure the backend is started before running Vite dev.
