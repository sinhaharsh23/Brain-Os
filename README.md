# BrainOS

BrainOS is a local Transformer observatory. It runs `Qwen/Qwen2.5-0.5B-Instruct` with PyTorch and exposes observable computation through a real-time React/WebGL interface.

## Features

- Real tokenizer IDs, positions, special tokens, and token timing
- Real embeddings with PCA coordinates and vector statistics
- Real hidden states, attention weights, Q/K/V projections, MLP activations, logits, and probabilities
- Token-by-token generation over WebSocket
- Interactive 3D transformer stack with token nodes, attention links, and activation intensity
- Attention heatmap, embedding explorer, architecture view, token flow, developer mode, and replay
- CPU fallback, CUDA detection, RAM/VRAM monitoring, and model recommendations
- On-demand tensor inspection APIs to avoid continuously sending huge tensors to the browser

BrainOS does not expose or claim to expose private chain-of-thought. It visualizes inspectable model tensors and labels derived educational interpretations accordingly.

## Hardware

The default model is approximately 494M parameters and is selected for small local systems. CPU inference is supported. A CUDA device is used automatically when available. Larger Qwen2.5 models are listed in the model explorer but should only be loaded when the machine has enough RAM or VRAM.

## Quick Start

```bash
cd BrainOS
python3 -m pip install --user --break-system-packages virtualenv
python3 -m virtualenv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
cd frontend && npm install && npm run build
cd ../backend && HF_HOME=../models/hf .venv/bin/python run.py
```

Open `http://127.0.0.1:8765`. The first run may download the model from Hugging Face.

For frontend development, run the backend on port `8765`, then run `npm run dev` in `frontend` and open the Vite URL.

## Tests

```bash
backend/.venv/bin/python -m pytest backend/tests -q
cd frontend && npm run build
```

## Layout

- `backend/app/models`: adapter abstraction and Qwen implementation
- `backend/app/inference`: KV-cache generation and capture orchestration
- `backend/app/instrumentation`: hooks and tensor statistics
- `backend/app/events`: real-time event types and event bus
- `backend/app/api`: REST inspection endpoints
- `frontend/src/brain3d`: React Three Fiber visualization
- `frontend/src/components`: inspectors, controls, monitoring, probability, and replay UI
- `frontend/src/views`: architecture, attention, embedding, token, and developer views

See `DEVELOPMENT.md`, `MODEL_GUIDE.md`, `docs/API.md`, and `TROUBLESHOOTING.md` for details.
