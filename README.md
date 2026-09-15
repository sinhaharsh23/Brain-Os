# BrainOS

BrainOS is a local Transformer observatory. It runs `Qwen/Qwen2.5-0.5B-Instruct` with PyTorch and exposes observable computation through a real-time React/WebGL interface.

## Features

- Real tokenizer IDs, positions, special tokens, and token timing
- Real embeddings with PCA coordinates and vector statistics
- Real hidden states, attention weights, Q/K/V projections, MLP activations, logits, and probabilities
- Architecture adapters for Qwen2, Llama-compatible, Mistral, and Gemma Hugging Face checkpoints
- Token-by-token generation over WebSocket
- Interactive 3D transformer stack with token nodes, attention links, and activation intensity
- Real top-K MLP activation units rendered as selectable 3D neuron nodes
- Attention heatmap, embedding explorer, architecture view, token flow, developer mode, and replay
- Inference pause/resume between real model steps, replay stepping, and timeline scrubbing
- Pause state is event-driven: an in-flight forward pass finishes safely, then the next step waits
- Apple M5 Pro / MPS as the primary runtime, with CPU fallback, CUDA/ROCm detection, unified-memory monitoring, and model recommendations
- On-demand tensor inspection APIs to avoid continuously sending huge tensors to the browser
- Persistent replay tensor bundles restored after backend restart
- Server-side external observation adapters for OpenAI, Anthropic, and Gemini when credentials are configured
- Optional bearer/token authentication, rate limiting, strict configurable CORS, and per-connection WebSocket isolation

BrainOS does not expose or claim to expose private chain-of-thought. It visualizes inspectable model tensors and labels derived educational interpretations accordingly.

## Hardware

The default model is approximately 494M parameters and is selected for the current Apple M5 Pro unified-memory environment. Apple MPS is selected automatically when available, with CPU fallback. NVIDIA CUDA and AMD ROCm remain additional compatibility targets. Larger Qwen2.5 models are listed in the model explorer but should only be loaded when the machine has enough RAM or VRAM.

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

For a protected deployment, set `BRAINOS_AUTH_TOKEN` on the backend and the matching `VITE_BRAINOS_TOKEN` when building the frontend. HTTP API calls use `X-BrainOS-Token`; WebSocket connections use the token query parameter. Session inference and replay events are routed only to the owning WebSocket connection.

To run credential-gated external provider tests, configure the required server-side key and explicitly opt in:

```bash
BRAINOS_LIVE_EXTERNAL_TESTS=1 OPENAI_API_KEY=... backend/.venv/bin/python -m pytest backend/tests/test_external_providers.py -q
```

The default test suite never sends requests to external providers.

## Tests

```bash
backend/.venv/bin/python -m pytest backend/tests -q
(cd frontend && npm run build)
backend/.venv/bin/python scripts/local_smoke.py
(cd frontend && npm run test:browser)
backend/.venv/bin/python scripts/benchmark_local.py --repeats 3 --clients 1 --max-new-tokens 4
```

Browser tests require the local backend to be running at `http://127.0.0.1:8765` and use the installed system Chrome executable. They exercise the actual built UI and local Qwen WebSocket path; no paid provider credentials are needed.

`scripts/benchmark_local.py` records real WebSocket TTFT, total latency, event rate, server-reported tokens/sec, hardware backend, and the latest monitoring snapshot. Its output is labeled `LOCAL M5 PRO BENCHMARK`; it is not a production-scale load test.

## Layout

- `backend/app/models`: adapter abstraction and Qwen implementation
- `backend/app/inference`: KV-cache generation and capture orchestration
- `backend/app/instrumentation`: hooks and tensor statistics
- `backend/app/events`: real-time event types and event bus
- `backend/app/api`: REST inspection endpoints
- `frontend/src/brain3d`: React Three Fiber visualization
- `frontend/src/components`: inspectors, controls, monitoring, probability, and replay UI
- `frontend/src/views`: architecture, attention, embedding, token, and developer views

Live sessions report time-to-first-token, generation speed, context usage, and total generation time. Replay controls operate on recorded event indices; they do not fabricate intermediate model states.

See `DEVELOPMENT.md`, `MODEL_GUIDE.md`, `docs/API.md`, and `TROUBLESHOOTING.md` for details.
