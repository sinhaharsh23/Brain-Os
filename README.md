# BrainOS

BrainOS is a local-first Transformer observatory. It runs supported Hugging Face causal models, including the default `Qwen/Qwen2.5-0.5B-Instruct`, and can deeply inspect installed Ollama GGUF models through an instrumented llama.cpp runner. A real-time React/Three.js interface displays the computation each runtime exposes.

## Features

- Real tokenizer IDs, positions, special tokens, and token timing
- Real embeddings with PCA coordinates and vector statistics
- Real hidden states, attention weights, Q/K/V projections, MLP activations, logits, and probabilities
- Architecture adapters for Qwen2, Llama-compatible, Mistral, and Gemma Hugging Face checkpoints
- Deep GGUF inspection for installed Ollama models through BrainOS' instrumented llama.cpp runner
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

BrainOS reports observable model telemetry only. It does not expose private chain-of-thought or infer hidden reasoning text.

## Hardware

The default checkpoint is `Qwen/Qwen2.5-0.5B-Instruct`. Parameter count, architecture, context, runtime dtype, and device are read from the loaded model. Apple MPS is selected when available, with CPU fallback. NVIDIA CUDA and AMD ROCm remain additional compatibility targets.

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

## Providers

**Hugging Face Local** loads a supported local checkpoint and tokenizer. BrainOS reports the applied chat template and captures model-produced token IDs, embeddings, hidden states, attention, Q/K/V, MLP and residual activations, logits, probabilities, and KV-cache tensors when those captures are enabled.

**Ollama** uses the selected installed GGUF checkpoint through BrainOS' instrumented llama.cpp runner. It can capture the GGUF model's real token IDs, embeddings, attention, Q/K/V, layer outputs, MLP activations, logits, probabilities, and generated-token flow. The runner is built on first use and requires the Ollama CLI, Homebrew `llama.cpp` and `ggml`, and a C++17 compiler. Set `OLLAMA_GGUF_PATH` to use a specific GGUF file directly. The GGUF path currently does not expose PyTorch logit-lens or tensor KV-cache views.

**OpenAI, Anthropic, and Gemini** use their provider APIs. BrainOS shows only returned response text, stream events, timing, usage, model identity, and errors. Their internal transformer tensors are not available through these adapters.

Set `OLLAMA_URL` and `OLLAMA_MODELS` (comma-separated model names installed in Ollama); the default URL is `http://127.0.0.1:11434`. Ollama is the runtime provider, while the selected tag (for example, `llama3.2:3b`) is the model identity. Its sampling options are the options BrainOS sends with the request.

For a protected deployment, set `BRAINOS_AUTH_TOKEN` on the backend and the matching `VITE_BRAINOS_TOKEN` when building the frontend. HTTP API calls use `X-BrainOS-Token`; WebSocket connections use the token query parameter. Session inference and replay events are routed only to the owning WebSocket connection.

To run credential-gated external provider tests, configure the required server-side key and explicitly opt in:

```bash
BRAINOS_LIVE_EXTERNAL_TESTS=1 OPENAI_API_KEY=... backend/.venv/bin/python -m pytest backend/tests/test_external_providers.py -q
```

The default test suite never sends requests to external providers.

## Architecture and deployment

The React/Three.js frontend uses REST for catalogs, health, replay, and on-demand inspection, and WebSocket for generation events, telemetry, and cancellation. The backend routes session events to their owning WebSocket and stores replay events and tensor captures for later inspection. See [ARCHITECTURE.md](ARCHITECTURE.md) and [docs/BRAINOS_ARCHITECTURE.md](docs/BRAINOS_ARCHITECTURE.md) for the runtime boundaries and data flow.

The repository does not contain Vercel deployment configuration, and the Vercel `brain-os` project is disconnected from this GitHub repository. The supported deployment path is Docker Compose, with a WebSocket-capable reverse proxy for networked installations; see [DEPLOYMENT.md](DEPLOYMENT.md).

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
- `frontend/src/observatory`: the active React Three Fiber observatory, inspectors, controls, monitoring, probability, and replay UI
- `frontend/src/store`: canonical session/step/token/layer/head/module inspection state
- `frontend/src/api`: REST and WebSocket clients for real runtime data

Live sessions report time-to-first-token, generation speed, context usage, and total generation time. Replay controls operate on recorded event indices; they do not fabricate intermediate model states.

See `DEVELOPMENT.md`, `MODEL_GUIDE.md`, `docs/API.md`, and `TROUBLESHOOTING.md` for details.

## BrainOS 3.0 observability notes

The dashboard is driven by the actual local model runtime. Before a backend run, tensor and telemetry panels remain empty or show `unavailable`; the frontend does not fall back to synthetic tokens, attention, activations, logits, or system metrics. External provider sessions are response/metadata-only by design.

Architecture and completion details are documented in [docs/BRAINOS_ARCHITECTURE.md](docs/BRAINOS_ARCHITECTURE.md) and [docs/BRAINOS_3_COMPLETION_REPORT.md](docs/BRAINOS_3_COMPLETION_REPORT.md). The full implementation specification is retained in [docs/BRAINOS_3_MASTER_PLAN.md](docs/BRAINOS_3_MASTER_PLAN.md).
