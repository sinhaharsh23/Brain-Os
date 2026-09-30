# BrainOS Architecture

BrainOS is a local-first model observability app. The browser presents an interactive React/Three.js dashboard, while a FastAPI process owns inference, capture, provider adapters, replay, and hardware telemetry. The browser uses REST for catalogs and inspection and WebSocket for live generation events.

## Runtime shape

```text
React + TypeScript + Three.js browser app
  ├── REST: health, provider/model catalog, replay, tensor inspection
  └── WebSocket: prompts, streamed events/text, telemetry, cancel/pause/resume
          │
          ▼
FastAPI backend
  ├── Hugging Face adapter → tokenizer + PyTorch model + forward hooks
  ├── Ollama model lookup → installed GGUF + native llama.cpp capture runner
  ├── OpenAI / Anthropic / Gemini adapters → API response and exposed metadata
  ├── provider-neutral telemetry + connection-scoped event routing
  ├── system monitor and model/hardware catalog
  └── replay store → JSON events + tensor capture bundles
```

## Provider runtimes

### Hugging Face Local

The local model manager resolves the selected supported checkpoint and architecture adapter, then owns tokenizer and model lifetime. Inference uses PyTorch. Architecture-aware hooks capture real embeddings, hidden states, attention, Q/K/V projections, MLP activations, residuals, logits, probabilities, and KV-cache tensors when the corresponding capture path is enabled. Token IDs and model metadata come from the loaded checkpoint and tokenizer.

The reference checkpoint is `Qwen/Qwen2.5-0.5B-Instruct`. Device selection can use Apple MPS or CPU and supports CUDA/ROCm compatibility paths. Actual device availability depends on the host.

### Ollama GGUF through llama.cpp

Ollama provides the local model inventory and installed GGUF model. For deep inspection, BrainOS resolves the GGUF file from `ollama show <model> --modelfile`, or uses `OLLAMA_GGUF_PATH` / `BRAINOS_OLLAMA_GGUF_PATH` when set. BrainOS launches its native runner built from `backend/native/gguf_runner.cpp`; the standard Ollama chat API alone does not provide the internal tensors used by these views.

The current native build path looks up Homebrew `llama.cpp` and `ggml` and compiles a C++17 runner on first use. The runner emits model metadata and captured graph signals for token IDs, embeddings, attention, Q/K/V, layer outputs, MLP activations, logits, probabilities, and generated tokens. Provider telemetry identifies the Ollama model and the sources of the model, token, timing, and tensor data. The current GGUF capability set does not include the PyTorch logit lens or tensor KV-cache view.

`OLLAMA_URL` (or `BRAINOS_OLLAMA_URL`) defaults to `http://127.0.0.1:11434`. `OLLAMA_MODELS` (or `BRAINOS_OLLAMA_MODELS`) selects the comma-separated installed models shown in the provider catalog. The Ollama CLI, Homebrew libraries, and a C++17 compiler are required for the deep GGUF path.

### Hosted providers

OpenAI, Anthropic, and Gemini use their APIs and expose only response text, stream events, model identity, timing, usage, stop metadata, and errors when returned by the provider. Their adapters do not claim to expose private layers, attention, Q/K/V, MLP activations, hidden states, or an internal KV cache. The frontend marks these sessions as limited external observation and leaves unavailable tensor views empty.

## Inference and event flow

Each local run is represented by a session record containing the prompt, selected model, sampling parameters, events, output, timings, status, and any captured tensors. The event stream reports the signals produced by the selected runtime. Hugging Face events are backed by PyTorch execution and hooks; Ollama events are backed by llama.cpp graph capture; hosted events are backed by the provider API response. A missing signal stays unavailable rather than being estimated from output text.

The WebSocket manager associates inference sessions with the requesting connection so one user's run is not broadcast to another. Provider and session events share the frontend event reducer, while global hardware/monitor data follows the global telemetry path. Local generation supports cancellation; the native GGUF subprocess is terminated through the runtime cancellation event. The scheduler controls native Hugging Face work and its safe pause/resume boundaries.

## Backend modules and API

- `backend/app/providers` describes provider/model capabilities and builds provider-neutral telemetry.
- `backend/app/models` owns Hugging Face adapters, tokenization, forward passes, sampling, and local model metadata.
- `backend/app/inference` owns generation orchestration, scheduling, capture, cancellation, and session summaries. `gguf_runtime.py` launches the native capture runner for Ollama.
- `backend/app/events` defines typed events, and `backend/app/ws/manager.py` scopes event delivery to session owners.
- `backend/app/api` exposes health/readiness, model/provider information, hardware/monitoring, replay, and on-demand tensor inspection.

Frequently used endpoints include:

```text
GET  /api/health
GET  /api/ready
GET  /api/models
GET  /api/providers
GET  /api/model
GET  /api/model/architecture
GET  /api/hardware
GET  /api/monitoring/history
GET  /api/sessions/{id}/attention
GET  /api/sessions/{id}/qkv
GET  /api/sessions/{id}/mlp
GET  /api/sessions/{id}/hidden
GET  /api/sessions/{id}/logits
WS   /ws
```

Tensor inspection endpoints are on-demand. They return only data captured for that session; unavailable captures are reported explicitly.

## Frontend data flow

`frontend/src/store/useBrainStore.ts` reduces typed backend events into provider state, session status, token and layer data, inspection results, telemetry, and replay state. The views include the 3D observatory, token flow, embeddings, attention, architecture, developer tools, model selection, and telemetry details. The 3D stage uses selected model metadata and captured signals such as token PCA positions, layer progress, attention links, and top-K MLP units. It is a filtered visualization, not a rendering of every tensor element.

Before a local session has produced captures, panels show empty or unavailable states. External provider sessions do not fabricate local tensor data. Cancellation, failure, timeout, and unavailable-provider states are displayed as runtime states rather than synthetic output.

## Replay, persistence, and security

Replay events are JSON files and captured tensors are `.pt` bundles under `BRAINOS_REPLAY_DIR`. Multi-user mode stores user, workspace, project, session, inference-run, replay-reference, model, and provider metadata with SQLAlchemy. SQLite is the development default; PostgreSQL-compatible URLs are supported. Large tensor data stays in files while relational records hold ownership and references.

Authentication, token protection, CORS, and rate limits are configured by backend environment settings. Provider keys remain server-side; do not put them in the frontend bundle, Docker image, or repository.

## Running and deployment

The local setup uses the project Python environment and frontend dev server as described in the root README. Docker Compose is the repository's supported deployment artifact; a networked installation also needs HTTPS/TLS and a reverse proxy that forwards HTTP and WebSocket traffic. The Vercel `brain-os` project is currently disconnected from this GitHub repository. See [DEPLOYMENT.md](../DEPLOYMENT.md) for container, proxy, secret, and operations guidance.
