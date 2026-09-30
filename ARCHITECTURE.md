# BrainOS Architecture

BrainOS is a local-first Transformer observability app. A React and Three.js frontend talks to a FastAPI backend over REST and WebSocket. The backend can run a Hugging Face PyTorch checkpoint, an Ollama-installed GGUF model through an instrumented llama.cpp runner, or a supported hosted provider.

```text
Browser UI (React, TypeScript, Three.js)
  ├── REST: provider catalog, model/hardware metadata, tensor inspection, replay
  └── WebSocket: prompt runs, generated text, captures, timing, cancellation
          │
          ▼
FastAPI backend
  ├── Hugging Face adapter → tokenizer + PyTorch model + forward hooks
  ├── Ollama adapter → installed GGUF + instrumented llama.cpp runner
  ├── Hosted provider adapters → response stream + exposed usage/metadata
  ├── connection-scoped event routing and system monitor
  └── replay store → JSON events + captured tensor bundles
```

## Inference providers

### Hugging Face Local

BrainOS loads a supported causal model through its architecture adapter. The PyTorch execution path captures model-produced token IDs, embeddings, hidden states, attention, Q/K/V, MLP activations, residuals, logits, probabilities, and KV-cache data where enabled. Device selection supports Apple MPS, CUDA/ROCm compatibility, and CPU fallback.

### Ollama GGUF

Ollama supplies the installed model identity and GGUF file. BrainOS runs that GGUF through its small native llama.cpp evaluation-capture runner instead of relying on the standard Ollama chat API for internal tensors. It streams token IDs and generated text and captures available embeddings, attention, Q/K/V, layer outputs, MLP activations, logits, probabilities, and runtime telemetry. GGUF capture does not currently provide the PyTorch logit lens or tensor KV-cache views.

The runner is built on first use from `backend/native/gguf_runner.cpp`. The current build path expects the Ollama CLI, Homebrew `llama.cpp` and `ggml` libraries, and a C++17 compiler. `OLLAMA_GGUF_PATH` can point directly to a model file; otherwise BrainOS resolves the installed model through `ollama show`.

### Hosted providers

OpenAI, Anthropic, and Gemini adapters expose only data returned by their APIs, such as response text, stream events, timing, usage, model identity, and errors. They do not expose private transformer internals. The UI labels these runs as limited external observation and leaves unavailable tensor panels empty.

## Backend and event flow

- `backend/app/providers` describes provider capabilities and creates provider-neutral telemetry snapshots.
- `backend/app/models` owns Hugging Face model adapters and model metadata.
- `backend/app/inference` coordinates local runs, sampling, capture, cancellation, and session records. Native GGUF execution lives in `gguf_runtime.py`.
- `backend/app/events` defines typed events. The WebSocket manager sends session events only to their owning connection; global hardware and monitor events use the appropriate broadcast path.
- `backend/app/api` exposes health/readiness, provider and model catalogs, hardware/monitoring, replay, and on-demand tensor inspection.

For local inference, the event stream reports the prompt and model, input tokenization, available tensor captures, selected/generated tokens, telemetry, and completion or cancellation. Captures are actual runtime outputs; the frontend does not synthesize missing tensors or metrics.

## Frontend

`frontend/src/store/useBrainStore.ts` reduces the typed WebSocket events into session, token, layer, attention, tensor, telemetry, and replay state. React views render provider-aware model details and observability panels. The 3D stage uses selected tokens, layer progress, captured attention links, PCA embeddings, and top-K MLP units rather than drawing every tensor element.

## Persistence and security

Replay events are stored as JSON and captured tensors as `.pt` bundles under `BRAINOS_REPLAY_DIR`. Multi-user mode can persist users, workspaces, projects, sessions, runs, and model/provider metadata through SQLAlchemy with SQLite or a PostgreSQL-compatible database. Tensor payloads remain file-backed; database records store ownership and references.

Authentication mode, bearer/token protection, CORS, and rate limits are configured by backend environment settings. Provider keys are read by the backend and must stay out of the frontend bundle and Git.

## Deployment

The repository's supported deployment artifact is the root Dockerfile and Docker Compose setup, with a WebSocket-capable reverse proxy for networked use. The `brain-os` Vercel project is currently disconnected from this GitHub repository, so pushes do not create Vercel deployments. See [DEPLOYMENT.md](DEPLOYMENT.md) for the supported container and reverse-proxy setup.
