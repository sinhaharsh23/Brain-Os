# BrainOS 3.0 Architecture

BrainOS is a local-first Transformer observatory. The current supported runtime is a single FastAPI process with one loaded Hugging Face model, a React/WebGL client, and file-backed replay artifacts.

```text
Browser
  │ REST + WebSocket
  ▼
FastAPI API / WebSocket endpoint
  ├── connection-scoped event routing
  ├── Qwen inference engine (serialized heavy execution)
  ├── instrumentation hooks and tensor inspection
  ├── external observation adapters (limited metadata)
  ├── system monitor
  └── replay store (JSON events + tensor capture bundles)
       │
       ├── Hugging Face model cache (project-relative by default)
       └── Apple MPS / CPU fallback
```

## Runtime boundaries

- `backend/app/models` owns model adapters, tokenization, forward passes, sampling, and metadata.
- `backend/app/inference` owns per-run tokens, events, captures, timings, cancellation, and replay summaries. The model object is shared; heavy local execution is serialized by the engine lock.
- `backend/app/events` owns the typed event bus. `backend/app/ws/manager.py` routes session-scoped events to the owning WebSocket and broadcasts only global telemetry.
- `backend/app/api` exposes readiness, hardware, model/provider metadata, replay, and on-demand tensor endpoints.
- `frontend/src/store` consumes real REST/WebSocket data. The Three.js observatory renders selective summaries: actual token positions, captured layer state, selected attention links, PCA embeddings, and top-K MLP units.

## Persistence

Replay metadata and events remain JSON files and captured tensors remain `.pt` bundles under `BRAINOS_REPLAY_DIR`. When `BRAINOS_AUTH_MODE=multi_user`, SQLAlchemy metadata persistence adds User, Workspace, Project, session, inference-run, replay-reference, model, and provider records using SQLite by default and PostgreSQL-compatible URLs. Large tensor payloads stay in the replay/object files; relational rows store references and ownership metadata. Legacy replay files remain readable and can be claimed into the authenticated ownership index.

## Accelerator policy

Device selection is `auto` by default and supports MPS, CUDA/ROCm, and CPU fallback. The physical verification target on the current development machine is Apple Silicon MPS with `Qwen/Qwen2.5-0.5B-Instruct`. CUDA and ROCm are compatibility paths and are not claimed as physically verified here.

## Observability honesty

Deep inspection is only advertised for local checkpoints whose tensors are captured by BrainOS. External providers are explicitly limited to provider-exposed response text, timing, usage, model, and errors. The browser never claims to render every neuron or every edge: the 3D view is a filtered, performant representation with on-demand full data for selected components.
