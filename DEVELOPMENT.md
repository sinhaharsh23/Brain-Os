# Development

## Backend

The backend is a FastAPI application. `app.main` owns application lifecycle, model startup, the WebSocket endpoint, monitoring, and replay control.

The `ModelAdapter` interface in `backend/app/models/base.py` keeps model-specific behavior separate from inference orchestration. The initial `QwenAdapter` uses Qwen2's native `q_proj`, `k_proj`, `v_proj`, MLP projections, eager attention, hidden states, and KV cache.

Inference runs in a worker thread so the async server remains responsive. The worker emits `Event` objects into the event bus. Tensor captures remain in memory for the current session; summaries and event streams are persisted under `backend/replay_sessions`.

## Frontend

The frontend is React + TypeScript + React Three Fiber + Zustand. `src/ws/client.ts` receives events and dispatches them into `useBrainStore`. Components render only summaries continuously. Large tensors are fetched on demand from REST endpoints.

The 3D scene is deliberately data-driven:

- Layer intensity uses captured hidden-state norms and MLP statistics.
- Attention lines use top links from actual attention matrices.
- Token nodes use actual tokenizer output and generated IDs.
- Embedding views use PCA coordinates computed from actual embedding vectors.

## Adding A Model

1. Implement `ModelAdapter`.
2. Add metadata extraction for the model configuration.
3. Expose tokenizer, embedding, forward, decode, sampling, and candidate methods.
4. Register the adapter in `models/registry.py`.
5. Add architecture-specific hooks in `instrumentation/hooks.py` or a model-local hook module.
6. Add shape and value tests before enabling the model in the explorer.

## Verification

Run the backend test suite after instrumentation changes. Then run a live WebSocket inference and inspect `/api/sessions/{id}/...` endpoints. Do not label a tensor as available unless it was captured from the real model or mathematically derived from captured values.
