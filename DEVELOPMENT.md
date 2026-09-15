# Development

## Backend

The backend is a FastAPI application. `app.main` owns application lifecycle, model startup, the WebSocket endpoint, monitoring, and replay control.

The `ModelAdapter` interface in `backend/app/models/base.py` keeps model-specific behavior separate from inference orchestration. The initial `QwenAdapter` uses Qwen2's native `q_proj`, `k_proj`, `v_proj`, MLP projections, eager attention, hidden states, and KV cache.

Inference runs in a worker thread so the async server remains responsive. `InferenceScheduler` bounds the waiting queue and serializes the shared local model runtime by default (`BRAINOS_SCHEDULER_MAX_ACTIVE=1`). Each run has isolated user/session/request/run/connection identifiers, cancellation, timeout, event, timing, and replay state. The worker emits `Event` objects into the event bus. Current tensors remain in memory for the active session, while completed summaries, event streams, and tensor capture bundles are persisted under `backend/replay_sessions`.

BrainOS supports two persistence layers. Legacy replay summaries/events remain file-backed JSON and tensor captures remain local `.pt` bundles. `backend/app/persistence.py` provides SQLAlchemy metadata persistence for multi-user mode, using SQLite by default and PostgreSQL-compatible URLs in production. Relational rows store ownership, lifecycle metadata, and bundle references—not large tensors. Set `BRAINOS_AUTH_MODE=multi_user` to enable Argon2-backed registration/login, opaque auth sessions, protected REST/WebSocket access, and per-user session/run/replay ownership.

## Database migrations

The relational schema is managed by Alembic; application startup applies only
forward, idempotent migrations when `BRAINOS_DATABASE_AUTO_MIGRATE=true` (the
default). It no longer calls SQLAlchemy `create_all`. Deployments that manage
migrations separately can set that variable to `false` and run:

```bash
alembic upgrade head
alembic current
alembic history
```

These commands run from the repository root and use `BRAINOS_DATABASE_URL`.
The initial revision is additive and destructive downgrades are disabled. A
complete legacy BrainOS database created before Alembic is safely stamped at
the initial revision without rewriting its rows; incomplete schemas fail
explicitly so they can be reviewed rather than guessed at. SQLite is the
development default and PostgreSQL URLs use the same SQLAlchemy metadata and
migration path.

Workspace and project rows are owner-scoped. Sessions may retain a nullable
project association, while replay JSON and tensor `.pt` bundles remain
backward-compatible file artifacts referenced by relational metadata.

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
