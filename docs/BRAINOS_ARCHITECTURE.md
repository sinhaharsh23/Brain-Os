# BrainOS 3.0 Architecture

BrainOS is a localhost-only observability application for a locally loaded causal transformer. The reference runtime is `Qwen/Qwen2.5-0.5B-Instruct`; model-specific extraction is isolated behind adapters so the frontend can consume one stable event and inspection schema.

## Runtime shape

```text
React dashboard
    │ REST: metadata, inspection, replay, hardware
    │ WebSocket: one live run/event stream per connection
    ▼
FastAPI application
    ├── ModelManager ── adapter ── tokenizer + PyTorch model
    ├── InferenceEngine ── HookManager ── tensor capture/statistics
    ├── RunScheduler ── cancellation, pause/resume, queue limits
    ├── EventBus ── typed events ── connection owner
    ├── ReplayStore ── session metadata + bounded tensor bundles
    └── SystemMonitor ── process/device metrics
```

The backend binds to `127.0.0.1` by default. The Hugging Face Local provider owns its tokenizer and PyTorch model, which makes real hook-based inspection possible. Ollama is a separate runtime provider: BrainOS uses Ollama's model metadata, stream, token counts, cache counts, and timings; the regular Ollama API does not expose direct PyTorch hooks or the internal tensors listed above. Missing values remain null/unavailable. No provider is treated as the model architecture, and Ollama model metadata is never filled from the native Qwen checkpoint.

## Model lifecycle

`AppState.model_manager` is the single owner of adapter/model/tokenizer lifetime. Loading resolves the requested adapter, device (`MPS → CPU`, with optional CUDA/ROCm compatibility), and dtype, then loads on a worker thread. Reload first stops the scheduler/engine and unloads the previous adapter. Cleanup releases model references, runs garbage collection, and clears accelerator caches where supported.

`ModelMetadata` is emitted as `model.ready` and is also available from:

- `GET /api/model`
- `GET /api/model/architecture`
- `GET /api/hardware`
- `GET /api/models`

The adapter boundary currently covers Qwen2 plus compatible Llama/Mistral/Gemma-style causal models. Hook registration is architecture-aware; unsupported internals are omitted rather than filled with estimates.

## Inference and capture

`InferenceEngine.run()` creates a `SessionRecord`, registers hooks for that generation, and removes them in `finally`. A session contains the prompt, parameters, model metadata, event history, output tokens, timings, errors, and a `TensorStore` keyed by layer/step. The generation loop performs real tokenization, embedding lookup, transformer forward passes, logits/probability calculation, sampling, and KV-cache updates.

The observable event sequence is:

```text
inference.started
→ tokenization.complete
→ embeddings.complete
→ step.started
→ qkv.captured / attention.captured / residual.captured / mlp.captured
→ layer.complete
→ logits.ready / logit_lens.captured
→ token.selected
→ token.generated
→ kv_cache.captured
→ inference.completed
```

Only values produced by the actual model are sent. Candidate probabilities are computed from the real logits; entropy and sampling method are reported with logits/token-selection events. A deterministic seed is accepted by the backend for reproducible sampling.

### Capture policy

Capture is bounded by configuration to prevent long-context runs from exhausting memory:

- `BRAINOS_CAPTURE_LEVEL=summary` (default) bounds retained sequence rows to `BRAINOS_CAPTURE_LIMIT_TOKENS`.
- `selected` keeps the same bounded transport while exposing on-demand tensor inspection.
- `full` disables sequence-row truncation and should only be used deliberately.
- `BRAINOS_ATTENTION_CAPTURE=false` disables attention tensor capture and makes the UI show the unavailable state.

Session summaries include capture availability and entry counts. REST inspection endpoints return `404`/an explicit unavailable response when a tensor was not captured, rather than synthesizing a substitute:

```text
GET /api/sessions/{id}/embedding/{position}
GET /api/sessions/{id}/attention
GET /api/sessions/{id}/qkv
GET /api/sessions/{id}/mlp
GET /api/sessions/{id}/residual
GET /api/sessions/{id}/logit-lens
GET /api/sessions/{id}/kv-cache
GET /api/sessions/{id}/hidden
GET /api/sessions/{id}/logits
```

## WebSocket ownership and backpressure

`/ws` keeps a connection-local subscription and request context. A run is associated with its owning connection; inference events are not broadcast across sessions. The scheduler enforces queue and active-run limits, supports pause/resume at safe step boundaries, and emits structured timeout/failure/cancellation events. The frontend treats `inference.failed`, `inference.timeout`, `system.error`, and queue-full responses as first-class states.

## Frontend data flow

`useBrainStore` is the typed event reducer. It resets capture state at each run, stores token/layer/attention/QKV/MLP/residual/logit-lens/KV data, and keeps the session timeline. The active `App`/`ViewRouter` shell consumes this state; it does not create token IDs, embeddings, attention matrices, layer counts, timings, or system metrics on its own.

The primary shell contains:

- a real-data 3D stage driven by model metadata, token PCA coordinates, layer completion, and captured attention links;
- token flow, embedding, attention, architecture, tensor inspector, developer, and system modules;
- on-demand introspection panels for Q/K/V, MLP, residual stream, logit lens, logits, and KV cache;
- replay and session comparison backed by persisted session records;
- explicit empty/unavailable states before a run, when capture is disabled, and for external providers.

The dashboard store is deliberately initialized empty. The removed offline/demo branch never fabricates a response or tensor stream when the backend is unavailable.

## Persistence, security, and local operation

Replay metadata and tensor bundles are stored under the configured local replay directory. The API supports optional bearer/token authentication, rate limiting, strict configurable CORS, and local/multi-user auth modes already present in the repository. Default CORS and bind settings are loopback-oriented.

The normal local process pair is:

```bash
HF_HOME=./models/hf backend/.venv/bin/python backend/run.py
(cd frontend && npm run dev)
```

See [BRAINOS_3_COMPLETION_REPORT.md](BRAINOS_3_COMPLETION_REPORT.md) for the implemented scope and verification results.
