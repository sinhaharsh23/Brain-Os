# BrainOS 3.0 Final Completion Report

## Outcome

BrainOS 3.0 is complete as a local transformer observatory over the existing
Observatory UI. This final pass hardened real-data handling, synchronized
session inspection state, checked generation lifecycle behavior, validated the
current Apple Silicon runtime, and preserved the existing visual identity.

No major feature or architecture rewrite was added. `frontend/src/App.tsx`
mounts `Observatory` as the product surface; the old UI is not mounted and its
legacy modules are not included in the production bundle.

## Exact environment tested

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Architecture | `Qwen2ForCausalLM` |
| Device | Apple Silicon MPS (`mps`) |
| Host report | Apple M5, 16.0 GB unified memory |
| Model shape | 24 layers, hidden 896, 14 query heads, 2 KV heads, head dimension 64, MLP intermediate 4864, vocabulary 151,936 |
| Python | 3.9.6 |
| Node | v26.8.1 |
| PyTorch | 2.8.0 |
| Transformers | 4.57.6 |
| CUDA | unavailable on this machine; not required |
| MPS | detected and used for live inference |

The hardware endpoint reports `backend: MPS`, `device: mps`,
`cuda_available: false`, and `vram_total_gb: null`. Apple unified memory is
reported as `MPS allocated memory`, not as fabricated VRAM.

## Real-data audit

The active runtime source was searched for random/demo/mock/placeholder data and
hardcoded transformer values. The active observatory and backend paths contain
no generated transformer internals. Every unavailable signal is represented by
an explicit unavailable state.

The repository still contains dormant legacy `frontend/src/components/dashboard3`
source files with old UI-only demo code. They are not imported by `App.tsx`,
are not in the built bundle, and cannot supply data to the active observatory.
This distinction is recorded rather than silently deleting unrelated user
working-tree files.

The following surfaces were verified against the real local adapter/session:

| Surface | Evidence |
| --- | --- |
| Token IDs and special tokens | `tokenization.complete` and tokenizer vocabulary; control tokens remain in the explorer |
| Embeddings | Adapter embedding layer and PCA payload |
| Position/RoPE | Loaded model metadata: Rotary Position Embedding, `rope_theta: 1000000` |
| Hidden states | Per-layer forward-hook/model outputs, shape `[1 × 896]` for a selected token |
| Attention | Actual model attention weights and bounded query rows |
| Q/K/V | Actual projection hooks; Q/K/V shape `[1 × 64]` for the selected head |
| MLP activations | Actual Qwen gated intermediate dimensions, shape `[1 × 4864]` |
| Residual stream | Actual layer-input, attention-output, MLP-output, and layer-output hooks |
| Logits | Actual LM-head vocabulary scores |
| Probabilities | Exact temperature/top-k/top-p distribution used by sampling |
| Logit Lens | Actual intermediate hidden representation through final norm and LM head |
| KV cache | Actual per-layer key/value shapes, lengths, and byte estimates |
| Architecture/model metadata | Adapter introspection of the loaded checkpoint |
| Performance | Recorded tokenization, prefill, TTFT, per-token, total, and throughput timings |
| System metrics | Live CPU, RAM, process RAM, MPS allocation, and scheduler values |

## Canonical synchronization

The frontend now retains per-step histories for layers, attention links, QKV
summaries, MLP summaries, candidates, and logits. The visible inspector derives
from one canonical context:

`Session → Generation Step → Token → Layer → Head → Module`

Stale events from a previous session are ignored once a new session is active.
Selecting a generated token resolves the matching decode step and the selected
layer/head/module are passed to all inspector requests. Attention target rows do
not silently replace the canonical query token.

The live UI sweep verified compatible state in the 3D scene, Token Explorer,
Attention, QKV, MLP, Residual, Hidden State, Logits, Probability, Logit Lens,
KV Cache, and Tensor Inspector views. Layer and head selection remained tied to
the selected step.

## Pipeline and token handling

The prompt `The capital of France is` was run through the local model. The
observed event chain covered chat-template serialization, tokenization, token
IDs, embeddings, generation steps, layer summaries, attention, Q/K/V, gated
MLP, residual stream, KV cache, Logit Lens, logits, sampler probabilities,
selected tokens, decoding, and completion.

The clean-start smoke run produced ordered steps `[0, 1, 2, 3]` and the
human-readable response `The capital of France`, with no `<|im_...|>` control
token in the displayed response. A longer UI run produced the expected
human-readable continuation beginning `The capital of France is Paris.`

Special/control tokens remain in the introspection sequence, including chat
template markers such as `<|im_start|>` and `<|im_end|>`. Normal output uses
the adapter's special-token-skipping decode path. EOS and chat-template control
tokens are therefore inspectable without leaking into ordinary response text.

Additional final demo prompts were run through the UI:

- `Explain why the sky appears blue.`
- `Write three reasons transformers use attention.`

Both completed with real streamed output and captured token/timing state.

## Streaming, cancellation, and recovery

Three consecutive direct WebSocket generations produced ordered steps
`[0, 1, 2]` with no duplicate steps and one terminal completion each. A longer
generation was stopped after tokens began arriving; it produced exactly one
`inference.cancelled` terminal event, persisted a `cancelled` session with
captured events, and a new generation succeeded immediately afterward.

The following recovery checks passed:

- Invalid model path: HTTP 400 with an explicit unsupported-model error.
- Invalid `top_p`: WebSocket error with validation text, followed by a successful
  generation on the same connection.
- Backend/model lifecycle failures: scheduler and UI failure states are
  explicit; hooks are removed on complete, cancel, and failure.
- WebSocket ownership/isolation: browser and backend tests cover independent
  connections and stale-event isolation.

## Model switching and capabilities

The verified local checkpoint is the Qwen 0.5B adapter above. Model descriptors
for Qwen 1.5B/3B/7B, TinyLlama, Mistral, and Gemma are exposed with truthful
`implemented-unverified` status when a safe real-checkpoint run was not
performed. The Qwen 3B checkpoint is present in the local cache, but was not
loaded because the current memory headroom does not make that a safe final
verification run.

Model load/unload is owned by `ModelManager`; the old scheduler, adapter,
hooks, and tensors are released before a replacement runtime becomes ready.
Unsupported capabilities are shown as unavailable instead of being inferred
from a different model family.

## Apple Silicon and CPU fallback

MPS was detected and used for the clean-start inference run. A separate CPU
adapter load and forward pass also completed successfully with logits shape
`[1, 34, 151936]` and `torch.float32`. Device resolution reported `auto → mps`
and explicit `cpu → cpu`.

There is no CUDA dependency. MPS-specific memory is displayed only when the
runtime exposes it; GPU utilization and VRAM totals remain unavailable on this
unified-memory host rather than being fabricated.

## Memory and Three.js audit

Generation cleanup now removes all per-generation hooks, releases the hook's
session reference, and performs a guarded accelerator cache release. Replay
event arrays are persisted to disk and loaded on demand instead of being kept
inside every in-memory session. Restored tensor captures have a bounded
three-session working set.

The soak run used repeated short generations, replay probes, inspector access,
and a later post-cancellation generation. Process RSS fluctuated during MPS
warm-up and was reclaimed later; it did not increase monotonically through the
repeated run set. MPS allocated memory remained stable at approximately 1.84 GB
in the monitor samples. This is allocator behavior, not a claim that all host
memory is fixed at one number.

`TransformerScene3D.tsx` was audited without replacing it:

- one React Three Fiber animation loop;
- bounded rendering of at most 40 layers, 18 token labels, and 20 attention links;
- no per-frame activation generation or random values;
- selected model/layer/attention state controls the visuals;
- `resetKey` resets the camera through the existing Canvas lifecycle;
- React Three Fiber owns geometry/material cleanup on unmount; no duplicate
  animation loop or thousands of DOM labels were introduced.

## Long-input behavior

A 9,679-character prompt was run. The tokenizer respected the configured
1,024-token prompt bound. Attention capture retained a bounded selected query
row: the returned attention payload had 1,024 weights and
`is_full_matrix: false` for a 1,025-token sequence. Earlier prompt positions
outside the retained row correctly returned unavailable rather than causing a
large full matrix to be rendered.

## Sessions, replay, and compare

Session summaries preserve prompt, output tokens, output text, model metadata,
generation parameters, timing data, event timelines, and tensor-capture files.
The persisted replay store was exercised after backend restart.

Replay protocol validation covered load, pause, speed `0.25x`, seek, single-step
advance, and stop. The Observatory timeline exposes First, Previous, Play,
Pause, Next, Last, scrubber, and `0.25x / 0.5x / 1x / 2x` controls.

Compare Mode was run with:

- Run A: temperature `0.2`
- Run B: temperature `1.0`

The comparison returned real configuration, output, token, probability,
entropy, TTFT, prefill, average/current token latency, total time, throughput,
KV-cache peak, attention compatibility/summary, and Logit-Lens compatibility
data. The observed comparison included average entropy `0.000004` versus
`0.163821`, and both sessions reported a real KV-cache peak of `0.8438 MB`.
Process memory is explicitly marked unavailable for replay comparison because
it is not sampled into individual session records.

## Mathematical verification

The loaded Qwen architecture was checked directly on CPU with a final-layer
forward hook:

- projecting the pre-final-normalization final block output through the actual
  final norm and LM head matched the model logits with maximum absolute
  difference `2.57e-05`;
- applying the final norm a second time produced a maximum difference of
  `5.10`, confirming the final-layer overwrite bug was fixed;
- raw-logit and final-layer Logit-Lens top-1 token ID both matched `785`.

The live QKV endpoint returned query head `7 → 7` and grouped K/V head
`7 → 1`, consistent with 14 query heads and 2 KV heads. Live shapes were
`[1, 64]` for Q/K/V, `[1, 4864]` for the gated MLP intermediate, and
`[1, 896]` for the hidden state. KV-cache entries reported per-layer shapes
`[1, 2, sequence_length, 64]` across 24 layers.

Attention rows are treated as routing weights, not causal explanations. MLP
entries are actual intermediate dimensions. Residual labels are computed from
captured layer input, attention output, MLP output, and layer output tensors.

## Normal/developer modes and educational text

Normal Mode presents the human-to-model pipeline, model map, tokens, attention,
basic embeddings, probabilities, timeline, and plain-language explanations.
Developer Mode exposes event/tensor-oriented details including paths, shapes,
dtype/device/statistics, QKV, hidden state, MLP, residuals, logits, cache,
performance, and events. Large tensors remain on-demand to keep the interface
responsive.

The active UI consistently explains:

`text → token IDs → vectors → transformer computation → logits → selected token → decoded output`

Labels use internal representation, activation, attention, residual stream,
hidden state, layer prediction, and model computation terminology.

## Experiments and unavailable capabilities

The current Qwen adapter reports `activationPatching: false`. Attribution and
intervention controls are disabled and visibly say `Unavailable for this
model`. No fake intervention result was presented, and no experiment hook was
run or left installed. External provider sessions likewise expose response and
metadata only; private provider internals are unavailable by design.

## Final test suite

The final results after the clean-start hardening pass are:

```text
PYTHONPATH=backend backend/.venv/bin/python -m pytest -q
95 passed, 3 skipped, 1 warning

cd frontend && npm run build
TypeScript build and Vite production build passed

cd frontend && npm run test:browser
6 passed

PYTHONPATH=backend backend/.venv/bin/python -m compileall -q backend/app
passed
```

The one warning is the environment's urllib3/LibreSSL compatibility warning
from Python 3.9; it did not fail a test. No unrelated tooling was added.

## Exact clean-start commands

The final clean run used the repository startup command:

```bash
cd /Users/harshsinha/Documents/BrainOS
./scripts/start-brainos.sh
```

It starts:

- Observatory UI: `http://127.0.0.1:8765` (integrated backend-served UI)
- Vite development UI: `http://localhost:5173`
- REST API: `http://127.0.0.1:8765/api`
- WebSocket: `ws://127.0.0.1:8765/ws`

The split development commands are also supported:

```bash
./scripts/start-backend.sh
./scripts/start-frontend.sh
```

## Known limitations

- Default capture is bounded summary capture with a 256-position tensor bound;
  full capture is intentionally opt-in and can be expensive.
- Long-context attention may be unavailable for an early position when only a
  selected query row was retained. The UI reports this instead of rendering a
  misleading full matrix.
- Only Qwen 0.5B was fully verified on this machine. Other descriptors remain
  explicitly unverified until a safe real-checkpoint run is available.
- Activation patching, causal attribution, and private external-provider
  internals are unavailable for the current adapter.
- MPS GPU utilization and VRAM totals are unavailable on this unified-memory
  host. MPS allocation and host RAM remain visible from real sources.
- Process-memory samples are live system telemetry; per-session process-memory
  comparison is intentionally unavailable.
- The Python runtime emits the existing LibreSSL warning noted above.

These limitations are surfaced as unavailable states and do not use generated
transformer internals.

## LOCAL + CLOUD AI ENGINE

The architecture correction preserves the existing Observatory, local
inference engine, tensor store, replay APIs, and 3D scene. It adds a unified
provider/model boundary in `backend/app/providers`:

- `AIProvider`, `ModelDescriptor`, `ChatMessage`, and capability metadata are
  provider-neutral.
- `ProviderRegistry` exposes `qwen-local`, `openai`, `anthropic`, and `gemini`
  with separate local/cloud availability and truthful capability flags.
- `AIProviderRouter` exposes `LocalModelProvider`, `OpenAIProvider`,
  `AnthropicProvider`, and `GeminiProvider`. Local execution remains owned by
  the existing `InferenceEngine` and scheduler so the completed introspection
  path is not duplicated or replaced.
- Cloud stream events normalize to `generation.started`, `provider.started`,
  `response.created`, `response.text.delta`, `provider.usage`,
  `response.text.completed`, `generation.completed`, `generation.cancelled`,
  and `generation.error`. Stream deltas are labeled as deltas, not fabricated
  tokenizer tokens.
- The OpenAI adapter uses the official Responses API when the official SDK is
  installed, with a server-side compatibility path for source checkouts that
  have not installed optional SDK dependencies. The implementation follows the
  documented `responses.create(..., stream=True)` event model.
- Anthropic Messages streaming and Google GenAI `generate_content_stream` are
  supported when their official SDKs are installed; raw SSE compatibility is
  retained for the existing offline/mock test path.

### Validation status

| Area | Status |
| --- | --- |
| Local models | Verified offline against the existing cached `Qwen/Qwen2.5-0.5B-Instruct` snapshot at `models/hf/hub/.../snapshots/7ae557604adf67be50417f59c2c2f167def9a775`; 24 layers loaded with no Hub request. |
| OpenAI | Not live-tested; `OPENAI_API_KEY` is missing. Backend-only configuration and mocked normalized streaming pass. |
| Anthropic | Not live-tested; `ANTHROPIC_API_KEY` is missing. Mock/compatibility path remains covered. |
| Gemini | Not live-tested; `GEMINI_API_KEY` and `GOOGLE_API_KEY` are missing. Mock/compatibility path remains covered. |
| Model switching | UI switcher groups local full-trace models and configured cloud models; cloud selection clears stale local transient inspectors without unloading the local runtime automatically. |
| Local model resolution | `LocalModelResolver` checks `LOCAL_MODEL_PATH`, `models/<model-name>`, and HF cache snapshots without network access, validates config/tokenizer/weights, and only downloads when explicitly allowed. `LOCAL_MODEL_OFFLINE=true` and `LOCAL_MODEL_ALLOW_DOWNLOAD=false` are the safe defaults. |
| Local loading | The resolved filesystem path is passed to both Transformers loaders with `local_files_only=True`; `HF_HUB_OFFLINE=1` was verified. Missing and incomplete checkpoints return `LOCAL_MODEL_NOT_INSTALLED` / `LOCAL_MODEL_INCOMPLETE`, not a DNS exception. |
| Local introspection | Offline generation verified real token IDs, embeddings, 24 actual layers, RoPE/model metadata, attention, Q/K/V, GQA-compatible projections, MLP, residuals, hidden states, logits, probabilities, Logit Lens, KV cache, and decoded output. |
| Layer events | Real forward pre/post hooks emitted 192 `layer.started` and 192 `layer.completed` events for 8 generated tokens × 24 layers, alongside attention/MLP lifecycle events. The existing `layer.complete` summary event remains intact. |
| Layer visualization | The existing 3D scene consumes the canonical active layer/module state; no random pulse loop was introduced. |
| Token → embedding | Existing real tokenizer and embedding events remain the source of the token explorer and PCA view. |
| Chat | Prompt history is provider-neutral; assistant messages retain provider/model/mode metadata. Local conversation messages are passed to the model tokenizer chat template when supported. |
| Streaming | Local token events and cloud text-delta events are handled separately and rendered progressively in the chat surface. |
| Secrets | `.env` remains ignored; `.env.example` contains placeholders only; frontend code receives provider status and metadata, never API keys. |

### Tests for the architecture correction

`backend/tests/test_provider_architecture.py` covers registry contents,
capability separation, Google/Gemini alias normalization, normalized OpenAI
Responses-style streaming, and missing-key error handling. New resolver tests
cover explicit paths, project-local checkpoints, HF cache snapshots, incomplete
checkpoints, and no-download failure behavior. The offline backend suite passed:

```text
109 passed, 3 skipped
```

The frontend TypeScript/Vite build passed. `scripts/download-local-model.sh`
detected the existing cached checkpoint and printed `Model already installed`
without downloading. `scripts/start-brainos.sh` now reports local checkpoint
source/path and `Network Required: NO` before starting the existing localhost
services. The cloud providers were not live-tested because their API keys are
not configured in this workspace; their mocked/compatibility tests remain
covered independently.

## UI / INTROSPECTION CORRECTION PASS

This pass preserved the Observatory shell and connected the requested
presentation fixes to existing runtime state rather than adding simulated
signals.

### 1. Provider configuration

- Root cause: the API catalog used a static import-time descriptor and the
  settings surface collapsed missing credentials and untested connections into
  Not configured.
- Files changed: backend/app/config.py, backend/app/providers/base.py,
  backend/app/providers/registry.py, backend/app/api/routes.py,
  frontend/src/api/client.ts, frontend/src/types.ts,
  frontend/src/observatory/Observatory.tsx, .env.example.
- Fix: provider descriptors refresh from backend settings, root .env aliases
  are recognized, explicit metadata-only Test Connection actions return
  distinct missing/auth/network/rate-limit/unavailable states, and the
  frontend receives booleans/status strings only.
- Test/result: provider status unit tests and browser provider-status check;
  full backend suite 109 passed, 3 skipped. No API secret appears in the
  frontend or provider payload.

### 2. Transformer State Map and focus

- Root cause: mode selection was local component state and the scene mostly
  kept the same representation across buttons; focus only changed an
  inspector selection.
- Files changed: frontend/src/store/useBrainStore.ts,
  frontend/src/observatory/Observatory.tsx,
  frontend/src/observatory/TransformerScene3D.tsx,
  frontend/src/observatory/observatory.css.
- Fix: canonical transformerViewMode drives architecture, flow, attention,
  activation, residual, cache, and performance views. Layer count comes from
  loaded metadata, camera focus follows selectedLayer, and Reset camera / Fit
  model are available.
- Test/result: browser clicks verified Architecture, Flow, Residual flow,
  Cache, Focus layer, Reset camera, and Fit model; frontend build passed.

### 3. Embedding inspector

- Root cause: the previous inspector placed the PCA map and high-dimensional
  summary into one cramped surface and had no vector range control.
- Files changed: frontend/src/observatory/Observatory.tsx and
  frontend/src/observatory/observatory.css.
- Fix: added summary metrics, Overview, Vector with 64-dimension ranges, Top
  Dimensions, Distribution, 2D Map, and 3D Map views. Heatmap hover titles,
  top dimensions, histogram values, and PCA positions use captured vectors.
- Test/result: browser generation check opened Embedding space and Top
  dimensions; local embedding endpoint returned real vector/stats data.

### 4. Position/RoPE and chat template

- Root cause: the position panel assumed RoPE and was not provider-aware.
- Files changed: backend/app/models/qwen.py and
  frontend/src/observatory/Observatory.tsx.
- Fix: adapters report position_encoding_type; Qwen exposes RoPE, rotary
  dimension, head dimension, theta and an educational Q/K -> RoPE diagram.
  Cloud mode reports positional internals unavailable. Raw chat template
  output stays in the dedicated inspector and the normal prompt starts empty.
- Test/result: browser inspector check verified RoPE and normal clean chat;
  local metadata reports Qwen RoPE.

### 5. Layers and lighting

- Root cause: layer cards lacked lifecycle status/detail and completed hook
  events did not settle a visible state with measured duration.
- Files changed: backend/app/inference/engine.py,
  frontend/src/store/useBrainStore.ts,
  frontend/src/observatory/Observatory.tsx,
  frontend/src/observatory/observatory.css.
- Fix: real layer/module durations attach to events, states transition through
  waiting/processing-attention/processing-mlp/completed, and the layer
  detail diagram uses actual norms, residual summaries and Q capture.
  Replay controls distinguish slowed/step playback from live timestamps.
- Test/result: exact offline Qwen run produced 192 starts and 192 completions
  for 8 x 24; browser state-map and inspector checks passed.

### 6. Attention

- Root cause: only a selected-row bar chart was exposed.
- Files changed: backend/app/api/routes.py, frontend/src/api/client.ts,
  frontend/src/types.ts, frontend/src/observatory/Observatory.tsx,
  frontend/src/observatory/observatory.css.
- Fix: bounded captured matrix/average/head summaries are exposed from the
  attention store. The UI adds selected-token context, head browser,
  average-head mode, heatmap hover values, causal-mask indication, and the
  Q/K/V -> score -> mask -> softmax -> output diagram.
- Test/result: local capture and browser synchronized-inspector check passed;
  backend suite passed. Attention remains actual data and is not presented as
  a complete reasoning explanation.

### 7. Q/K/V and GQA

- Root cause: vector bars were technically correct but lacked the conceptual
  path and grouped-query mapping context.
- Files changed: backend/app/api/routes.py, frontend/src/types.ts,
  frontend/src/observatory/Observatory.tsx, frontend/src/observatory/observatory.css.
- Fix: added presentation-friendly Q/K/V diagrams, selected
  token/layer/query-head/KV-head details, GQA mapping, real Q dot K match
  rows, scaled scores, attention probabilities, and value-contribution
  summaries when tensors are captured.
- Test/result: offline local signal audit confirmed QKV and attention stores;
  browser check opened Q/K/V after generation.

### 8. Attribution

- Root cause: the old panel was a generic unsupported placeholder and did
  not explain the required computation or current adapter limitation.
- Files changed: frontend/src/observatory/Observatory.tsx.
- Fix: replaced it with a direct-logit-attribution contract panel stating the
  selected output token/step, required residual/final-norm/LM-head signals,
  and actionable local-capture limitation. Cloud mode reports private
  attribution unavailable.
- Test/result: browser attribution check and capability inspection passed;
  no fabricated chart was added. The current Qwen adapter reports
  attribution unavailable truthfully.

### 9. Experiment Lab

- Root cause: disabled buttons implied experiments existed although the
  current adapter reported no intervention hook.
- Files changed: frontend/src/observatory/Observatory.tsx.
- Fix: the lab explains baseline -> temporary hook -> run -> cleanup and
  presents Head Ablation, MLP Ablation, and Activation Patching as Coming
  later without broken input forms until a real hook is implemented.
- Test/result: browser experiment check verified the explanatory surface and
  absence of fake ablation controls; normal generation remains unaffected.

### Correction-pass verification

The full backend suite passed with 109 passed, 3 skipped. The frontend
TypeScript/Vite build passed. The existing browser regression suite passed
with 7 passed; two added correction checks for synchronized inspectors and
provider status also passed. The localhost start script reported the
installed Qwen checkpoint, Network Required: NO, and loaded BrainOS offline
with HF_HUB_OFFLINE=1. OpenAI, Gemini, and Anthropic were not live-tested
because no valid credentials are configured in this workspace.

## Reference UI Replication Pass (historical, rolled back)

- Reference image: `references /ChatGPT Image Sep 19, 2026, 05_23_21 PM.png`.
  The requested `docs/ui-reference/brainos-target-ui.png` was not present;
  this supplied image was used as the visual source of truth.
- Target viewport: 1672 × 941.
- Main files changed: `frontend/src/observatory/Observatory.tsx`,
  `frontend/src/observatory/TransformerScene3D.tsx`,
  `frontend/src/observatory/observatory.css`, `frontend/src/App.tsx`,
  `frontend/src/store/useBrainStore.ts`, `frontend/src/ws/client.ts`,
  `backend/app/providers/registry.py`, `backend/app/ws/manager.py`,
  `backend/app/config.py`, and `frontend/tests/browser.spec.ts`.
- Historical result: this pass temporarily introduced the compact reference
  shell, reference-style navigation, and Chat/Token Stream dock. It is no
  longer the current BrainOS interface.
- Functional fixes included in this pass: provider catalog retry during
  startup, provider-registry responsiveness while local generation runs,
  truthful local/cloud status, canonical mode/inspector controls, and
  connection-scoped websocket routing for multi-session isolation. The local
  Qwen model continues to load from the offline checkpoint without Hugging
  Face network access.
- Visual comparison: captured
  `docs/ui-reference/brainos-before-replication.png`,
  `docs/ui-reference/actual-brainos-ui.png`, and
  `docs/ui-reference/brainos-ui-diff.png` using
  `scripts/visual_diff.py`. The final measured difference was 36.03% at the
  configured threshold. This is not literal pixel equality: the reference
  includes an open model menu and different live prompt/trace content, while
  the application screenshot contains real runtime data and WebGL rendering.
  The report therefore does not claim pixel-perfect parity.
- Verification: `npm run build` passed; backend tests passed with
  `109 passed, 3 skipped`; the final integrated browser suite passed with
  `9 passed` in 48.4s. Browser coverage exercised the real local prompt,
  token/inspector surfaces, transformer modes, focus controls, provider
  status without secrets, multi-session isolation, and the narrow viewport.
- Cloud status: OpenAI, Gemini, and Anthropic were not live-called because no
  valid credentials were configured. Their backend registry entries remain
  visible as independently missing/configurable providers; no key is sent to
  frontend code.

## Reference UI Replication Rollback

- Reason: the latest screenshot-replication shell was not the desired BrainOS
  interface. The requested target image was therefore treated as historical
  input only and the pre-replication Observatory was restored.
- History audit: `git status`, the last 20 commits, and the last 20 reflog
  entries showed that the replication work was uncommitted in the existing
  dirty worktree; there was no isolated commit safe to revert wholesale.
  Unrelated provider, offline-model, websocket, and introspection work was
  preserved.
- UI restored: `frontend/src/observatory/Observatory.tsx` now uses the earlier
  Live trace / Models / Sessions / Settings header, Trace Explorer sidebar,
  inference pipeline, 3D state map, generation timeline, prompt console, and
  full inspector-tab layout. `observatory.css` no longer contains the
  1672×941 reference-shell override.
- Reference-only code removed from the rendered application: target-width
  layout rules, reference badge and rails, Chat/Token Stream dock, target
  navigation items, inspector quick-link layer, and reference-specific scene
  camera/layout additions. The real 3D scene, layer overlay, playback, mode
  controls, focus/reset/fit camera controls, and inspector data remain.
- Functional fixes preserved: secure provider configuration and model
  switching, offline Qwen resolution/loading, real token and transformer
  capture, cloud observability limits, layer events, sessions/replay/compare,
  websocket session isolation, and provider startup retry behavior.
- Verification: `HF_HUB_OFFLINE=1 ./scripts/start-brainos.sh` resolved the
  installed Qwen snapshot with `Network Required: NO`; frontend build passed;
  backend tests passed with 109 passed and 3 skipped; the restored browser
  regression suite passed with 9 passed.
- Screenshot: `docs/ui-reference/brainos-rollback.png` was captured at
  1672×941 and inspected. It shows the earlier Observatory composition rather
  than the unwanted reference-replication shell. No claim of pixel identity is
  made; runtime model state and WebGL rendering remain real.

## Previous Dashboard Restoration

- Git audit: `bb5c266` is the current repository tip. The Reference UI
  Replication Pass was not an isolated commit; its Observatory changes were
  present in the existing dirty worktree. The nearest committed Observatory
  snapshot, `b29ba13`, predates the later correction pass and was not restored
  wholesale because doing so would regress provider selection, camera controls,
  canonical visualizer modes, and the richer inspectors.
- Dashboard restored: the pre-reference BrainOS shell is active again in
  `frontend/src/observatory/Observatory.tsx`,
  `frontend/src/observatory/TransformerScene3D.tsx`, and
  `frontend/src/observatory/observatory.css`. The earlier header, Trace
  Explorer, pipeline, 3D state map, timeline, prompt console, and inspector
  arrangement are restored. Reference-only 1672×941 layout overrides,
  screenshot dock/rails, target navigation, and decorative reference markup
  are no longer rendered.
- Functional fixes retained: provider hydration and status handling, local
  Qwen offline resolution and `local_files_only=True`, cloud-provider
  architecture, websocket isolation, transformer event capture, mode and
  camera controls, readable embedding/position/attention/QKV inspectors,
  attribution/experiment capability states, sessions, replay, and compare.
- Confirmation screenshot: `docs/ui-reference/restored-previous-brainos-ui.png`
  at 1672×941. It was inspected as a confirmation artifact only and is not a
  production pixel-match constraint.
- Verification: frontend build passed; backend tests passed with `109 passed,
  3 skipped`; browser regression tests passed with `9 passed`; offline startup
  under `HF_HUB_OFFLINE=1` resolved the installed local Qwen checkpoint,
  reported `Network Required: NO`, and reached the loaded/ready state. No live
  cloud call was claimed because no valid provider credentials were available.

## Original UI Restoration + 800 Token Generation Limit

- Original UI source: verified tag `brainos-v3.0-rc1` at commit `a5dcfe7`
  (`release: BrainOS 3.0 local release candidate`). The tracked first-dashboard
  frontend was restored from that commit without resetting the repository.
- Restored UI files: `frontend/src/App.tsx`, `frontend/src/main.tsx`,
  `frontend/src/styles.css`, the original `frontend/src/components/` shell,
  dashboard, prompt, bottom-panel, and inspector files,
  `frontend/src/brain3d/`, `frontend/src/views/`, and the original mock-hook
  compatibility file. Current API client, store, WebSocket client, provider
  backend, offline resolver, and model/introspection code were retained.
- Active appearance: the original BRAINOS / Autonomous Intelligence OS
  dashboard is active again. The reference-image Observatory shell is not
  rendered by the production entry point. The screenshot fixture
  `docs/ui-reference/original-brainos-ui-restored.png` was captured at
  1672×941 for documentation only.
- Output limit: the audited request ceilings were unified at `800`.
  `backend/app/config.py` and `backend/app/security.py` now accept
  `max_new_tokens` from 1 through 800; `backend/app/inference/engine.py`
  clamps local output to the model's remaining context capacity. The original
  prompt control now shows `Max Output Tokens` with a 1–800 range and still
  defaults to 800 in the active dashboard and backend configuration. The
  older active controls had divergent 64/256/512 defaults and ceilings, so
  the generation default and validation ceiling were unified at 800.
- Provider mapping remains normalized through the existing backend adapters:
  local Qwen uses `max_new_tokens`, while cloud adapters retain their existing
  provider-specific output-token fields. Context-window configuration was not
  changed.
- Validation: generation-parameter tests cover 1, 256, 512, 799, and 800 as
  accepted and 801 as rejected. The original-dashboard browser suite passed
  7 tests, including the real 800-token WebSocket payload and cancellation
  path. The existing offline local smoke test passed with 367 real events,
  including tokenization, embeddings, attention, QKV, MLP, logits, and token
  generation. Frontend build passed; the full backend suite passed with
  `109 passed, 3 skipped`.
- Cloud status: OpenAI, Gemini, and Anthropic remain backend-only and were not
  live-called because no valid credentials were configured. No cloud secret is
  exposed in the restored frontend.

## Output Visibility Adjustment

- The original dashboard’s bottom Output workspace was increased from its
  cramped 168px area to 232px while preserving the original layout and tabs.
- The streamed assistant response now has a dedicated scrollable response
  region, while the generated-token strip is bounded so long generations do
  not push the readable answer out of view.
- Verification: frontend build passed; the browser suite passed 7 tests,
  including assertions for the visible response region and expanded output
  workspace, plus local streaming and 800-token cancellation.

## Neuro Core Neural Field Animation

- Replaced the previous Neuro Core `BrainScene` composition with a dense,
  lighted neural-network scene while leaving the surrounding original
  dashboard and Observatory features unchanged.
- The scene renders the actual loaded layer count as illuminated neural
  columns, uses captured attention links for purple attention paths, uses real
  PCA points when the embedding-map toggle is enabled, and uses real output
  candidates at the output node.
- The network includes nine nodes per real layer, cross-layer bundles,
  intra-layer connections, long-range skip paths, and deterministic flowing
  light pulses. Pulse speed and intensity respond to the live generation
  state and active layer/module; no random or mock inference data is used.
- Layer/token selection, orbit controls, token lights, attention-path filter,
  embedding map, output candidates, and benchmark counters remain connected.
- Verification: frontend build passed; browser regression suite passed 8
  tests, including direct Neuro Core canvas/control rendering, local
  streaming, provider choices, navigation, cancellation, and narrow-layout
  coverage.

## Unified Model Selection and 800-Token Control

- The active original dashboard now exposes a `Provider` selector and a
  `Model` selector together. Local model choices use the existing backend
  model catalog and route through `/api/model/load`; cloud choices are sent as
  the selected provider model in the unified WebSocket request.
- Cloud mode now keeps the prompt/model controls mounted while showing the
  truthful external-observation notice, so selecting OpenAI, Gemini, or Claude
  does not remove the chat input.
- `Max Output Tokens` remains available for every provider with a maximum of
  800; provider adapters continue mapping that normalized value to their own
  API parameter.
- Verification: frontend build passed; browser suite passed 8 tests covering
  local model options, cloud model options, 800-token request/cancellation,
  local streaming, Neuro Core, and dashboard navigation.
