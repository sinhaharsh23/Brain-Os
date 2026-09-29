# BrainOS 3.0 — COMPLETE IMPLEMENTATION MASTER PLAN

You are now the **Lead AI Systems Engineer, ML Engineer, Backend Engineer, Frontend Engineer, 3D Visualization Engineer, QA Engineer, and Technical Architect** responsible for completing my existing project:

# BrainOS 3.0

Do NOT create a new unrelated project.

Work inside the existing BrainOS repository.

Your responsibility is to inspect everything already implemented, preserve everything that is working, repair incomplete/broken functionality, refactor poor architecture where necessary, implement every missing feature described below, integrate frontend and backend completely, test the complete application, and leave BrainOS 3.0 in a stable working state.

Do not stop after creating plans.

Do not simply generate TODO files.

Do not implement fake demonstrations.

Actually implement, run, debug, verify, test, and complete the project.

---

# 0. PRIMARY OBJECTIVE

BrainOS 3.0 must become a real-time **LLM Introspection, Visualization, Debugging, and Observability Platform** for locally running transformer models.

The core pipeline that BrainOS must visualize is:

Prompt
→ Tokenization
→ Token Embeddings
→ Transformer Layer
→ Attention
→ Query / Key / Value
→ Attention Heads
→ Residual Stream
→ MLP / Feed Forward Network
→ Hidden States
→ Layer-by-Layer Prediction
→ Logits
→ Token Probabilities
→ Sampling
→ Selected Token
→ KV Cache
→ Generated Token
→ Next Generation Step

The frontend must visualize information produced by the **actual model running in the backend**.

Never generate random activation values for visualization.

Never use hardcoded attention matrices.

Never simulate Q/K/V.

Never generate fake CPU/GPU statistics.

Never display fabricated transformer data.

If some information genuinely cannot be extracted from a specific model architecture, the UI must show:

`Unavailable for this model`

instead of showing fake information.

---

# 1. NON-NEGOTIABLE PROJECT CONSTRAINTS

## Deployment

BrainOS 3.0 MUST NOT be deployed.

Do not configure:

* Vercel
* AWS
* Azure
* GCP
* Railway
* Render
* Cloudflare
* Kubernetes
* production Docker infrastructure
* remote inference
* public hosting

The application is intended to run on:

`localhost`

only.

It should have a clean local development startup procedure.

---

## Primary Machine

Optimize BrainOS for an Apple Silicon Mac.

Use device selection similar to:

`MPS → CPU fallback`

Detect hardware automatically.

Never assume CUDA exists.

Do not make NVIDIA CUDA a dependency.

CUDA support may remain optional if existing code already supports it.

---

## AI Model Philosophy

The deepest BrainOS features must work with **local models** because hosted AI APIs usually do not expose internal transformer tensors.

The initial reference model should remain the existing Qwen-compatible local model.

After Qwen works perfectly, design adapters so additional Hugging Face causal language models can be supported.

---

# 2. GLOBAL ENGINEERING RULES

Before changing anything:

1. Inspect the complete repository.
2. Inspect frontend.
3. Inspect backend.
4. Inspect package files.
5. Inspect Python dependencies.
6. Inspect environment configuration.
7. Inspect model loading code.
8. Inspect current WebSocket implementation.
9. Inspect existing REST APIs.
10. Inspect Three.js / React Three Fiber implementation.
11. Identify completed features.
12. Identify partially implemented features.
13. Identify duplicated code.
14. Identify broken code.
15. Identify mock data.
16. Identify hardcoded visualization values.
17. Identify dead code.
18. Identify memory leaks.
19. Identify rendering performance issues.
20. Identify backend inference performance issues.

Do not unnecessarily rewrite working functionality.

Prefer incremental refactoring.

Every major component must have clean responsibility boundaries.

Use TypeScript strict typing wherever possible.

Use Python type hints.

Use structured errors.

Use predictable schemas between frontend and backend.

Do not stop because one approach fails.

Debug it and continue.

---

# =========================================================

# PART I — BACKEND / AI ENGINE / MODEL INTROSPECTION

# =========================================================

# BACKEND PHASE B0

# EXISTING SYSTEM AUDIT

Perform a complete backend audit before implementing new functionality.

Identify:

* backend framework
* backend folder structure
* model loader
* current Qwen integration
* Hugging Face integration
* PyTorch version
* transformers version
* WebSocket implementation
* token streaming logic
* tensor capture logic
* API routes
* environment variables
* session logic
* performance monitoring
* system monitoring
* existing tests

Create an internal implementation checklist from the actual repository.

Do not assume something is missing before checking.

Preserve working code.

Replace mock transformer information with real data.

---

# BACKEND PHASE B1

# LOCAL DEVELOPMENT FOUNDATION

BrainOS must start reliably on localhost.

Create a robust configuration system.

Configuration should support things such as:

```text
MODEL_ID
MODEL_PATH
DEVICE
DTYPE
MAX_NEW_TOKENS
DEFAULT_TEMPERATURE
DEFAULT_TOP_K
DEFAULT_TOP_P
ATTENTION_CAPTURE
INTROSPECTION_MODE
SESSION_DIRECTORY
```

Do not require unnecessary environment variables.

Provide safe defaults.

Device detection:

```text
if Apple MPS available:
    use MPS
else:
    use CPU
```

If CUDA already exists in architecture:

```text
CUDA
MPS
CPU
```

may be supported in that order according to availability/configuration.

Add readable startup logs:

```text
BrainOS Backend
Model: Qwen...
Device: MPS
Precision: ...
Introspection: Enabled
Server: http://localhost:....
```

Errors must be understandable.

Example:

Bad:

```text
RuntimeError
```

Good:

```text
Model failed to load because the configured model directory does not exist.
Expected path: ...
```

---

# BACKEND PHASE B2

# MODEL MANAGER

Create a centralized:

`ModelManager`

It must own:

* loading models
* unloading models
* tokenizer loading
* model metadata
* device selection
* dtype selection
* generation configuration
* memory cleanup
* currently loaded model state

The rest of BrainOS must not independently load copies of the model.

Prevent duplicate model instances.

Expose metadata:

```text
model name
architecture
parameter count
device
dtype
vocabulary size
context size
number of transformer layers
number of attention heads
number of KV heads
hidden dimension
head dimension
model memory
```

Calculate values from the actual loaded model configuration.

Do not hardcode architecture numbers.

---

# BACKEND PHASE B3

# MODEL ADAPTER ARCHITECTURE

Create a clean abstraction:

`BaseModelAdapter`

Example responsibilities:

```text
get_transformer_layers()
get_embedding_module()
get_attention_module()
get_q_projection()
get_k_projection()
get_v_projection()
get_mlp_module()
get_final_norm()
get_lm_head()
get_num_layers()
get_num_attention_heads()
get_num_kv_heads()
get_hidden_size()
get_head_dimension()
project_hidden_state_to_logits()
register_introspection_hooks()
remove_introspection_hooks()
```

Implement:

`QwenAdapter`

first.

Qwen must provide the deepest working introspection.

Then create a fallback:

`GenericHuggingFaceCausalLMAdapter`

The generic adapter should expose whatever information is genuinely available.

Never pretend that unsupported information exists.

Later support can include architectures such as:

* Llama-family
* Gemma
* Phi

but Qwen functionality must be completed before expanding.

---

# BACKEND PHASE B4

# INSTRUMENTED GENERATION ENGINE

Create:

`GenerationEngine`

BrainOS should support two generation modes.

## Instrumented Mode

This is BrainOS's primary mode.

Implement a manual autoregressive generation loop instead of relying entirely on:

`model.generate()`

Pseudo pipeline:

```text
encode prompt

run model

collect tensors

obtain final logits

calculate probabilities

apply sampling configuration

choose token

record generation step

stream token

update KV cache

repeat
```

Each generation step must have an ID.

Example:

```text
step = 0
step = 1
step = 2
```

Capture:

```text
token id
token text
timestamp
generation latency
top probabilities
entropy
selected token probability
layer information
cache length
```

Support:

* greedy decoding
* temperature
* top-k
* top-p
* max new tokens
* EOS
* cancellation
* deterministic random seed

Generation must be cancellable from frontend.

---

## Fast Mode

Optional secondary mode.

Fast mode may prioritize generation speed and capture fewer internals.

However:

BrainOS default visualization mode should use instrumented generation.

---

# BACKEND PHASE B5

# INTROSPECTION ENGINE

Create one central subsystem:

`IntrospectionEngine`

Do NOT scatter PyTorch hooks throughout random files.

The IntrospectionEngine should coordinate all tensor collection.

Suggested internal collectors:

```text
TokenCollector
EmbeddingCollector
HiddenStateCollector
AttentionCollector
QKVCollector
MLPCollector
ResidualCollector
LogitCollector
KVCacheCollector
PerformanceCollector
```

All hooks must be:

* registered before generation
* properly tracked
* removed after generation
* cleaned after errors
* cleaned after cancellation

Do not allow hooks to accumulate between requests.

---

# BACKEND PHASE B6

# TOKENIZATION CAPTURE

Capture real tokenizer information.

For every token expose:

```text
token index
token id
token text
decoded representation
special-token status
character span where possible
```

The frontend must be able to highlight individual tokens.

Prompt tokens and generated tokens must be distinguishable.

Example:

```text
type = prompt
type = generated
```

---

# BACKEND PHASE B7

# EMBEDDING CAPTURE

Capture actual input token embeddings.

Expose:

```text
embedding vector dimension
embedding norm
token-to-token similarity
selected vector slice
```

Do not continuously send huge embedding vectors through WebSocket.

Store raw vectors backend-side for the session.

Send only summaries initially.

Provide on-demand endpoints for detailed vectors.

Provide data required for PCA visualization.

PCA should operate on real embeddings.

Prefer lightweight local mathematical implementation.

Use existing dependency if already installed.

Do not introduce unnecessary heavy dependencies.

---

# BACKEND PHASE B8

# HIDDEN STATE CAPTURE

Capture transformer hidden states.

For each layer and selected token:

```text
layer index
tensor shape
vector norm
mean
std
min
max
selected dimensions
```

Raw hidden-state vectors should be available on demand.

Do not transmit all hidden states for all tokens continuously if it creates memory or performance problems.

Implement configurable capture levels:

```text
SUMMARY
SELECTED
FULL
```

Default:

`SUMMARY`

Developer Mode may request more detail.

---

# BACKEND PHASE B9

# REAL ATTENTION CAPTURE

Implement real attention extraction.

Capture:

```text
layer
head
source token
target token
attention score
```

Support:

```text
individual head
average heads
selected token attention
full attention matrix when reasonably sized
```

For transformer implementations that optimize attention and do not return weights, configure the model appropriately in instrumented mode when possible.

For Qwen:

inspect the installed Transformers implementation and use the correct supported mechanism to obtain real attention.

Do not invent attention scores.

If capturing attention requires switching from a faster attention implementation to an eager implementation in introspection mode, that is acceptable.

Document the performance tradeoff.

For long sequences, implement downsampling or selected-token inspection instead of sending gigantic matrices to the browser.

---

# BACKEND PHASE B10

# QUERY / KEY / VALUE CAPTURE

Implement real Q/K/V introspection.

Capture actual projection outputs.

For each:

```text
layer
token
attention head
vector dimension
vector norm
vector values on demand
```

Handle:

* multi-head attention
* grouped-query attention
* different number of attention heads vs KV heads

For Qwen specifically:

correctly respect:

```text
num_attention_heads
num_key_value_heads
head_dim
```

Reshape tensors correctly.

Never assume Q, K, and V have identical head counts.

UI requests such as:

```text
Layer 5
Head 3
Token 11
```

must return corresponding real vectors.

---

# BACKEND PHASE B11

# MLP / FEED-FORWARD INTROSPECTION

Capture actual MLP computation.

Where architecture permits, expose:

```text
MLP input
gate projection
up projection
activation
combined intermediate activation
down projection
MLP output
```

For Qwen-style gated MLPs, correctly calculate or capture the actual gated intermediate representation.

Provide:

```text
top activated neurons
activation magnitude
neuron index
activation distribution
```

Do not transmit the entire intermediate dimension on every generation step.

Provide:

```text
top 10
top 25
top 50
```

and on-demand full inspection where practical.

---

# BACKEND PHASE B12

# RESIDUAL STREAM

Capture residual-stream information.

For each transformer layer expose:

```text
layer input
attention contribution
post-attention residual
MLP contribution
layer output
```

At minimum BrainOS must be able to show:

```text
Input Residual
      ↓
Attention Contribution
      ↓
Residual Update
      ↓
MLP Contribution
      ↓
Layer Output
```

Calculate norms and differences.

Useful fields:

```text
input norm
attention delta norm
post-attention norm
MLP delta norm
output norm
cosine similarity
```

Residual data must come from actual model execution.

---

# BACKEND PHASE B13

# LOGITS AND TOKEN PROBABILITIES

At every generated token step capture:

```text
final logits
top candidate tokens
probabilities
selected token
selected token probability
entropy
```

Default top candidates:

`20`

Allow frontend to request more.

Expose generation settings alongside probabilities:

```text
temperature
top_k
top_p
sampling method
seed
```

---

# BACKEND PHASE B14

# LOGIT LENS

Implement a real layer-by-layer prediction explorer.

For selected layers:

1. obtain the hidden state
2. apply the appropriate final normalization strategy
3. project through the actual language-model head
4. calculate top token predictions

Expose:

```text
layer
top predicted tokens
logits
probabilities
```

Example:

```text
Layer 2  → token candidates
Layer 6  → token candidates
Layer 12 → token candidates
Final    → token candidates
```

Do not calculate every layer for every token if doing so severely damages performance.

Allow configurable sampling of layers.

Example:

```text
every layer
every 2 layers
selected layers
```

---

# BACKEND PHASE B15

# KV CACHE VISUALIZATION DATA

Capture real cache metadata.

Expose per layer:

```text
cache sequence length
key tensor shape
value tensor shape
estimated memory
number of cached tokens
```

By default do NOT stream the complete KV cache.

BrainOS primarily needs structural visualization.

Detailed slices may be requested on demand.

Show actual growth during autoregressive generation.

Example:

```text
step 1 → cache length 21
step 2 → cache length 22
step 3 → cache length 23
```

---

# BACKEND PHASE B16

# MODEL ARCHITECTURE INSPECTOR

Automatically inspect the loaded model.

Build a model architecture representation containing:

```text
module name
module type
parameter count
tensor dimensions where meaningful
parent
children
```

Expose a tree such as:

```text
Model
├── Embedding
├── Transformer
│   ├── Layer 0
│   │   ├── Attention
│   │   ├── MLP
│   │   └── Norm
│   ├── Layer 1
│   └── ...
├── Final Norm
└── LM Head
```

The frontend must not rely on a hardcoded number of layers.

---

# BACKEND PHASE B17

# PERFORMANCE PROFILER

Measure actual generation performance.

Collect:

```text
tokenization time
prompt processing time
time to first token
per-token latency
tokens per second
attention processing time where measurable
MLP processing time where measurable
frontend event serialization overhead
total generation time
```

If layer timing hooks are possible, expose:

```text
slowest layer
fastest layer
average layer time
```

Profiling should not itself destroy performance.

Allow profiling to be enabled/disabled.

---

# BACKEND PHASE B18

# SYSTEM MONITOR

Implement real system metrics.

Use actual APIs.

Capture:

```text
CPU usage
RAM used
RAM available
process RAM
thread count
model memory estimate
MPS allocated memory where PyTorch exposes it
```

Important:

Apple Silicon uses unified memory.

Do not label fake metrics as:

`VRAM`

if the information is really unified memory.

Use technically correct labels such as:

```text
MPS Allocated Memory
System Unified Memory
Process Memory
```

If actual GPU utilization percentage cannot be reliably obtained, display:

`GPU Utilization: unavailable`

instead of generating a random number.

---

# BACKEND PHASE B19

# SESSION RECORDING

Create local BrainOS session recording.

Do NOT require a remote database.

Suggested local structure:

```text
sessions/
    <session-id>/
        metadata.json
        events.jsonl
        tensors/
        metrics.json
```

A session should store:

```text
prompt
model
model metadata
generation configuration
tokens
generation events
probabilities
attention summaries
QKV summaries
MLP summaries
residual summaries
logit-lens information
cache metadata
performance metrics
timestamps
```

Large tensors should be stored efficiently.

Do not put giant tensors directly inside one huge JSON document.

Use compressed numeric storage where appropriate.

---

# BACKEND PHASE B20

# REPLAY ENGINE

Recorded sessions must be replayable without rerunning inference.

Create:

`ReplayService`

Frontend should be able to request:

```text
session
step
token
layer
head
```

and receive recorded data.

Replay must update:

```text
tokens
attention
QKV
MLP
residual
probabilities
logits
performance
3D state
```

---

# BACKEND PHASE B21

# COMPARISON ENGINE

Allow comparing two sessions.

Comparison dimensions:

```text
generated tokens
token probability
attention patterns
latency
tokens per second
memory usage
layer prediction
generation settings
```

Initially compare by:

```text
generation step
```

and where possible by token.

Use cases:

```text
Temperature 0.2 vs Temperature 1.0
```

and eventually:

```text
Model A vs Model B
```

---

# BACKEND PHASE B22

# API DESIGN

Create clean REST endpoints.

Exact naming may adapt to existing backend architecture.

Logical APIs should include functionality equivalent to:

```text
GET  /api/health

GET  /api/model
GET  /api/model/architecture

POST /api/generation/start
POST /api/generation/cancel

GET  /api/sessions
GET  /api/sessions/{id}

GET /api/introspection/token
GET /api/introspection/attention
GET /api/introspection/qkv
GET /api/introspection/mlp
GET /api/introspection/residual
GET /api/introspection/logits
GET /api/introspection/logit-lens
GET /api/introspection/kv-cache
GET /api/introspection/hidden-state
GET /api/introspection/embedding

GET /api/metrics/system
GET /api/metrics/performance
```

Do not duplicate endpoints unnecessarily.

Use response models/schemas.

Validate query parameters.

---

# BACKEND PHASE B23

# WEBSOCKET EVENT PROTOCOL

Use WebSocket for real-time generation.

Do not send unstructured random JSON.

Create typed event types.

Example:

```text
generation.started

prompt.tokens

token.generated

layer.summary

attention.summary

qkv.summary

mlp.summary

residual.summary

probabilities.updated

kv_cache.updated

performance.updated

system.updated

generation.completed

generation.cancelled

generation.error
```

Every event should include appropriate identifiers:

```text
session_id
generation_id
step_id
timestamp
```

Large tensors should NOT continuously travel over WebSocket.

WebSocket carries summaries and references.

Detailed tensor inspection should use on-demand API requests.

---

# BACKEND PHASE B24

# BACKPRESSURE AND MEMORY SAFETY

BrainOS introspection can generate enormous amounts of data.

Implement protections.

Do not retain unnecessary computation graphs.

Always use:

```python
tensor.detach()
```

before storing analysis data.

Move tensors to CPU only when necessary.

Do not repeatedly copy huge tensors from MPS to CPU.

Use:

```text
summary-first capture
on-demand raw vectors
bounded event queues
bounded session memory
```

Set reasonable limits for:

```text
attention matrix size
number of full hidden states
number of stored full tensors
WebSocket queue length
```

Long prompts must not freeze the application.

---

# BACKEND PHASE B25

# ERROR HANDLING

Implement structured BrainOS errors.

Categories should include:

```text
MODEL_LOAD_ERROR
MODEL_UNSUPPORTED
OUT_OF_MEMORY
INVALID_GENERATION_CONFIG
INTROSPECTION_UNAVAILABLE
SESSION_NOT_FOUND
GENERATION_CANCELLED
WEBSOCKET_ERROR
TENSOR_CAPTURE_ERROR
```

Frontend must receive meaningful error descriptions.

Backend must recover from cancelled or failed generations.

A failed generation must not require restarting the server.

---

# BACKEND PHASE B26

# TESTING

Create meaningful tests.

## Unit Tests

Test:

```text
sampling
probability calculation
entropy
tensor summaries
session serialization
adapter utilities
model metadata
event schemas
comparison logic
```

## Model Integration Tests

Using the actual configured local reference model, verify:

```text
model loads
tokenizer loads
generation works
token streaming works
attention capture works
QKV capture works
hidden state capture works
MLP capture works
residual capture works
logits work
logit lens works
KV cache metadata works
```

## WebSocket Tests

Verify:

```text
connection
start
stream
cancel
completion
errors
reconnection behavior
```

## Memory Tests

Run several generations consecutively.

Verify model memory does not continually increase because of leaked hooks or retained tensors.

---

# BACKEND PHASE B27

# BACKEND DEFINITION OF DONE

Backend is not complete until:

* model loads reliably
* MPS works
* CPU fallback works
* generation works
* token streaming works
* cancellation works
* attention is real
* Q/K/V are real
* MLP data is real
* residual data is real
* hidden states are real
* embeddings are real
* probabilities are real
* logit lens works
* KV cache metadata works
* architecture inspection works
* sessions save
* sessions replay
* session comparison works
* performance monitoring works
* system metrics work
* hooks are cleaned correctly
* errors recover correctly
* tests pass

---

# =========================================================

# PART II — FRONTEND + UI/UX + 3D VISUALIZATION

# =========================================================

# FRONTEND PHASE F0

# FRONTEND AUDIT

Inspect the current frontend completely.

Identify:

```text
framework
React version
TypeScript configuration
Three.js setup
React Three Fiber
state management
WebSocket client
REST client
component architecture
styles
existing visualizations
mock data
hardcoded data
responsive issues
render performance
```

Do not rewrite good existing components without reason.

---

# FRONTEND PHASE F1

# FINAL APPLICATION LAYOUT

BrainOS should feel like a professional AI research/debugging application.

Target desktop layout:

```text
┌─────────────────────────────────────────────────────────────┐
│ BrainOS | Model | Device | Status | Session | Settings      │
├──────────────┬────────────────────────────┬─────────────────┤
│              │                            │                 │
│ MODEL /      │                            │ INSPECTOR       │
│ LAYER        │       3D AI VIEW           │                 │
│ EXPLORER     │                            │ Attention       │
│              │                            │ QKV             │
│ Tokens       │                            │ MLP             │
│ Layers       │                            │ Residual        │
│ Heads        │                            │ Logits          │
│ Architecture │                            │ Tensor          │
│              │                            │                 │
├──────────────┴────────────────────────────┴─────────────────┤
│ GENERATION TIMELINE                                         │
├─────────────────────────────────────────────────────────────┤
│ Prompt / Generation Console                                 │
└─────────────────────────────────────────────────────────────┘
```

It should work well on a 14-inch MacBook screen.

Support panel resizing/collapsing where useful.

---

# FRONTEND PHASE F2

# DESIGN LANGUAGE

Design BrainOS as an AI research laboratory.

Visual principles:

```text
dark
clean
technical
high contrast
minimal clutter
data-first
smooth
modern
professional
```

Avoid childish visuals.

Avoid excessive gradients.

Avoid unnecessary glowing effects everywhere.

Animations should communicate model activity.

Not decorate randomly.

Use consistent:

```text
spacing
typography
border radius
panel structure
tooltips
buttons
selectors
tabs
charts
```

---

# FRONTEND PHASE F3

# APPLICATION STATE ARCHITECTURE

Create clean state domains.

Example:

```text
connection state
model state
generation state
token state
selected token
selected layer
selected head
introspection state
timeline state
session state
metrics state
3D visualization state
UI preferences
```

Do not place everything inside one massive component.

If no suitable state manager currently exists, use a lightweight solution such as Zustand.

If the existing project already uses a good state-management solution, preserve it.

---

# FRONTEND PHASE F4

# TYPED BACKEND CLIENT

Create a centralized API layer.

Do not call:

`fetch()`

from dozens of components randomly.

Create typed services:

```text
BrainOSApi
ModelApi
GenerationApi
IntrospectionApi
SessionApi
MetricsApi
```

Create one centralized WebSocket manager.

It should handle:

```text
connect
disconnect
reconnect
messages
error
generation events
cancellation
```

---

# FRONTEND PHASE F5

# PROMPT / GENERATION CONSOLE

Create a proper generation interface.

Controls:

```text
prompt textarea
generate
stop
clear
temperature
top-k
top-p
max tokens
seed
generation mode
```

Display:

```text
prompt tokens
generated tokens
streaming output
current generation state
```

Generated tokens should be individually clickable.

Clicking a token selects its:

```text
generation step
attention
QKV
MLP
residual
probabilities
logits
```

---

# FRONTEND PHASE F6

# TOKEN EXPLORER

Display tokens clearly.

Differentiate:

```text
prompt token
generated token
selected token
special token
```

Hover tooltip:

```text
token
token ID
position
type
probability where relevant
```

Selecting a token must synchronize the whole interface.

---

# FRONTEND PHASE F7

# 3D BRAIN / TRANSFORMER VISUALIZATION

The center visualization is a major BrainOS feature.

Keep/use React Three Fiber + Three.js where appropriate.

The visualization must represent actual model architecture.

Possible structural representation:

```text
Embedding

Layer 0
    Attention heads
    MLP

Layer 1
    Attention heads
    MLP

...

Final Norm

LM Head
```

Do NOT create thousands of unnecessary physical "neurons" if that destroys performance.

Use abstraction.

Represent:

```text
layers
heads
connections
signals
activations
token flow
```

Use real metrics to control visual state.

Examples:

```text
strong activation → stronger visual emphasis
selected layer → highlighted
selected attention head → highlighted
token progressing → animated signal
```

Do not use random animations to represent computation.

---

# FRONTEND PHASE F8

# 3D VISUALIZATION MODES

Provide view modes.

## Architecture Mode

Shows model structure.

## Activation Mode

Highlights layers/MLP regions with actual activation magnitude.

## Attention Mode

Shows token/head attention relationships.

## Residual Flow Mode

Shows how residual-state magnitude changes through layers.

## Generation Mode

Shows a token progressing through the model.

View mode changes should not recreate the entire WebGL scene unnecessarily.

---

# FRONTEND PHASE F9

# CAMERA AND 3D CONTROLS

Implement:

```text
orbit
zoom
pan where appropriate
reset view
focus selected layer
focus selected head
fit architecture
```

Provide:

`Reset Camera`

The user must never become permanently lost in 3D space.

---

# FRONTEND PHASE F10

# ATTENTION EXPLORER

Create a proper attention heatmap.

Controls:

```text
layer selector
head selector
average heads
selected token
```

Heatmap axes:

```text
source tokens
target tokens
```

Hover:

```text
source token
target token
attention score
```

Clicking a heatmap cell may select tokens.

For long sequences implement:

```text
zoom
scroll
selected-token view
downsampling
```

Do not freeze browser rendering.

---

# FRONTEND PHASE F11

# Q / K / V INSPECTOR

Create dedicated Q/K/V interface.

Show:

```text
selected layer
selected head
selected token
```

Tabs:

```text
QUERY
KEY
VALUE
```

Display:

```text
vector dimension
norm
mean
std
selected vector values
```

Provide compact visualization of vector components.

Do not try to render thousands of raw dimensions simultaneously.

Add:

```text
show first N
show strongest dimensions
show full table
```

where appropriate.

---

# FRONTEND PHASE F12

# MLP / NEURON ACTIVATION EXPLORER

Show:

```text
layer
token
top activated neurons
activation values
```

Visualization:

```text
Neuron #125      ██████████
Neuron #481      ███████
Neuron #2012     █████
```

Controls:

```text
Top 10
Top 25
Top 50
```

Display activation distribution.

Click neuron to inspect actual value.

---

# FRONTEND PHASE F13

# RESIDUAL STREAM VISUALIZER

Build a dedicated flow visualization.

Example:

```text
Layer Input
     │
     ▼
Attention Delta
     │
     ▼
Post-Attention Residual
     │
     ▼
MLP Delta
     │
     ▼
Layer Output
```

Display:

```text
vector norms
delta norms
cosine similarity
relative contribution
```

Add layer-to-layer residual chart.

---

# FRONTEND PHASE F14

# TOKEN PROBABILITY PANEL

For every generated token display top candidate tokens.

Example:

```text
blue        72.4%
clear        9.2%
bright       5.1%
dark         3.7%
beautiful    2.8%
```

Clearly mark:

`SELECTED`

Display:

```text
temperature
top-k
top-p
entropy
```

Make probability bars proportional to actual probabilities.

---

# FRONTEND PHASE F15

# LOGIT LENS VIEWER

Build a layer-by-layer token prediction viewer.

Example:

```text
Layer 2
Paris     4%
France    3%

Layer 6
Paris    19%
France    8%

Layer 12
Paris    71%

Final
Paris    94%
```

Allow:

```text
selected layers
all sampled layers
selected token generation step
```

Add a visual evolution graph showing how candidate probabilities change across layers.

---

# FRONTEND PHASE F16

# EMBEDDING SPACE EXPLORER

Provide:

```text
2D PCA
3D PCA
```

Use actual embedding vectors.

Each point represents a token.

Hover shows:

```text
token
token ID
position
```

Clicking a point selects that token in BrainOS.

Allow:

```text
input embedding
selected hidden-state layer
```

if backend exposes both.

---

# FRONTEND PHASE F17

# KV CACHE VISUALIZER

Visualize cache growth.

Example:

```text
Layer 0    ██████████████
Layer 1    ██████████████
Layer 2    ██████████████
```

Display:

```text
cached tokens
cache sequence length
key shape
value shape
estimated memory
```

Animate growth as tokens generate.

Do not pretend cache entries are neurons.

Use technically accurate representation.

---

# FRONTEND PHASE F18

# GENERATION TIMELINE

The generation timeline is essential.

Represent:

```text
Prompt tokens

Generated token 1

Generated token 2

Generated token 3
...
```

Controls:

```text
play
pause
previous step
next step
first
last
live
```

Selecting a timeline position must update:

```text
3D view
attention
QKV
MLP
residual
probabilities
logit lens
KV cache
performance
```

Everything should represent the same generation step.

---

# FRONTEND PHASE F19

# REPLAY MODE

Recorded sessions should open in:

`Replay Mode`

Replay Mode should look almost identical to live inference.

Differences:

```text
no inference running
timeline completely available
playback speed control
step backward
step forward
```

Support speeds such as:

```text
0.25x
0.5x
1x
2x
```

---

# FRONTEND PHASE F20

# MODEL ARCHITECTURE EXPLORER

Build a collapsible tree.

Example:

```text
Qwen
├── Embeddings
├── Transformer
│   ├── Layer 0
│   │   ├── Attention
│   │   │   ├── Q Projection
│   │   │   ├── K Projection
│   │   │   ├── V Projection
│   │   │   └── Output Projection
│   │   ├── MLP
│   │   └── RMSNorm
│   └── ...
├── Final Norm
└── LM Head
```

Clicking a component should focus it where possible.

Display:

```text
module name
type
parameter count
tensor shape
```

---

# FRONTEND PHASE F21

# NORMAL MODE AND DEVELOPER MODE

Provide two modes.

## NORMAL MODE

Designed for visual understanding.

Show:

```text
tokens
architecture
attention
activation flow
simple explanations
probabilities
timeline
```

## DEVELOPER MODE

Expose advanced information.

Show:

```text
tensor names
tensor shapes
dtype
device
module paths
raw vector slices
norms
statistics
timings
advanced model metadata
```

Example:

```text
Module
model.layers.12.self_attn.q_proj

Tensor shape
[1, 14, 1, 64]

dtype
float32

device
mps
```

---

# FRONTEND PHASE F22

# TENSOR INSPECTOR

Create a reusable Tensor Inspector.

Support:

```text
shape
dtype
device
mean
std
min
max
norm
number of elements
vector slice
matrix slice
```

For large tensors:

do NOT render every element immediately.

Use pagination/slicing.

---

# FRONTEND PHASE F23

# PERFORMANCE DASHBOARD

Display:

```text
Tokens / second
Time to first token
Average token latency
Current token latency
Total generation time
Prompt processing time
```

If layer timings exist:

```text
Slowest layer
Average layer duration
```

Use charts that update smoothly without causing the UI to re-render excessively.

---

# FRONTEND PHASE F24

# SYSTEM DASHBOARD

Display actual backend system metrics.

Examples:

```text
CPU
RAM
Process Memory
MPS Allocated Memory
Model Memory
```

Do not display fake GPU utilization.

Clearly indicate unavailable metrics.

---

# FRONTEND PHASE F25

# SESSION MANAGER

Create local session management UI.

Functions:

```text
save session
rename session
open session
delete session
replay session
compare session
```

Display:

```text
date
model
prompt
number of tokens
duration
generation settings
```

No cloud account required.

---

# FRONTEND PHASE F26

# COMPARE MODE

Create:

`Compare Mode`

Support:

```text
Session A
vs
Session B
```

Compare:

```text
generation settings
tokens
probabilities
latency
tokens per second
memory
attention summaries
logit-lens results
```

Example use case:

```text
Temperature 0.2
vs
Temperature 1.0
```

Eventually support different models when backend adapters permit it.

---

# FRONTEND PHASE F27

# LOADING STATES

Every asynchronous operation needs proper UI state.

Examples:

```text
Loading model…
Initializing introspection…
Connecting to backend…
Generating…
Capturing attention…
Loading session…
Processing PCA…
```

Do not leave blank areas.

---

# FRONTEND PHASE F28

# ERROR STATES

Show clear errors.

Examples:

```text
Backend offline

Model could not load

MPS unavailable — switched to CPU

Attention extraction unavailable for this architecture

Generation cancelled

Session corrupted
```

Provide recovery actions such as:

```text
Retry
Reconnect
Reload Model
Dismiss
```

where appropriate.

---

# FRONTEND PHASE F29

# RESPONSIVE BEHAVIOR

Primary optimization target:

desktop/laptop.

Especially 14-inch MacBook.

Panels should not overlap.

Use collapsible sidebars.

Allow inspector resizing.

On smaller displays reduce the number of simultaneously visible panels.

Do not destroy functionality.

---

# FRONTEND PHASE F30

# PERFORMANCE OPTIMIZATION

BrainOS must remain responsive while inference runs.

Optimize React rendering.

Avoid global re-render for every system metric update.

Use memoization where justified.

Throttle extremely frequent UI events.

For Three.js:

```text
reuse geometry
reuse materials
avoid unnecessary object creation
use instancing where appropriate
use level-of-detail abstraction
```

Do not render one 3D mesh per individual neuron for billions of parameters.

Represent architecture intelligently.

Target smooth interaction.

---

# FRONTEND PHASE F31

# ACCESSIBILITY AND USABILITY

Provide:

```text
keyboard navigation
clear focus states
readable contrast
tooltips
labels
empty states
```

Use terminology consistently.

Example:

Do not alternate randomly between:

```text
FFN
MLP
Feed Forward
```

without explanation.

Use one main term and show aliases where helpful.

---

# FRONTEND PHASE F32

# FRONTEND TESTING

Test components involving:

```text
token selection
layer selection
head selection
timeline selection
WebSocket updates
generation cancellation
sessions
replay
compare mode
error states
```

Add integration tests for:

```text
prompt → generation
token → inspector sync
timeline → all panels sync
session → replay
```

No UI panel should rely on mock values during normal application execution.

---

# FRONTEND PHASE F33

# FRONTEND DEFINITION OF DONE

Frontend is not finished until:

* prompt input works
* generation streams
* tokens are interactive
* 3D visualization uses actual backend data
* architecture reflects loaded model
* attention explorer works
* QKV explorer works
* MLP explorer works
* residual viewer works
* probabilities work
* logit lens works
* embeddings work
* KV-cache viewer works
* timeline works
* replay works
* Developer Mode works
* Tensor Inspector works
* performance dashboard works
* system dashboard works
* sessions work
* Compare Mode works
* loading states work
* errors work
* resizing works
* application remains responsive

---

# =========================================================

# FINAL FRONTEND ↔ BACKEND INTEGRATION

# =========================================================

After backend and frontend phases are individually complete, verify every feature end-to-end.

Use one actual generation session.

Example prompt:

```text
Explain why the sky appears blue.
```

Verify this exact flow:

```text
Prompt submitted

↓

Tokenizer data appears

↓

Prompt tokens appear

↓

Model processes prompt

↓

3D architecture activates

↓

Attention becomes available

↓

Q/K/V becomes inspectable

↓

Hidden states become available

↓

MLP activation becomes available

↓

Residual-stream data appears

↓

Logit-lens data appears

↓

Final logits appear

↓

Top token probabilities appear

↓

Selected token is shown

↓

KV cache increases

↓

Generated token streams to UI

↓

Timeline adds generation step

↓

Repeat
```

When generation completes:

```text
session saved

↓

timeline replayable

↓

all historical introspection accessible

↓

performance available

↓

session can be compared
```

---

# SYNCHRONIZATION REQUIREMENT

At any moment BrainOS should have one canonical selection state:

```text
session
generation step
token
layer
head
```

Every visualization should derive from that selection.

If user selects:

```text
Token 17
Layer 8
Head 4
```

then:

```text
3D view
attention heatmap
QKV
MLP
residual
logit lens
tensor inspector
```

must all represent compatible information.

Do not allow panels to silently display different generation steps.

---

# REAL-DATA REQUIREMENT

Search the complete codebase for things such as:

```text
Math.random
random attention values
random activation values
hardcoded token probabilities
fake system metrics
fake tensor values
demo tensor arrays
```

Determine whether each is legitimate.

Remove fake runtime AI visualization.

Development/test fixtures may exist inside tests, but production UI must use real model values.

---

# LOCALHOST DEVELOPMENT EXPERIENCE

BrainOS must be easy to start.

Create or preserve a simple workflow.

Prefer a root-level local development helper.

For example:

```text
./scripts/dev.sh
```

or an equivalent command suited to the existing architecture.

It should start:

```text
backend
frontend
```

and properly stop both when terminated.

Do not introduce Docker solely for local development unless Docker already forms an essential part of the repository.

The final project should clearly document:

```text
install backend dependencies
install frontend dependencies
configure model
start BrainOS
open localhost URL
```

---

# DOCUMENTATION

When implementation is complete, update:

`README.md`

Include:

```text
What BrainOS is

Architecture

Requirements

Apple Silicon setup

Python setup

Node setup

Model setup

How to run

Available features

Developer Mode

Sessions

Replay

Troubleshooting
```

Create:

`BRAINOS_ARCHITECTURE.md`

Explain:

```text
Frontend
Backend
ModelManager
GenerationEngine
IntrospectionEngine
ModelAdapter
Session system
WebSocket
3D visualization
```

Create:

`BRAINOS_3_COMPLETION_REPORT.md`

This file must honestly describe:

```text
features completed
features tested
tests passed
model tested
hardware tested
known technical limitations
commands required to run
```

Do not claim unsupported features as complete.

---

# CODE QUALITY REQUIREMENTS

Do not create giant files unless justified.

Separate responsibilities.

Use clear names.

Remove abandoned implementations after safely replacing them.

Avoid duplicated interfaces.

Avoid duplicated type definitions.

Share API schemas/types where practical.

Document complex transformer introspection code.

Especially document:

```text
QKV reshaping
grouped-query attention
residual calculations
logit lens
hook lifetime
KV-cache interpretation
```

---

# PERFORMANCE SAFETY

The purpose of BrainOS is introspection, but introspection must not crash the computer.

Implement sensible controls.

Example settings:

```text
Capture Level:
Low
Medium
Full
```

Possible behavior:

### Low

```text
tokens
probabilities
system metrics
basic layer summaries
```

### Medium

```text
attention
selected QKV
MLP summaries
residual summaries
```

### Full

```text
developer tensor inspection
additional hidden states
detailed layer data
```

Default should be balanced.

---

# LONG-SEQUENCE SAFETY

For very long contexts:

Do not automatically transmit:

```text
every attention value
every QKV vector
every hidden-state vector
```

Use selected-token/layer/head inspection.

Show warnings when full capture would consume significant memory.

---

# CANCELLATION

When user presses:

`STOP`

BrainOS must stop generation safely.

It must:

```text
stop generation loop
flush final events
remove hooks
free temporary tensors
mark session cancelled
keep completed generation steps available for inspection
```

Backend should remain ready for another prompt.

---

# CONNECTION RECOVERY

If frontend reloads while backend remains active:

attempt reconnection.

Backend health endpoint should help frontend determine system state.

WebSocket errors should not permanently destroy the UI.

---

# FINAL VALIDATION RUN

After all implementation work is completed:

Run BrainOS from a clean start.

Verify no server from earlier development is accidentally satisfying tests.

Then test:

## Test 1

Short prompt.

## Test 2

Longer prompt.

## Test 3

Generation cancellation.

## Test 4

Second generation after cancellation.

## Test 5

Attention inspection.

## Test 6

QKV inspection.

## Test 7

MLP inspection.

## Test 8

Residual inspection.

## Test 9

Logit Lens.

## Test 10

KV cache.

## Test 11

Replay.

## Test 12

Compare Mode.

## Test 13

Backend restart.

## Test 14

Frontend reload.

## Test 15

Several generations consecutively to check memory leakage.

---

# BRAINOS 3.0 FINAL DEFINITION OF COMPLETE

BrainOS 3.0 is complete only when it behaves as a real working system.

It must provide:

```text
Local LLM execution

Real-time token streaming

Real transformer architecture visualization

Real tokenization

Real embeddings

Real attention

Real Query vectors

Real Key vectors

Real Value vectors

Real hidden states

Real MLP activations

Real residual-stream information

Real logits

Real probabilities

Real logit lens

Real KV-cache metadata

Generation timeline

Pause / inspect / replay

Developer Mode

Tensor Inspector

Model architecture explorer

System metrics

Inference performance metrics

Session recording

Session replay

Session comparison

3D transformer visualization

Stable Mac localhost operation
```

---

# FEATURES THAT ARE NOT PART OF BRAINOS 3.0

Do NOT spend project time implementing:

```text
public cloud deployment
multi-user SaaS
user accounts
payments
remote GPU clusters
Kubernetes
OpenAI internal tensor visualization
Claude internal tensor visualization
Gemini internal tensor visualization
distributed inference
full model training
fine-tuning platform
agent orchestration
MCP marketplace
```

Those belong to a future BrainOS version.

---

# EXECUTION ORDER

Follow this priority:

```text
1. Inspect repository

2. Make existing project run correctly

3. Stabilize Qwen model loading

4. Stabilize Apple MPS / CPU fallback

5. Stabilize token generation

6. Stabilize WebSocket streaming

7. Build ModelManager

8. Build ModelAdapter architecture

9. Build GenerationEngine

10. Build IntrospectionEngine

11. Real token capture

12. Real embedding capture

13. Real hidden states

14. Real attention

15. Real Q/K/V

16. Real MLP activations

17. Real residual stream

18. Real logits / probabilities

19. Logit Lens

20. KV-cache introspection

21. Model architecture API

22. Session recording

23. Replay

24. Performance profiler

25. System monitor

26. Comparison engine

27. Refactor frontend state

28. Stabilize frontend-backend protocol

29. Prompt / token UI

30. Model explorer

31. Real 3D visualization

32. Attention heatmap

33. QKV inspector

34. MLP explorer

35. Residual explorer

36. Probability explorer

37. Logit Lens interface

38. Embedding explorer

39. KV-cache interface

40. Timeline

41. Replay UI

42. Developer Mode

43. Tensor Inspector

44. Performance dashboard

45. System dashboard

46. Session manager

47. Compare Mode

48. Error handling

49. UI optimization

50. Backend optimization

51. Memory-leak testing

52. End-to-end testing

53. Documentation

54. Final clean run
```

---

# VERY IMPORTANT EXECUTION INSTRUCTION

Do not stop after one phase and ask me to continue.

Continue through the implementation plan autonomously.

When a problem occurs:

1. investigate it
2. identify the root cause
3. fix it
4. test the fix
5. continue

Only leave a feature incomplete when there is a genuine technical limitation that cannot be solved from the local model/library.

If this happens:

document the exact limitation in:

`BRAINOS_3_COMPLETION_REPORT.md`

Never replace an impossible feature with fake output.

---

# COMPLETION STANDARD

Do not use statements such as:

```text
This should work.
```

Verification must be based on:

```text
I ran it.
I tested it.
The actual data was returned.
The frontend displayed it.
The expected behavior was observed.
```

Do not mark a phase complete merely because code was written.

A phase is complete only after its functionality has been executed and verified.

---

# FINAL RESULT EXPECTED FROM YOU

At the end I should be able to open the BrainOS repository and start the application locally.

I should be able to enter a prompt and watch the real local transformer generate text while BrainOS exposes and visualizes its internal computation.

I should be able to click a generated token and inspect what happened inside the transformer for that generation step.

I should be able to explore:

```text
Layers
Attention Heads
Q
K
V
MLP
Residual Stream
Hidden States
Logits
Probabilities
Logit Lens
KV Cache
Embeddings
Performance
System Usage
```

I should then be able to save/replay the generation session and compare it with another run.

That is the completion target for BrainOS 3.0.

Begin by inspecting the entire existing repository and then execute the phases in the order above.

Do not deploy anything.

BrainOS 3.0 must remain a fully functional localhost application.
