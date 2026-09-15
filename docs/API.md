# API

Base URL: `http://127.0.0.1:8765`

When `BRAINOS_AUTH_TOKEN` is set, send it as `Authorization: Bearer ...` or `X-BrainOS-Token: ...`. WebSocket clients send it as the `token` query parameter. Requests are rate-limited per client identity.

## Health And Hardware

- `GET /api/health`
- `GET /api/ready`
- `GET /api/hardware`
- `GET /api/models`
- `GET /api/providers`
- `GET /api/model`
- `GET /api/monitoring/history`

## Model And Tokenization

- `POST /api/model/load` with `{ "model_id": "Qwen/Qwen2.5-0.5B-Instruct" }`
- `POST /api/tokenize` with `{ "text": "...", "use_chat_template": true }`

## Sessions

- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `GET /api/sessions/{session_id}/events`
- `GET /api/sessions/{session_id}/embedding/{position}`
- `GET /api/sessions/{session_id}/attention?layer=0&head=0&position=0`
- `GET /api/sessions/{session_id}/qkv?layer=0&name=q&position=0`
- `GET /api/sessions/{session_id}/mlp?layer=0&position=0&topk=16`
- `GET /api/sessions/{session_id}/mlp?layer=0&position=0&topk=16&neuron=1432`
- `GET /api/sessions/{session_id}/hidden?layer=0&position=0`
- `GET /api/sessions/{session_id}/logits?step=0&k=16`

## Authenticated workspaces and projects

In `BRAINOS_AUTH_MODE=multi_user`, these routes require the owning user's
token. A missing or foreign resource returns `404` to avoid resource
enumeration.

- `POST /api/workspaces` with `{ "name": "Research" }`
- `GET /api/workspaces`
- `GET /api/workspaces/{workspace_id}`
- `PATCH /api/workspaces/{workspace_id}` with `{ "name": "Renamed" }`
- `DELETE /api/workspaces/{workspace_id}`
- `POST /api/projects` with `{ "workspace_id": "...", "name": "Trace run" }`
- `GET /api/projects?workspace_id=...`
- `GET /api/projects/{project_id}`
- `PATCH /api/projects/{project_id}` with `{ "name": "Renamed" }` or a new owner workspace ID
- `DELETE /api/projects/{project_id}`

Workspace deletion removes its owned projects and clears their nullable
association from sessions. Project deletion likewise preserves the session
record and clears only its project association.

## WebSocket

Connect to `/ws` and send:

```json
{
  "action": "run",
  "prompt": "Explain artificial intelligence.",
  "params": {
    "max_new_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "use_chat_template": true
  }
}
```

Other commands are `cancel`, `pause_inference`, `resume_inference`, `replay`, `replay_pause`, `replay_seek`, `replay_step`, and `replay_stop`. Events contain `type`, `data`, `ts`, and `session_id`.

`pause_inference` is cooperative. It never corrupts or forcibly interrupts a PyTorch forward pass. The backend emits `inference.paused` between steps and `inference.resumed` when the next step is allowed to start.

`tokenization.complete` includes `context_used`, `context_remaining`, and `context_length`. `inference.complete.data.timings` includes `ttft_ms`, `tokens_per_second`, and `total_ms` when available.

External provider runs use the same `run` command with `provider` set to `openai`, `anthropic`, or `google`. The optional `params.model` selects a provider model; `/api/providers` lists the configured defaults. API keys are read only by the backend from environment variables or the backend `.env` file and are never returned to the frontend. External events are `external.started`, `external.chunk`, and `external.usage`, followed by `inference.complete`; timing and provider-reported token usage are included when available. No internal tensor events are emitted for proprietary providers, and the UI labels these runs `External Observation`.
