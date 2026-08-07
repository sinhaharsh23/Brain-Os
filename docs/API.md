# API

Base URL: `http://127.0.0.1:8765`

## Health And Hardware

- `GET /api/health`
- `GET /api/hardware`
- `GET /api/models`
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
- `GET /api/sessions/{session_id}/hidden?layer=0&position=0`
- `GET /api/sessions/{session_id}/logits?step=0&k=16`

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

Other commands are `cancel`, `replay`, `replay_pause`, and `replay_stop`. Events contain `type`, `data`, `ts`, and `session_id`.
