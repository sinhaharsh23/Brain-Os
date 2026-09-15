# Troubleshooting

## Model Loading

The first launch downloads the checkpoint. Check `HF_HOME`, disk space, and network access if startup remains in `loading`.

If memory is tight, set `BRAINOS_DTYPE=bfloat16` or use the smaller default checkpoint. CPU fallback is automatic when MPS/CUDA/ROCm acceleration is unavailable or explicitly disabled.

## Frontend Build

The frontend requires a current Node version. Vite 7 prefers Node `20.19+` or Node 22. If npm reports a missing optional native binding, run `npm install --include=optional` and repeat `npm run build`.

## WebSocket

The frontend expects the backend WebSocket at `ws://127.0.0.1:8765/ws`. In development, keep the backend running on port `8765`. Browser devtools should show `model.ready` before sending a run command.

If `BRAINOS_AUTH_TOKEN` is configured, build the frontend with the matching `VITE_BRAINOS_TOKEN`. Requests without the token receive `401` and WebSocket connections close with policy code `1008`.

## Slow CPU Generation

The initial prompt pass is slower because it computes full prompt attention. Generated cached steps are faster. Lower `max_new_tokens`, use a shorter prompt, or select a smaller model if supported.

## Honest Availability States

Tensor endpoints return `404` when a requested capture is unavailable. This is intentional: BrainOS does not synthesize unavailable activations or attention.
