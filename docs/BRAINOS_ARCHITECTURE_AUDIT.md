# BrainOS architecture audit

This document records the existing architecture used by the unified H3D dashboard. It is intentionally based on the repository implementation; BrainOS does not contain a training or RAG subsystem, so the UI does not pretend that either exists.

## Runtime entry and ownership

```text
frontend/src/main.tsx
  -> App.tsx
     -> WebSocket connect() + initial REST health/hardware/model/auth calls
     -> Dashboard3 (the single application shell)
        -> Dashboard3Header (one question/search input and inference controls)
        -> Dashboard3Sidebar (home + existing observability modules)
        -> HolographicStage3D / telemetry / output panels
        -> Dashboard3FeatureWorkspace (existing feature views inside the shell)
     -> AuthPanel (only when the backend requires multi-user auth)
```

`useBrainStore.ts` is the browser event reducer. `ws/client.ts` owns connection, reconnect, and action dispatch. `api/client.ts` owns typed REST calls. `useDashboard3Store.ts` is only a presentation bridge for the H3D shell; it does not simulate inference data.

## Existing functional features preserved

- Local Qwen inference with streaming output, cancellation, pause/resume, queue status, and generation parameters.
- Provider registry for Qwen local (deep inspection) and OpenAI, Anthropic, and Google external observation (limited metadata/text mode).
- Real token IDs/positions, generated-token probabilities, token flow, embedding PCA, attention links, Q/K/V statistics, MLP top neurons, logits/candidate tokens, layer hidden-state statistics, timing, and hardware monitoring.
- Architecture, Attention, Embeddings, Token Flow, Developer/Tensor Probe, Model Hub, Memory Matrix, Training Hub, Plugins, System Monitor, Settings, and Neural Interface sections. They are mounted inside one dashboard rather than separate legacy shells.
- Session persistence and ownership, REST tensor endpoints, WebSocket event streaming, replay loading/playback/seek/step/stop, and multi-user auth.
- Responsive layout, light/dark/system theme, file-to-prompt loading, clipboard copy, text-to-speech, fullscreen, and graceful offline/error notices.

The current backend exposes inference/observability rather than document retrieval, fine-tuning, agent execution, image generation, or automation orchestration. Those dashboard options now explain the limitation or route to the real model/diagnostic view instead of showing mock records.

## Real question-to-answer pipeline

```text
H3D question input
  -> ws action: run { provider, prompt, params }
  -> backend auth/rate-limit/parameter validation
  -> scheduler queue and owner binding
  -> local adapter + tokenizer
  -> tokenization.complete
  -> embedding hooks + PCA capture
  -> transformer layer hooks
       attention / QKV / MLP / hidden-state capture
  -> logits and candidate probabilities
  -> generated token events and streamed response
  -> inference.complete summary + replay persistence
  -> useBrain reducer
  -> H3D output, token cards, telemetry, layers, attention, and run log
```

For external providers, the same input/action boundary is used, but the backend intentionally emits only provider response text, usage, and timing. The UI labels this `External Observation` and explicitly avoids inventing private tensors.

Every captured stage has an event payload, reducer path, and visible fallback. Empty prompts, disconnected sockets, local-model loading, queue/rate-limit errors, provider failures, cancelled/timed-out runs, and a completed run with no response are surfaced as notices rather than silent failures.

## NeuroCore removal

The legacy NeuroCore/BrainScene route and its decorative `brain3d` implementation were removed. The old shell, old prompt bar, old bottom panel, obsolete inspector/model explorer, mock hook, and Antigravity's unused duplicate `components/hologram` tree were also removed because they were not connected to the real backend and would have created duplicate dashboards. The active H3D display is `components/dashboard3`, and its layer projection is driven by the real model layer count and captured layer progress.

## Verification record

The checks for this change are:

- `npm run build` — TypeScript and Vite production build pass.
- `npm run test:browser -- --reporter=line` — all 8 local observatory tests pass, including provider modes, real inference, 3D frame probe, two-browser isolation, external-provider labeling, module navigation/replay, and narrow viewport overflow.
- `npm run test:browser:auth -- --reporter=line` — both multi-user auth tests pass, including registration, real inference, owned sessions, logout, and invalid-password handling.
- `backend/.venv/bin/pytest -q` — 89 passed, 3 skipped; the only remaining output is the environment-level `urllib3` LibreSSL compatibility warning, not an application failure.
- Manual-style Chromium smoke verification — empty input notice, one-token real answer, 13 primary/observability navigation checks, responsive shell, and console/page-error capture all passed; the single question input count was 1.
- `npm audit --omit=dev` and full `npm audit` — no reported vulnerabilities after lockfile refresh.
