from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings
from app.events.types import Event
from app.providers.external import ExternalObservationEngine
from app.providers.registry import provider_registry
from app.providers.base import ChatMessage
from app.inference.scheduler import InferenceScheduler, SchedulerQueueFull
from app.security import RateLimiter, bearer_token, validate_generation_params, validate_replay_speed, validate_session_id, valid_token
from app.state import state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("brainos")


def _bridge(loop: asyncio.AbstractEventLoop, owner: WebSocket | None = None, engine=None):
    async def deliver(ev: Event) -> None:
        if owner is not None and ev.session_id:
            await state.manager.bind_session(ev.session_id, owner)
        await state.bus.publish(ev)
        if owner is not None and ev.session_id and ev.type in {"inference.complete", "inference.cancelled", "inference.failed", "inference.timeout"}:
            await state.manager.unbind_session(ev.session_id, owner)

    def on_event(ev: Event) -> None:
        if state.persistence is not None and ev.data.get("run_id"):
            status_map = {
                "inference.started": "STARTING",
                "inference.status": str(ev.data.get("status", "RUNNING")),
                "inference.paused": "PAUSED",
                "inference.resumed": "RUNNING",
                "inference.complete": "COMPLETED",
                "inference.cancelled": str(ev.data.get("status", "CANCELLED")),
                "inference.timeout": "TIMED_OUT",
                "inference.failed": "FAILED",
            }
            if ev.type in status_map:
                state.persistence.update_run(str(ev.data["run_id"]), status_map[ev.type], str(ev.data.get("message", "")))
        if ev.type in {"inference.complete", "inference.cancelled", "inference.failed"}:
            active_engine = engine or state.engine
            session = active_engine.current_session if active_engine is not None else None
            if session is not None and session.session_id == ev.session_id:
                state.replay.save_session(session.summary())
                state.replay.append_events(session.session_id, session.events)
                state.replay.save_capture(session)
                if state.persistence is not None and ev.data.get("run_id"):
                    owner_id = ev.data.get("user_id")
                    if owner_id == "local":
                        owner_id = None
                    state.persistence.upsert_replay(session.session_id, owner_id, str(state.replay.directory / f"{session.session_id}.pt"), len(session.events))
        asyncio.run_coroutine_threadsafe(deliver(ev), loop)

    return on_event


async def _load_model_on_startup() -> None:
    state.model_status = "loading"
    state.model_error_code = None
    state.model_error = None
    try:
        await unload_current_model()
        from app.inference.engine import InferenceEngine

        adapter = await state.model_manager.load_model(
            model_id=settings.default_model,
            device=settings.device,
            dtype=settings.dtype,
        )
        engine = InferenceEngine(
            adapter,
            state.bus,
            max_prompt_tokens=settings.max_prompt_tokens,
            capture_limit=settings.capture_limit_tokens,
            capture_level=settings.capture_level,
            attention_capture=settings.attention_capture,
        )
        state.adapter = adapter
        state.engine = engine
        scheduler = InferenceScheduler(
            engine,
            state.bus,
            max_queue_size=settings.scheduler_max_queue_size,
            max_active=settings.scheduler_max_active,
            timeout_s=settings.scheduler_run_timeout_s,
        )
        await scheduler.start()
        state.scheduler = scheduler
        state.model_status = "loaded"
        await state.bus.publish(Event("model.ready", {"metadata": adapter.metadata.to_dict()}))
    except Exception as e:
        # ModelManager owns cleanup for both startup and explicit reloads.
        # Keep state aliases empty after a failed load so readiness cannot
        # accidentally report a partially initialized runtime.
        state.adapter = None
        state.engine = None
        state.model_status = "error"
        state.model_error_code = str(getattr(e, "code", "LOCAL_MODEL_ERROR" if "LOCAL_MODEL_" in str(e) else "MODEL_LOAD_FAILED"))
        state.model_error = str(e)
        log.exception("model load failed")
        await state.bus.publish(Event("system.error", {"stage": "model_load", "code": state.model_error_code, "message": str(e)}))


async def unload_current_model() -> None:
    old_scheduler = state.scheduler
    old_engine = state.engine
    old_adapter = state.adapter
    state.scheduler = None
    state.engine = None
    state.adapter = None
    if old_scheduler is not None:
        await old_scheduler.shutdown()
    if old_engine is not None:
        await asyncio.to_thread(old_engine.close)
    managed_adapter = state.model_manager.adapter
    if managed_adapter is not None:
        await state.model_manager.unload()
    if old_adapter is not None and old_adapter is not managed_adapter:
        await asyncio.to_thread(old_adapter.unload)


async def _monitoring_loop() -> None:
    while True:
        try:
            snapshot = state.monitor.snapshot()
            await state.bus.publish(Event("monitoring.tick", snapshot))
        except Exception:
            log.exception("monitoring tick failed")
        await asyncio.sleep(settings.capture_interval_ms / 1000)


class ReplayController:
    def __init__(self) -> None:
        self.task: asyncio.Task | None = None
        self.paused = False
        self.speed = 1.0
        self.session_id: str | None = None
        self.events: list[dict[str, Any]] = []
        self.index = 0
        self.owner: WebSocket | None = None
        self.active = False

    async def play(self, session_id: str, speed: float = 1.0, owner: WebSocket | None = None) -> None:
        speed = validate_replay_speed(speed)
        await self.stop(release=True)
        self.session_id = session_id
        self.speed = speed
        self.owner = owner
        self.active = True
        self.paused = False
        self.events = state.replay.load_events(session_id)
        self.index = 0
        if owner is not None:
            if not await state.manager.bind_session(session_id, owner):
                self.owner = None
                self.active = False
                self.session_id = None
                self.events = []
                raise ValueError("replay session is already active on another connection")
        summary = state.replay.get_session(session_id) if hasattr(state.replay, "get_session") else None
        summary = summary or {}
        await state.bus.publish(Event("replay.loaded", {"session_id": self.session_id, "count": len(self.events), "index": self.index,
            "provider": summary.get("provider", "qwen-local"), "model_id": summary.get("model_id"),
            "inspection_mode": summary.get("inspection_mode", "deep"),
            "model_metadata": summary.get("metadata") if summary.get("provider", "qwen-local") == "qwen-local" else None}, session_id=self.session_id))
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        previous_ts = self.events[self.index]["ts"] if self.events and self.index < len(self.events) else 0.0
        while self.index < len(self.events):
            while self.paused:
                await asyncio.sleep(0.05)
            if self.task is None or self.task.cancelled():
                return
            ev = self.events[self.index]
            delay = max(0.0, (ev["ts"] - previous_ts) / self.speed)
            if delay > 0:
                await asyncio.sleep(delay)
            await self._emit_event(ev)
            previous_ts = ev["ts"]
            self.index += 1
        session_id = self.session_id
        await state.bus.publish(Event("replay.play", {"session_id": session_id, "status": "done"}, session_id=session_id))

    async def _emit_event(self, raw: dict[str, Any]) -> None:
        await state.bus.publish(Event.from_dict({**raw, "ts": asyncio.get_event_loop().time()}))

    async def pause(self, paused: bool) -> None:
        if not self.active:
            return
        self.paused = paused
        await state.bus.publish(Event("replay.play", {"session_id": self.session_id, "status": "paused" if paused else "resumed"}, session_id=self.session_id))

    async def set_speed(self, speed: float) -> None:
        if not self.active:
            return
        self.speed = validate_replay_speed(speed)
        await state.bus.publish(
            Event(
                "replay.play",
                {"session_id": self.session_id, "status": "paused" if self.paused else "playing", "speed": self.speed},
                session_id=self.session_id,
            )
        )

    async def seek(self, index: int) -> None:
        if not self.active:
            return
        await self.stop()
        if not self.events and self.session_id:
            self.events = state.replay.load_events(self.session_id)
        self.index = max(0, min(index, max(0, len(self.events) - 1)))
        for raw in self.events[: self.index + 1]:
            await self._emit_event(raw)
        self.paused = True
        await state.bus.publish(Event("replay.seek", {"session_id": self.session_id, "index": self.index, "count": len(self.events)}, session_id=self.session_id))

    async def step(self, direction: int = 1) -> None:
        if not self.active:
            return
        await self.stop()
        if not self.events and self.session_id:
            self.events = state.replay.load_events(self.session_id)
        self.index = max(0, min(self.index + direction, max(0, len(self.events) - 1)))
        if self.events:
            await self._emit_event(self.events[self.index])
        self.paused = True
        await state.bus.publish(Event("replay.seek", {"session_id": self.session_id, "index": self.index, "count": len(self.events)}, session_id=self.session_id))

    async def stop(self, release: bool = False) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        if release and self.session_id is not None and self.owner is not None:
            await state.manager.unbind_session(self.session_id, self.owner)
            self.owner = None
        if release:
            self.active = False


external_engine = ExternalObservationEngine()
_external_event_buffers: dict[str, list[dict[str, Any]]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.config import BASE_DIR
    env_present = (BASE_DIR / ".env").exists()
    log.info("BrainOS Backend Starting")
    log.info("Device: %s", settings.device)
    log.info("Model: %s", settings.default_model)
    log.info("Environment file: %s", "loaded" if env_present else "default")
    log.info("Model source: Hugging Face/local")
    log.info("API port: %d", settings.port)

    state.wire(
        settings.replay_dir,
        max_sessions=settings.max_replay_sessions,
        max_disk_gb=settings.max_replay_disk_gb,
        enable_full_tensor_cache=settings.enable_full_tensor_cache,
        capture_steps=settings.replay_capture_steps,
        mlp_capture_steps=settings.replay_mlp_capture_steps,
        logit_top_k=settings.replay_logit_top_k,
    )
    state.wire_persistence(settings.database_url, auto_migrate=settings.database_auto_migrate)
    loop = asyncio.get_running_loop()
    state._bridge = _bridge(loop)
    load_task = asyncio.create_task(_load_model_on_startup())
    monitor_task = asyncio.create_task(_monitoring_loop())
    await state.bus.publish(
        Event(
            "system.ready",
            {
                "app": settings.app_name,
                "default_model": settings.default_model,
                "ws_url": f"ws://{settings.host}:{settings.port}/ws",
            },
        )
    )
    yield
    load_task.cancel()
    monitor_task.cancel()
    if state.adapter is not None or state.engine is not None:
        await unload_current_model()


app = FastAPI(title="BrainOS", version="0.1.0", lifespan=lifespan)
request_limiter = RateLimiter(settings.rate_limit_per_minute)


def _request_token(request: Request) -> str | None:
    return bearer_token(request.headers.get("authorization")) or request.headers.get("x-brainos-token")


PUBLIC_AUTH_PATHS = {"/api/health", "/api/ready", "/api/auth/register", "/api/auth/login", "/api/auth/me"}


@app.middleware("http")
async def security_middleware(request, call_next):
    if request.url.path.startswith("/api"):
        provided = _request_token(request)
        if settings.auth_mode == "multi_user" and request.url.path not in PUBLIC_AUTH_PATHS:
            user = state.persistence.user_for_token(provided) if state.persistence is not None else None
            if user is None:
                return JSONResponse({"detail": "authentication required"}, status_code=401)
            request.state.user = user
        elif settings.auth_token and not valid_token(provided, settings.auth_token):
            return JSONResponse({"detail": "authentication required"}, status_code=401)
    identity = request.client.host if request.client else "unknown"
    if request.url.path.startswith("/api") and not request_limiter.allow(identity):
        return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        target = FRONTEND_DIST / full_path
        if full_path and target.is_file():
            return FileResponse(target)
        return FileResponse(FRONTEND_DIST / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    provided = ws.query_params.get("token") or ws.headers.get("x-brainos-token") or bearer_token(ws.headers.get("authorization"))
    user_id = "local"
    if settings.auth_mode == "multi_user":
        user = state.persistence.user_for_token(provided) if state.persistence is not None else None
        if user is None:
            await ws.close(code=1008, reason="authentication required")
            return
        user_id = str(user["id"])
    elif settings.auth_token:
        if not valid_token(provided, settings.auth_token):
            await ws.close(code=1008, reason="authentication required")
            return
    await state.manager.connect(ws)
    replay = ReplayController()
    try:
        while True:
            try:
                msg = await ws.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception:
                await ws.send_json({"type": "error", "data": {"message": "invalid WebSocket message"}})
                continue
            if not isinstance(msg, dict):
                await ws.send_json({"type": "error", "data": {"message": "WebSocket message must be an object"}})
                continue
            action = msg.get("action")
            if action == "run":
                identity = ws.client.host if ws.client else "unknown"
                if not request_limiter.allow(f"ws:{identity}"):
                    await ws.send_json({"type": "error", "data": {"message": "rate limit exceeded"}})
                    continue
                provider = provider_registry.normalize_provider(str(msg.get("provider", "qwen-local")))
                if provider not in {"qwen-local", "openai", "anthropic", "gemini", "ollama"}:
                    await ws.send_json({"type": "error", "data": {"message": f"unsupported provider: {provider}"}})
                    continue
                try:
                    params = validate_generation_params(msg.get("params", {}), settings.max_new_tokens_default, settings.max_new_tokens_limit)
                except ValueError as exc:
                    await ws.send_json({"type": "error", "data": {"message": str(exc)}})
                    continue
                if provider != "qwen-local":
                    selected_model = str(params.get("model") or "")
                    if provider != "ollama" and selected_model and provider_registry.model(provider, selected_model) is None:
                        await ws.send_json({"type": "error", "data": {"message": f"model {selected_model} is not configured for provider {provider}"}})
                        continue
                    raw_prompt = msg.get("prompt", "")
                    if not isinstance(raw_prompt, str):
                        await ws.send_json({"type": "error", "data": {"message": "prompt must be a string"}})
                        continue
                    prompt = raw_prompt.strip()
                    if not prompt:
                        await ws.send_json({"type": "error", "data": {"message": "empty prompt"}})
                        continue
                    if len(prompt) > settings.max_prompt_chars:
                        await ws.send_json({"type": "error", "data": {"message": f"prompt exceeds {settings.max_prompt_chars} characters"}})
                        continue
                    raw_messages = msg.get("messages")
                    messages: list[ChatMessage] = []
                    if isinstance(raw_messages, list):
                        for item in raw_messages[-40:]:
                            if not isinstance(item, dict):
                                continue
                            role = str(item.get("role", ""))
                            content = item.get("content", "")
                            if role in {"system", "user", "assistant"} and isinstance(content, str) and content.strip():
                                messages.append(ChatMessage(role=role, content=content[:settings.max_prompt_chars]))
                    if not messages or messages[-1].role != "user":
                        messages.append(ChatMessage(role="user", content=prompt))
                    if sum(len(message.content) for message in messages) > settings.max_prompt_chars * 4:
                        await ws.send_json({"type": "error", "data": {"message": f"conversation exceeds {settings.max_prompt_chars * 4} characters"}})
                        continue
                    loop = asyncio.get_running_loop()

                    async def deliver_cloud_event(raw):
                        session_id = str(raw.get("session_id", ""))
                        if session_id:
                            await state.manager.bind_session(session_id, ws)
                        await state.bus.publish(Event.from_dict(raw))

                    def callback(raw):
                        session_id = str(raw.get("session_id", ""))
                        if session_id:
                            _external_event_buffers.setdefault(session_id, []).append(raw)
                        # Binding and publishing must be one ordered coroutine;
                        # otherwise a fast provider error can be broadcast
                        # before this websocket is subscribed to its session.
                        asyncio.run_coroutine_threadsafe(deliver_cloud_event(raw), loop)
                        if raw.get("type") == "generation.started":
                            legacy_started = {**raw, "type": "external.started"}
                            asyncio.run_coroutine_threadsafe(deliver_cloud_event(legacy_started), loop)
                        if raw.get("type") in {"generation.error", "inference.failed"}:
                            legacy = {
                                **raw,
                                "type": "system.error",
                                "data": {
                                    **raw.get("data", {}),
                                    "stage": "gguf_runtime" if provider == "ollama" else "external_provider",
                                },
                            }
                            asyncio.run_coroutine_threadsafe(deliver_cloud_event(legacy), loop)
                        if raw.get("type") in {"generation.completed", "generation.cancelled", "generation.error", "inference.complete", "inference.failed"} and session_id:
                            summary = raw.get("data", {}).get("summary") or {
                                "session_id": session_id,
                                "prompt": raw.get("data", {}).get("prompt", prompt),
                                "model_id": raw.get("data", {}).get("model", params.get("model", "")),
                                "provider": provider,
                                "inspection_mode": "limited",
                                "local_or_cloud": "local" if provider == "ollama" else "cloud",
                                "status": "complete" if raw.get("type") in {"generation.completed", "inference.complete"} else "cancelled" if raw.get("type") == "generation.cancelled" else "failed",
                                "created_at": raw.get("ts", 0),
                                "response": raw.get("data", {}).get("response", ""),
                                "usage": raw.get("data", {}).get("usage"),
                                "timings": raw.get("data", {}).get("timings", {}),
                            }
                            state.replay.save_session(summary)
                            state.replay.append_events(session_id, _external_event_buffers.pop(session_id, []))
                            native_session = state.native_sessions.get(session_id)
                            if native_session is not None and raw.get("type") == "inference.complete":
                                state.replay.save_capture(native_session)

                    model_id = str(params.get("model") or "") or None
                    if provider == "ollama":
                        # The Ollama HTTP API returns response text and runtime
                        # counters, but it does not expose the model's forward
                        # tensors. Run the same installed GGUF through the
                        # instrumented llama.cpp adapter so Ollama sessions can
                        # use the existing token, embedding, attention, and
                        # layer inspection views.
                        from app.inference.gguf_runtime import run_native_gguf

                        loop.run_in_executor(
                            None,
                            lambda: run_native_gguf(
                                model_id or settings.ollama_models.split(",")[0].strip(),
                                prompt,
                                params,
                                callback,
                                messages=[{"role": item.role, "content": item.content} for item in messages],
                            ),
                        )
                    else:
                        loop.run_in_executor(None, lambda: external_engine.run_unified(provider, messages, params, callback, model_id=model_id))
                    await ws.send_json({"type": "ack", "data": {"action": "run", "provider": provider}})
                    continue
                if state.engine is None or state.adapter is None or not state.adapter.is_loaded:
                    await ws.send_json({"type": "error", "data": {"message": "model not loaded"}})
                    continue
                raw_prompt = msg.get("prompt", "")
                if not isinstance(raw_prompt, str):
                    await ws.send_json({"type": "error", "data": {"message": "prompt must be a string"}})
                    continue
                prompt = raw_prompt.strip()
                if not prompt:
                    await ws.send_json({"type": "error", "data": {"message": "empty prompt"}})
                    continue
                if len(prompt) > settings.max_prompt_chars:
                    await ws.send_json({"type": "error", "data": {"message": f"prompt exceeds {settings.max_prompt_chars} characters"}})
                    continue
                raw_messages = msg.get("messages")
                if isinstance(raw_messages, list):
                    params["messages"] = [
                        {"role": str(item.get("role")), "content": str(item.get("content"))[:settings.max_prompt_chars]}
                        for item in raw_messages[-40:]
                        if isinstance(item, dict) and item.get("role") in {"system", "user", "assistant"} and isinstance(item.get("content"), str) and item.get("content", "").strip()
                    ]
                project_id = msg.get("project_id")
                if project_id is not None:
                    if user_id == "local" or not isinstance(project_id, str) or state.persistence is None or state.persistence.get_project(user_id, project_id) is None:
                        await ws.send_json({"type": "error", "data": {"message": "project is not owned by this user"}})
                        continue
                await replay.stop(release=True)
                loop = asyncio.get_running_loop()
                engine = state.engine
                scheduler = state.scheduler
                if scheduler is None:
                    await ws.send_json({"type": "error", "data": {"code": "scheduler_unavailable", "message": "inference scheduler unavailable"}})
                    continue
                try:
                    run = await scheduler.submit(
                        prompt,
                        params,
                        owner=ws,
                        user_id=user_id,
                        connection_id=state.manager.connection_id(ws),
                        on_event=_bridge(loop, ws, engine),
                    )
                except SchedulerQueueFull as exc:
                    await ws.send_json({"type": "error", "data": {"code": "queue_full", "message": str(exc), "queue_limit": exc.limit}})
                    continue
                if state.persistence is not None:
                    state.persistence.create_session(run.session_id, None if user_id == "local" else user_id, prompt, project_id)
                    state.persistence.create_run(run.run_id, run.request_id, run.session_id, None if user_id == "local" else user_id, run.connection_id, prompt, engine.adapter.model_id)
                await ws.send_json({"type": "ack", "data": {"action": "run", "run_id": run.run_id, "request_id": run.request_id, "session_id": run.session_id, "queue_position": run.queue_position}})
            elif action == "cancel":
                if external_engine.cancel():
                    continue
                try:
                    from app.inference.gguf_runtime import cancel_native_gguf
                    if cancel_native_gguf():
                        continue
                except ImportError:
                    pass
                if state.scheduler is not None:
                    run_id = msg.get("run_id")
                    cancelled = await state.scheduler.cancel_run(run_id, ws) if isinstance(run_id, str) else await state.scheduler.cancel_for_owner(ws)
                    if not cancelled:
                        await ws.send_json({"type": "error", "data": {"message": "inference is owned by another connection"}})
                else:
                    external_engine.cancel()
            elif action == "pause_inference":
                if state.engine is not None:
                    if not state.engine.pause(ws):
                        await ws.send_json({"type": "error", "data": {"message": "inference is owned by another connection"}})
            elif action == "resume_inference":
                if state.engine is not None:
                    if not state.engine.resume(ws):
                        await ws.send_json({"type": "error", "data": {"message": "inference is owned by another connection"}})
            elif action == "replay":
                session_id = msg.get("session_id")
                try:
                    session_id = validate_session_id(session_id)
                except ValueError as exc:
                    await ws.send_json({"type": "error", "data": {"message": str(exc)}})
                    continue
                try:
                    speed = validate_replay_speed(msg.get("speed", 1.0))
                    if settings.auth_mode == "multi_user":
                        session_payload = state.replay.get_session(session_id)
                        if state.persistence is None or not state.persistence.owns_session(session_id, user_id):
                            if session_payload is None or state.persistence is None or not state.persistence.claim_legacy_session(session_id, user_id, str(session_payload.get("summary", {}).get("prompt", ""))):
                                raise ValueError("session is not owned by this user")
                    await replay.play(session_id, speed, ws)
                except ValueError as exc:
                    await ws.send_json({"type": "error", "data": {"message": str(exc)}})
            elif action == "replay_pause":
                paused = msg.get("paused", True)
                if not isinstance(paused, bool):
                    await ws.send_json({"type": "error", "data": {"message": "paused must be boolean"}})
                    continue
                await replay.pause(paused)
            elif action == "replay_speed":
                try:
                    await replay.set_speed(validate_replay_speed(msg.get("speed", 1.0)))
                except ValueError as exc:
                    await ws.send_json({"type": "error", "data": {"message": str(exc)}})
            elif action == "replay_seek":
                index = msg.get("index", 0)
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    await ws.send_json({"type": "error", "data": {"message": "index must be a non-negative integer"}})
                    continue
                await replay.seek(index)
            elif action == "replay_step":
                direction = msg.get("direction", 1)
                if direction not in {-1, 1}:
                    await ws.send_json({"type": "error", "data": {"message": "direction must be -1 or 1"}})
                    continue
                await replay.step(direction)
            elif action == "replay_stop":
                await replay.stop(release=True)
            else:
                await ws.send_json({"type": "error", "data": {"message": f"unsupported action: {action}"}})
    except WebSocketDisconnect:
        pass
    finally:
        # A disconnected browser must not leave model execution or its event
        # stream attached to a dead connection. The engine remains shared and
        # serialized, so this only cancels work owned by this socket.
        if state.scheduler is not None:
            await state.scheduler.disconnect_owner(ws)
        elif state.engine is not None:
            state.engine.cancel(ws)
        await replay.stop(release=True)
        await state.manager.disconnect(ws)
