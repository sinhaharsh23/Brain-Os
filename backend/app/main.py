from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router
from app.config import settings
from app.events.types import Event
from app.state import state

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("brainos")


def _bridge(loop: asyncio.AbstractEventLoop):
    def on_event(ev: Event) -> None:
        asyncio.run_coroutine_threadsafe(state.bus.publish(ev), loop)
        if ev.type == "inference.complete":
            session = state.engine.current_session
            if session is not None:
                state.replay.save_session(session.summary())
                state.replay.append_events(session.session_id, session.events)

    return on_event


async def _load_model_on_startup() -> None:
    state.model_status = "loading"
    try:
        from app.models.qwen import QwenAdapter
        from app.inference.engine import InferenceEngine

        state.adapter = QwenAdapter(model_id=settings.default_model, device=settings.device, dtype=settings.dtype)
        await asyncio.to_thread(state.adapter.load)
        state.engine = InferenceEngine(state.adapter, state.bus, max_prompt_tokens=settings.max_prompt_tokens)
        state.engine._register_hooks()
        state.model_status = "loaded"
        await state.bus.publish(Event("model.ready", {"metadata": state.adapter.metadata.to_dict()}))
    except Exception as e:
        state.model_status = "error"
        log.exception("model load failed")
        await state.bus.publish(Event("system.error", {"stage": "model_load", "message": str(e)}))


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

    async def play(self, session_id: str, speed: float = 1.0) -> None:
        await self.stop()
        self.session_id = session_id
        self.speed = speed
        self.paused = False
        self.events = state.replay.load_events(session_id)
        self.index = 0
        await state.bus.publish(Event("replay.loaded", {"session_id": self.session_id, "count": len(self.events), "index": self.index}))
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        t0 = self.events[self.index]["ts"] if self.events and self.index < len(self.events) else 0.0
        while self.index < len(self.events):
            while self.paused:
                await asyncio.sleep(0.05)
            if self.task is None or self.task.cancelled():
                return
            ev = self.events[self.index]
            delay = max(0.0, (ev["ts"] - t0) / self.speed)
            if delay > 0:
                await asyncio.sleep(delay)
            await self._emit_event(ev)
            self.index += 1
        await state.bus.publish(Event("replay.play", {"session_id": self.session_id, "status": "done"}))

    async def _emit_event(self, raw: dict[str, Any]) -> None:
        await state.bus.publish(Event.from_dict({**raw, "ts": asyncio.get_event_loop().time()}))

    async def pause(self, paused: bool) -> None:
        self.paused = paused
        await state.bus.publish(Event("replay.play", {"session_id": self.session_id, "status": "paused" if paused else "resumed"}))

    async def seek(self, index: int) -> None:
        await self.stop()
        if not self.events and self.session_id:
            self.events = state.replay.load_events(self.session_id)
        self.index = max(0, min(index, max(0, len(self.events) - 1)))
        for raw in self.events[: self.index + 1]:
            await self._emit_event(raw)
        self.paused = True
        await state.bus.publish(Event("replay.seek", {"session_id": self.session_id, "index": self.index, "count": len(self.events)}))

    async def step(self, direction: int = 1) -> None:
        await self.stop()
        if not self.events and self.session_id:
            self.events = state.replay.load_events(self.session_id)
        self.index = max(0, min(self.index + direction, max(0, len(self.events) - 1)))
        if self.events:
            await self._emit_event(self.events[self.index])
        self.paused = True
        await state.bus.publish(Event("replay.seek", {"session_id": self.session_id, "index": self.index, "count": len(self.events)}))

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None


replay = ReplayController()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.wire(settings.replay_dir)
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
    await replay.stop()
    if state.adapter is not None:
        state.adapter.unload()


app = FastAPI(title="BrainOS", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    await state.manager.connect(ws)
    try:
        while True:
            msg = await ws.receive_json()
            action = msg.get("action")
            if action == "run":
                if state.engine is None or not state.adapter.is_loaded:
                    await ws.send_json({"type": "error", "data": {"message": "model not loaded"}})
                    continue
                prompt = str(msg.get("prompt", "")).strip()
                if not prompt:
                    await ws.send_json({"type": "error", "data": {"message": "empty prompt"}})
                    continue
                params = msg.get("params", {})
                params.setdefault("max_new_tokens", settings.max_new_tokens_default)
                await replay.stop()
                asyncio.get_running_loop().run_in_executor(None, lambda: state.engine.run(prompt, params, state._bridge))
                await ws.send_json({"type": "ack", "data": {"action": "run"}})
            elif action == "cancel":
                if state.engine is not None:
                    state.engine.cancel()
            elif action == "pause_inference":
                if state.engine is not None:
                    state.engine.pause()
                    await state.bus.publish(Event("dev.log", {"level": "info", "message": "inference paused between model steps"}))
            elif action == "resume_inference":
                if state.engine is not None:
                    state.engine.resume()
                    await state.bus.publish(Event("dev.log", {"level": "info", "message": "inference resumed"}))
            elif action == "replay":
                await replay.play(str(msg.get("session_id", "")), float(msg.get("speed", 1.0)))
            elif action == "replay_pause":
                await replay.pause(bool(msg.get("paused", True)))
            elif action == "replay_seek":
                await replay.seek(int(msg.get("index", 0)))
            elif action == "replay_step":
                await replay.step(int(msg.get("direction", 1)))
            elif action == "replay_stop":
                await replay.stop()
    except WebSocketDisconnect:
        pass
    finally:
        await state.manager.disconnect(ws)
