from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from app.events.types import Event

log = logging.getLogger("brainos.scheduler")

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


class SchedulerQueueFull(RuntimeError):
    def __init__(self, limit: int) -> None:
        super().__init__(f"inference queue is full (limit {limit})")
        self.limit = limit


@dataclass
class ScheduledRun:
    prompt: str
    params: dict[str, Any]
    owner: Any
    user_id: str
    connection_id: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "QUEUED"
    queue_position: int = 0
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    on_event: Callable[[Event], None] | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "connection_id": self.connection_id,
            "status": self.status,
        }


class InferenceScheduler:
    """Bounded async queue around the shared, serialized model runtime."""

    def __init__(self, engine, bus, *, max_queue_size: int = 4, max_active: int = 1, timeout_s: float = 180.0) -> None:
        self.engine = engine
        self.bus = bus
        self.max_queue_size = max(1, max_queue_size)
        self.max_active = max(1, max_active)
        self.timeout_s = max(0.01, timeout_s)
        self._queue: asyncio.Queue[ScheduledRun] = asyncio.Queue(maxsize=self.max_queue_size)
        self._runs: dict[str, ScheduledRun] = {}
        self._workers: list[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._started = False
        self._closed = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._workers = [asyncio.create_task(self._worker(), name=f"brainos-inference-{i}") for i in range(self.max_active)]

    async def submit(
        self,
        prompt: str,
        params: dict[str, Any],
        *,
        owner: Any,
        user_id: str,
        connection_id: str,
        on_event: Callable[[Event], None],
    ) -> ScheduledRun:
        async with self._lock:
            if self._closed or not self._started:
                raise RuntimeError("inference scheduler is not available")
            if self._queue.full():
                raise SchedulerQueueFull(self.max_queue_size)
            run = ScheduledRun(prompt, params, owner, user_id, connection_id, on_event=on_event)
            run.queue_position = self._queue.qsize() + 1
            self._runs[run.run_id] = run
            self._queue.put_nowait(run)
        self._emit(run, "inference.queued", {"queue_position": run.queue_position, "queue_limit": self.max_queue_size})
        return run

    async def cancel_for_owner(self, owner: Any) -> bool:
        found = False
        async with self._lock:
            for run in self._runs.values():
                if run.owner is owner and run.status not in TERMINAL:
                    run.cancel_requested.set()
                    found = True
                    if run.status == "QUEUED":
                        run.status = "CANCELLED"
                        run.finished_at = time.time()
                        self._emit(run, "inference.cancelled", {"reason": "cancelled while queued"})
                        self._refresh_positions()
                    elif run.status in {"STARTING", "RUNNING", "PAUSED"}:
                        run.status = "CANCELLING"
                        self._emit(run, "inference.status", {"status": run.status})
        if found and self.engine.current_session is not None:
            self.engine.cancel(owner)
        return found

    async def cancel_run(self, run_id: str, owner: Any) -> bool:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.owner is not owner or run.status in TERMINAL:
                return False
            run.cancel_requested.set()
            if run.status == "QUEUED":
                run.status = "CANCELLED"
                run.finished_at = time.time()
                self._emit(run, "inference.cancelled", {"reason": "cancelled while queued"})
                self._refresh_positions()
            elif run.status in {"STARTING", "RUNNING", "PAUSED"}:
                run.status = "CANCELLING"
                self._emit(run, "inference.status", {"status": run.status})
        self.engine.cancel(owner)
        return True

    async def disconnect_owner(self, owner: Any) -> None:
        await self.cancel_for_owner(owner)

    async def shutdown(self) -> None:
        self._closed = True
        async with self._lock:
            queued = list(self._runs.values())
            for run in queued:
                if run.status not in TERMINAL:
                    run.cancel_requested.set()
                    if run.status == "QUEUED":
                        run.status = "CANCELLED"
                        run.finished_at = time.time()
                        self._emit(run, "inference.cancelled", {"reason": "scheduler shutdown"})
                    elif run.status in {"STARTING", "RUNNING", "PAUSED"}:
                        run.status = "CANCELLING"
                        self._emit(run, "inference.status", {"status": run.status})
        if self.engine.current_session is not None:
            self.engine.cancel()
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    def snapshot(self) -> dict[str, Any]:
        active = [r for r in self._runs.values() if r.status in {"STARTING", "RUNNING", "PAUSED", "CANCELLING"}]
        queued = [r for r in self._runs.values() if r.status == "QUEUED"]
        return {
            "queue_size": len(queued),
            "queue_limit": self.max_queue_size,
            "active": len(active),
            "active_limit": self.max_active,
            "runs": len(self._runs),
        }

    async def _worker(self) -> None:
        while True:
            run = await self._queue.get()
            try:
                if run.status == "CANCELLED" or run.cancel_requested.is_set():
                    continue
                await self._execute(run)
            finally:
                self._queue.task_done()
                self._refresh_positions()
                self._prune_runs()

    async def _execute(self, run: ScheduledRun) -> None:
        run.status = "STARTING"
        run.started_at = time.time()
        self._emit(run, "inference.started", {"phase": "scheduler"})
        run.status = "RUNNING"
        self._emit(run, "inference.status", {"status": run.status})

        def on_engine_event(event: Event) -> None:
            if event.type == "inference.complete":
                run.status = "COMPLETED"
                run.finished_at = time.time()
            elif event.type == "inference.cancelled":
                run.status = "TIMED_OUT" if run.status == "TIMED_OUT" else "CANCELLED"
                run.finished_at = time.time()
            elif event.type == "inference.paused":
                run.status = "PAUSED"
            elif event.type == "inference.resumed":
                run.status = "RUNNING"
            event.data = {**run.metadata(), **event.data, "status": run.status}
            if run.on_event:
                run.on_event(event)

        task = asyncio.create_task(asyncio.to_thread(self.engine.run, run.prompt, run.params, on_engine_event, run.owner, run.session_id))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=self.timeout_s)
            if run.status not in TERMINAL:
                run.status = "COMPLETED"
                run.finished_at = time.time()
        except asyncio.TimeoutError:
            run.status = "TIMED_OUT"
            run.finished_at = time.time()
            run.cancel_requested.set()
            self._emit(run, "inference.timeout", {"timeout_s": self.timeout_s})
            self.engine.cancel()
            await asyncio.shield(task)
        except asyncio.CancelledError:
            run.status = "CANCELLED"
            run.finished_at = time.time()
            run.cancel_requested.set()
            self.engine.cancel()
            await asyncio.shield(task)
            raise
        except Exception as exc:
            run.status = "FAILED"
            run.finished_at = time.time()
            log.exception("inference run %s failed", run.run_id)
            self._emit(run, "inference.failed", {"message": str(exc)})

    def _emit(self, run: ScheduledRun, event_type: str, data: dict[str, Any]) -> None:
        if run.on_event:
            run.on_event(Event(event_type, {**run.metadata(), **data}, session_id=run.session_id))

    def _refresh_positions(self) -> None:
        position = 0
        for run in self._runs.values():
            if run.status == "QUEUED":
                position += 1
                run.queue_position = position

    def _prune_runs(self, keep: int = 256) -> None:
        terminal = sorted(
            (run for run in self._runs.values() if run.status in TERMINAL),
            key=lambda run: run.finished_at or run.submitted_at,
        )
        for run in terminal[:-keep]:
            self._runs.pop(run.run_id, None)
