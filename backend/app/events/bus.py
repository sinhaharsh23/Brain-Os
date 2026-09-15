from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable, Coroutine

from app.events.types import Event

log = logging.getLogger("brainos.events")


class EventBus:
    def __init__(self, max_history: int = 20000) -> None:
        self._subscribers: list[Callable[[Event], Coroutine[Any, Any, None]]] = []
        self._history: deque[Event] = deque(maxlen=max_history)
        self._lock: asyncio.Lock | None = None

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def subscribe(self, handler: Callable[[Event], Coroutine[Any, Any, None]]) -> None:
        self._subscribers.append(handler)

    def unsubscribe(self, handler: Callable[[Event], Coroutine[Any, Any, None]]) -> None:
        if handler in self._subscribers:
            self._subscribers.remove(handler)

    def history(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._history]

    def snapshot_since(self, since_ts: float) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._history if e.ts >= since_ts]

    async def publish(self, event: Event) -> None:
        async with self.lock:
            self._history.append(event)
        for handler in list(self._subscribers):
            try:
                await handler(event)
            except Exception:
                log.exception("event subscriber failed")

    def publish_sync(self, event: Event) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.publish(event), loop)
        else:
            asyncio.run(self.publish(event))

    def backfill(self, events: list[Event]) -> None:
        for e in events:
            self._history.append(e)
        self._history = deque(list(self._history)[-self._history.maxlen :], maxlen=self._history.maxlen)


bus = EventBus()
