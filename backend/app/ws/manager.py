from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import WebSocket

from app.events.types import Event

log = logging.getLogger("brainos.ws")


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._connection_ids: dict[WebSocket, str] = {}
        # A None owner is an intentionally disconnected sink. Keeping the
        # session key prevents an in-flight run from falling back to a global
        # broadcast while its original socket is being cleaned up.
        self._session_connections: dict[str, WebSocket | None] = {}
        self._lock: asyncio.Lock | None = None

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self.lock:
            self._connections.add(ws)
            self._connection_ids[ws] = uuid.uuid4().hex[:12]

    async def disconnect(self, ws: WebSocket) -> None:
        async with self.lock:
            self._connections.discard(ws)
            self._connection_ids.pop(ws, None)
            for session_id, owner in list(self._session_connections.items()):
                if owner is ws:
                    self._session_connections[session_id] = None

    async def bind_session(self, session_id: str, ws: WebSocket) -> bool:
        async with self.lock:
            current = self._session_connections.get(session_id)
            if current is not None and current is not ws:
                return False
            self._session_connections[session_id] = ws
            return True

    async def unbind_session(self, session_id: str, ws: WebSocket) -> None:
        async with self.lock:
            if self._session_connections.get(session_id) is ws:
                del self._session_connections[session_id]

    async def broadcast(self, event: Event) -> None:
        payload = event.to_dict()
        has_session_owner = bool(event.session_id and event.session_id in self._session_connections)
        owner = self._session_connections.get(event.session_id) if has_session_owner else None
        # A bound owner is normally an accepted WebSocket. Test/replay
        # controllers may bind a lightweight owner that is not transport-backed;
        # do not treat that as a dead socket and erase the ownership lock while
        # dispatching its event.
        recipients = [owner] if has_session_owner and owner in self._connections else ([] if has_session_owner else list(self._connections))
        dead: list[WebSocket] = []
        for ws in recipients:
            if ws is None:
                continue
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self.lock:
                for ws in dead:
                    self._connections.discard(ws)
                    for session_id, owner in list(self._session_connections.items()):
                        if owner is ws:
                            self._session_connections[session_id] = None

    @property
    def count(self) -> int:
        return len(self._connections)

    def connection_id(self, ws: WebSocket) -> str:
        return self._connection_ids.get(ws, f"conn-{id(ws)}")
