from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from app.events.types import Event

log = logging.getLogger("brainos.replay")


class ReplayStore:
    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, dict[str, Any]] = {}

    def save_session(self, session: dict[str, Any]) -> str:
        session_id = session["session_id"]
        payload = {
            "session_id": session_id,
            "prompt": session.get("prompt", ""),
            "model_id": session.get("model_id", ""),
            "created_at": session.get("created_at", time.time()),
            "summary": session,
        }
        self._sessions[session_id] = payload
        path = self.directory / f"{session_id}.json"
        try:
            path.write_text(json.dumps(payload, indent=1))
        except Exception as e:
            log.warning("failed to persist session %s: %s", session_id, e)
        return session_id

    def list_sessions(self) -> list[dict[str, Any]]:
        self._discover_disk_sessions()
        sessions = [s["summary"] for s in self._sessions.values()]
        sessions.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        return sessions

    def _discover_disk_sessions(self) -> None:
        for path in self.directory.glob("*.json"):
            session_id = path.stem
            if session_id in self._sessions:
                continue
            try:
                self._sessions[session_id] = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                log.warning("failed to discover replay session %s: %s", path, e)

    def has_capture(self, session_id: str) -> bool:
        return (self.directory / f"{session_id}.pt").exists()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        if session_id in self._sessions:
            return self._sessions[session_id]
        path = self.directory / f"{session_id}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            self._sessions[session_id] = payload
            return payload
        return None

    def append_events(self, session_id: str, events: list[dict[str, Any]]) -> None:
        payload = self._sessions.get(session_id)
        if payload is None:
            return
        payload["events"] = events
        path = self.directory / f"{session_id}.json"
        try:
            path.write_text(json.dumps(payload, indent=1))
        except Exception as e:
            log.warning("failed to persist events for %s: %s", session_id, e)

    def save_capture(self, session) -> None:
        capture = {
            "prompt_length": session.store.prompt_length,
            "embeddings": session.store.embeddings,
            "generated_embeddings": session.store.generated_embeddings,
            "pca": session.store.pca,
            "qkv": session.store.qkv,
            "mlp": session.store.mlp,
            "hidden": session.store.hidden,
            "attention": session.store.attention,
            "logits": session.store.logits,
            "logit_candidates": getattr(session.store, "logit_candidates", {}),
            "steps_total": session.store.steps_total,
        }
        path = self.directory / f"{session.session_id}.pt"
        try:
            torch.save(capture, path)
            payload = self._sessions.get(session.session_id)
            if payload is not None:
                payload["capture_file"] = path.name
                payload["summary"]["capture_available"] = True
                (self.directory / f"{session.session_id}.json").write_text(json.dumps(payload, indent=1))
        except Exception as e:
            log.warning("failed to persist tensor capture for %s: %s", session.session_id, e)

    def load_capture(self, session_id: str) -> dict[str, Any] | None:
        path = self.directory / f"{session_id}.pt"
        if not path.exists():
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            log.warning("failed to load tensor capture for %s: %s", session_id, e)
            return None

    def load_events(self, session_id: str) -> list[dict[str, Any]]:
        payload = self.get_session(session_id)
        if payload is None:
            return []
        events = payload.get("events")
        if events is None:
            path = self.directory / f"{session_id}.json"
            if path.exists():
                payload = json.loads(path.read_text())
                events = payload.get("events", [])
        return events or []
