from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import torch

from app.instrumentation.stats import vector_stats

log = logging.getLogger("brainos.replay")


class ReplayStore:
    """Persist replay metadata and bounded visualization-ready artifacts.

    The active inference engine still owns full-resolution tensors in memory.
    The default disk format is compact: representative activation steps,
    float16 tensors, top-k logits, and metric summaries. Full tensor persistence
    is an explicit debugging opt-in.
    """

    _COMPACT_CAPTURE_VERSION = 2
    _ATTENTION_COLUMN_LIMIT = 256
    _DROPPED_TIMELINE_EVENTS = {
        "layer.started",
        "layer.completed",
        "attention.started",
        "attention.completed",
        "mlp.started",
        "mlp.completed",
    }

    def __init__(
        self,
        directory: str,
        *,
        max_sessions: int = 10,
        max_disk_gb: float = 1.0,
        enable_full_tensor_cache: bool = False,
        capture_steps: int = 16,
        mlp_capture_steps: int = 4,
        logit_top_k: int = 50,
    ) -> None:
        self.directory = Path(directory) if directory else Path(".")
        self._configured = bool(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_sessions = max(1, int(max_sessions))
        self.max_disk_bytes = max(1, int(float(max_disk_gb) * 1024**3))
        self.enable_full_tensor_cache = bool(enable_full_tensor_cache)
        self.capture_steps = max(1, int(capture_steps))
        self.mlp_capture_steps = max(1, int(mlp_capture_steps))
        self.logit_top_k = max(1, int(logit_top_k))
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        if self._configured:
            self._clean_incomplete_files()
            self._clean_orphan_captures()
            self._enforce_retention()

    def save_session(self, session: dict[str, Any]) -> str:
        session_id = str(session["session_id"])
        payload = {
            "session_id": session_id,
            "prompt": session.get("prompt", ""),
            "model_id": session.get("model_id", ""),
            "created_at": session.get("created_at", time.time()),
            "summary": session,
        }
        with self._lock:
            self._sessions[session_id] = payload
            try:
                self._write_json_atomic(self.directory / f"{session_id}.json", payload)
            except Exception as exc:
                log.warning("failed to persist session %s: %s", session_id, exc)
            self._enforce_retention()
        return session_id

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
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
                payload = json.loads(path.read_text(encoding="utf-8"))
                # Events are intentionally loaded only by replay/events APIs.
                payload.pop("events", None)
                self._sessions[session_id] = payload
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                log.warning("failed to discover replay session %s: %s", path, exc)

    def has_capture(self, session_id: str) -> bool:
        return (self.directory / f"{session_id}.pt").is_file()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
            path = self.directory / f"{session_id}.json"
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload.pop("events", None)
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                log.warning("failed to read replay session %s: %s", session_id, exc)
                return None
            self._sessions[session_id] = payload
            return payload

    def append_events(self, session_id: str, events: list[dict[str, Any]]) -> None:
        with self._lock:
            payload = self._sessions.get(session_id) or self.get_session(session_id)
            if payload is None:
                return
            disk_payload = dict(payload)
            disk_payload["events"] = self._compact_events(events)
            try:
                self._write_json_atomic(self.directory / f"{session_id}.json", disk_payload)
            except Exception as exc:
                log.warning("failed to persist events for %s: %s", session_id, exc)
            self._enforce_retention()

    def save_capture(self, session) -> None:
        session_id = str(session.session_id)
        capture_path = self.directory / f"{session_id}.pt"
        try:
            self._write_torch_atomic(capture_path, self._capture_payload(session.store))
            with self._lock:
                payload = self._sessions.get(session_id) or self.get_session(session_id)
                if payload is not None:
                    payload["capture_file"] = capture_path.name
                    payload["summary"]["capture_available"] = True
                    disk_payload = dict(payload)
                    metadata_path = self.directory / f"{session_id}.json"
                    try:
                        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                        disk_payload["events"] = existing.get("events", [])
                    except (OSError, json.JSONDecodeError, TypeError):
                        disk_payload["events"] = []
                    self._write_json_atomic(metadata_path, disk_payload)
                self._enforce_retention()
        except Exception as exc:
            log.warning("failed to persist tensor capture for %s: %s", session_id, exc)

    def load_capture(self, session_id: str) -> dict[str, Any] | None:
        path = self.directory / f"{session_id}.pt"
        if not path.exists():
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            log.warning("failed to load tensor capture for %s: %s", session_id, exc)
            return None

    def load_events(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            path = self.directory / f"{session_id}.json"
            if not path.exists():
                return []
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                log.warning("failed to load replay events for %s: %s", session_id, exc)
                return []
            events = payload.get("events", [])
            return events if isinstance(events, list) else []

    def _capture_payload(self, store) -> dict[str, Any]:
        full = self.enable_full_tensor_cache
        steps = None if full else self._selected_steps(store.steps_total, self.capture_steps)
        mlp_steps = None if full else self._selected_steps(store.steps_total, self.mlp_capture_steps)

        attention, attention_column_starts = self._compact_attention(store.attention, steps)
        if full:
            attention = dict(store.attention)
            attention_column_starts = dict(getattr(store, "attention_column_starts", {}))

        if full:
            qkv = dict(store.qkv)
            mlp = dict(store.mlp)
            hidden = dict(store.hidden)
            logits = dict(store.logits)
        else:
            qkv = self._compact_tensor_map(store.qkv, steps)
            # The intermediate projection is derived from gate * up. Keep
            # distinct API-visible projections without persisting that
            # duplicate tensor.
            mlp = self._compact_tensor_map(
                store.mlp,
                mlp_steps,
                allowed_names={"gate_activation", "up", "down_output"},
            )
            hidden = self._compact_tensor_map(store.hidden, steps)
            logits = {}

        logit_stats = dict(getattr(store, "logit_stats", {}))
        logit_vocab_sizes = dict(getattr(store, "logit_vocab_sizes", {}))
        logit_indices: dict[int, torch.Tensor] = {}
        if not full:
            for step, value in store.logits.items():
                if steps is not None and step not in steps:
                    continue
                if not logit_stats.get(step):
                    logit_stats[step] = vector_stats(value)
                logit_vocab_sizes.setdefault(step, int(value.numel()))
                top_k = min(self.logit_top_k, int(value.numel()))
                _, indices = torch.topk(value, top_k)
                logits[step] = value[indices].to(torch.float16).contiguous()
                logit_indices[step] = indices.to(torch.int32).contiguous()

        return {
            "capture_format_version": self._COMPACT_CAPTURE_VERSION,
            "full_tensor_cache": full,
            "prompt_length": store.prompt_length,
            "embeddings": self._disk_tensor(store.embeddings, full),
            "generated_embeddings": [self._disk_tensor(v, full) for v in store.generated_embeddings],
            "pca": dict(store.pca),
            "qkv": qkv,
            "mlp": mlp,
            "hidden": hidden,
            "attention": attention,
            "logits": logits,
            "logit_stats": logit_stats,
            "logit_vocab_sizes": logit_vocab_sizes,
            "logit_indices": logit_indices,
            "logit_candidates": dict(getattr(store, "logit_candidates", {})),
            "residual": dict(getattr(store, "residual", {})),
            "logit_lens": dict(getattr(store, "logit_lens", {})),
            # This map contains sizes/shapes only; raw K/V tensors never enter
            # TensorStore.kv_cache and therefore are not duplicated on disk.
            "kv_cache": dict(getattr(store, "kv_cache", {})),
            "steps_total": store.steps_total,
            "sequence_starts": dict(getattr(store, "sequence_starts", {})),
            "qkv_starts": dict(getattr(store, "qkv_starts", {})),
            "attention_starts": dict(getattr(store, "attention_starts", {})),
            "attention_column_starts": attention_column_starts,
        }

    @staticmethod
    def _disk_tensor(value: Any, full: bool = False) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        if full:
            return value.detach().cpu().contiguous()
        return value.detach().to(torch.float16).cpu().contiguous()

    def _compact_tensor_map(
        self,
        values: dict[Any, Any],
        steps: set[int] | None,
        allowed_names: set[str] | None = None,
    ) -> dict[Any, Any]:
        compact: dict[Any, Any] = {}
        for key, value in values.items():
            step = key[-1] if isinstance(key, tuple) and key and isinstance(key[-1], int) else key
            name = key[1] if isinstance(key, tuple) and len(key) > 1 and isinstance(key[1], str) else None
            if steps is not None and isinstance(step, int) and step not in steps:
                continue
            if allowed_names is not None and name not in allowed_names:
                continue
            compact[key] = self._disk_tensor(value)
        return compact

    def _compact_attention(
        self,
        values: dict[Any, torch.Tensor],
        steps: set[int] | None,
    ) -> tuple[dict[Any, torch.Tensor], dict[int, int]]:
        compact: dict[Any, torch.Tensor] = {}
        starts: dict[int, int] = {}
        for key, value in values.items():
            step = key[-1] if isinstance(key, tuple) and key else key
            if steps is not None and isinstance(step, int) and step not in steps:
                continue
            tensor = self._disk_tensor(value)
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 3 and tensor.shape[-1] > self._ATTENTION_COLUMN_LIMIT:
                start = int(tensor.shape[-1] - self._ATTENTION_COLUMN_LIMIT)
                tensor = tensor[..., start:]
                starts[int(step)] = start
            compact[key] = tensor
        return compact, starts

    @staticmethod
    def _selected_steps(total: int, limit: int) -> set[int]:
        if total <= 0:
            return set()
        if total <= limit:
            return set(range(total))
        positions = torch.linspace(0, total - 1, steps=limit).round().to(torch.int64).tolist()
        return {int(value) for value in positions}

    def _compact_events(self, events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [event for event in events if event.get("type") not in self._DROPPED_TIMELINE_EVENTS]

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._atomic_write(path, encoded.encode("utf-8"))

    def _write_torch_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        temp_path = self._temp_path(path)
        try:
            with open(temp_path, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        temp_path = self._temp_path(path)
        try:
            with open(temp_path, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _temp_path(path: Path) -> Path:
        return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")

    def _clean_incomplete_files(self) -> None:
        for path in self.directory.iterdir():
            if path.is_file() and path.name.startswith(".") and path.name.endswith(".tmp"):
                try:
                    path.unlink()
                    log.info("removed incomplete replay file %s", path.name)
                except OSError as exc:
                    log.warning("failed to remove incomplete replay file %s: %s", path, exc)

    def _clean_orphan_captures(self) -> None:
        for path in self.directory.glob("*.pt"):
            if (self.directory / f"{path.stem}.json").exists():
                continue
            try:
                path.unlink()
                log.info("removed orphan replay capture %s", path.name)
            except OSError as exc:
                log.warning("failed to remove orphan replay capture %s: %s", path, exc)

    def _enforce_retention(self) -> None:
        if not self._configured:
            return
        records: list[tuple[float, str, int]] = []
        total_bytes = 0
        for metadata_path in self.directory.glob("*.json"):
            session_id = metadata_path.stem
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                created_at = float(payload.get("created_at", metadata_path.stat().st_mtime))
                size = metadata_path.stat().st_size
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
            capture_path = self.directory / f"{session_id}.pt"
            if capture_path.exists():
                size += capture_path.stat().st_size
            total_bytes += size
            records.append((created_at, session_id, size))

        records.sort(key=lambda item: (item[0], item[1]))
        while len(records) > self.max_sessions or total_bytes > self.max_disk_bytes:
            _, session_id, size = records.pop(0)
            self._delete_session_files(session_id)
            total_bytes = max(0, total_bytes - size)
            self._sessions.pop(session_id, None)
            log.info(
                "replay retention removed session %s (%d bytes); sessions=%d disk_bytes=%d limits=(%d, %d)",
                session_id,
                size,
                len(records),
                total_bytes,
                self.max_sessions,
                self.max_disk_bytes,
            )

    def _delete_session_files(self, session_id: str) -> None:
        for suffix in (".json", ".pt"):
            path = self.directory / f"{session_id}{suffix}"
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("failed to delete replay file %s: %s", path, exc)
