from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


EVENT_TYPES = [
    "system.ready",
    "system.error",
    "model.ready",
    "model.metadata",
    "monitoring.tick",
    "external.started",
    "external.chunk",
    "external.usage",
    "inference.started",
    "inference.queued",
    "inference.starting",
    "inference.status",
    "inference.timeout",
    "inference.failed",
    "inference.paused",
    "inference.resumed",
    "inference.cancelled",
    "tokenization.complete",
    "embeddings.complete",
    "step.started",
    "layer.complete",
    "attention.captured",
    "qkv.captured",
    "mlp.captured",
    "logits.ready",
    "token.selected",
    "token.generated",
    "inference.complete",
    "replay.loaded",
    "replay.play",
    "replay.pause",
    "replay.seek",
    "dev.log",
]

LAYER_EVENT_TYPES = {"layer.complete", "attention.captured", "qkv.captured", "mlp.captured"}


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data, "ts": self.ts, "session_id": self.session_id}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Event":
        return Event(type=d["type"], data=d.get("data", {}), ts=d.get("ts", 0.0), session_id=d.get("session_id"))


def serialize_tensor_summary(values: list[float], limit: int = 32) -> list[float]:
    if len(values) <= limit:
        return values
    step = len(values) / limit
    return [values[int(i * step)] for i in range(limit)]


def tensor_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        "n": n,
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": var**0.5,
        "l2_norm": sum(v * v for v in values) ** 0.5,
    }


def token_view(token: str) -> str:
    return token.replace("Ġ", " ").replace("\n", "\\n").replace("Ċ", "\\n")
