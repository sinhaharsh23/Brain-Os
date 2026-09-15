from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class RateLimiter:
    def __init__(self, limit: int = 60, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, identity: str) -> bool:
        now = time.monotonic()
        events = self._events[identity]
        while events and now - events[0] >= self.window_seconds:
            events.popleft()
        if len(events) >= self.limit:
            return False
        events.append(now)
        return True


def valid_token(provided: str | None, expected: str) -> bool:
    return bool(provided) and secrets.compare_digest(provided, expected)


def bearer_token(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


class GenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=200)
    use_chat_template: bool = True
    model: str | None = None


def validate_generation_params(raw: Any, default_max_tokens: int, max_tokens_limit: int) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("params must be an object")
    values = dict(raw)
    values.setdefault("max_new_tokens", default_max_tokens)
    try:
        params = GenerationParams.model_validate(values)
    except ValidationError as exc:
        error = exc.errors()[0]
        field = ".".join(str(part) for part in error.get("loc", ())) or "params"
        raise ValueError(f"{field}: {error['msg']}") from exc
    if params.max_new_tokens > max_tokens_limit:
        raise ValueError(f"max_new_tokens must be <= {max_tokens_limit}")
    return params.model_dump(exclude_none=True)


def validate_replay_speed(value: Any) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("replay speed must be a finite number") from exc
    if not math.isfinite(speed) or speed <= 0.0 or speed > 100.0:
        raise ValueError("replay speed must be finite and greater than 0")
    return speed


def validate_session_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("session_id must contain only letters, numbers, '_' or '-' and be at most 64 characters")
    return value
