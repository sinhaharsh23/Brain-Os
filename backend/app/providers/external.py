from __future__ import annotations

import json
import os
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import settings


class ExternalProviderError(RuntimeError):
    pass


@dataclass
class ExternalResult:
    text: str
    chunks: list[str]
    usage: dict[str, int] = field(default_factory=dict)
    ttft_ms: float | None = None


class ExternalObservationEngine:
    def run(self, provider: str, prompt: str, params: dict[str, Any], on_event: Callable[[dict[str, Any]], None]) -> None:
        session_id = uuid.uuid4().hex[:12]
        model = str(params.get("model") or self._default_model(provider))
        started = time.time()
        emit = lambda event, data: on_event({"type": event, "data": data, "ts": time.time(), "session_id": session_id})
        emit(
            "external.started",
            {
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "inspection_mode": "limited",
                "observable": ["streaming_text", "model", "timing", "token_usage"],
            },
        )
        try:
            if provider == "openai":
                result = self._openai(model, prompt, params)
            elif provider == "anthropic":
                result = self._anthropic(model, prompt, params)
            elif provider == "google":
                result = self._google(model, prompt, params)
            else:
                raise ExternalProviderError(f"unsupported external provider: {provider}")
            for index, chunk in enumerate(result.chunks):
                emit(
                    "external.chunk",
                    {
                        "provider": provider,
                        "model": model,
                        "text": chunk,
                        "index": index,
                    },
                )
            if result.usage:
                emit("external.usage", {"provider": provider, "model": model, "usage": result.usage})
            elapsed = (time.time() - started) * 1000
            timings = {"total_ms": round(elapsed, 2), "ttft_ms": result.ttft_ms}
            if result.usage.get("output_tokens"):
                timings["tokens_per_second"] = round(result.usage["output_tokens"] / max(elapsed / 1000, 1e-6), 2)
            output_tokens = result.usage.get("output_tokens")
            summary = {
                "session_id": session_id,
                "prompt": prompt,
                "model_id": model,
                "provider": provider,
                "inspection_mode": "limited",
                "status": "complete",
                "num_output_tokens": output_tokens,
                "response": result.text,
                "usage": result.usage or None,
                "timings": timings,
            }
            emit(
                "inference.complete",
                {
                    "summary": summary,
                    "response": result.text,
                    "provider": provider,
                    "inspection_mode": "limited",
                    "usage": result.usage or None,
                    "num_output_tokens": output_tokens,
                    "timings": timings,
                },
            )
        except Exception as exc:
            emit("system.error", {"stage": "external_provider", "provider": provider, "model": model, "message": str(exc)})

    def _openai(self, model: str, prompt: str, params: dict[str, Any]) -> ExternalResult:
        key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        if not key:
            raise ExternalProviderError("OPENAI_API_KEY is not configured")
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": float(params.get("temperature", 0.7)),
            "max_tokens": int(params.get("max_new_tokens", 128)),
        }
        return self._sse_request(
            "https://api.openai.com/v1/chat/completions",
            body,
            {"Authorization": f"Bearer {key}"},
            lambda payload: (payload.get("choices") or [{}])[0].get("delta", {}).get("content", ""),
            lambda payload: payload.get("usage") or {},
        )

    def _anthropic(self, model: str, prompt: str, params: dict[str, Any]) -> ExternalResult:
        key = os.getenv("ANTHROPIC_API_KEY") or settings.anthropic_api_key
        if not key:
            raise ExternalProviderError("ANTHROPIC_API_KEY is not configured")
        body = {
            "model": model,
            "max_tokens": int(params.get("max_new_tokens", 128)),
            "temperature": float(params.get("temperature", 0.7)),
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        return self._sse_request(
            "https://api.anthropic.com/v1/messages",
            body,
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
            lambda payload: payload.get("delta", {}).get("text", "") if payload.get("type") == "content_block_delta" else "",
            lambda payload: payload.get("usage") or payload.get("message", {}).get("usage", {}),
        )

    def _google(self, model: str, prompt: str, params: dict[str, Any]) -> ExternalResult:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or settings.google_api_key or settings.gemini_api_key
        if not key:
            raise ExternalProviderError("GOOGLE_API_KEY or GEMINI_API_KEY is not configured")
        query = urllib.parse.urlencode({"alt": "sse", "key": key})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model, safe='')}:streamGenerateContent?{query}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(params.get("temperature", 0.7)),
                "maxOutputTokens": int(params.get("max_new_tokens", 128)),
            },
        }
        return self._sse_request(
            url,
            body,
            {},
            lambda payload: "".join(
                part.get("text", "")
                for candidate in payload.get("candidates", [])
                for part in candidate.get("content", {}).get("parts", [])
            ),
            lambda payload: payload.get("usageMetadata") or {},
        )

    def _sse_request(
        self,
        url: str,
        body: dict[str, Any],
        extra_headers: dict[str, str],
        extract_text: Callable[[dict[str, Any]], str],
        extract_usage: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> ExternalResult:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **extra_headers},
            method="POST",
        )
        chunks: list[str] = []
        usage: dict[str, int] = {}
        first_chunk_at: float | None = None
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    value = line[5:].strip()
                    if value == "[DONE]":
                        break
                    try:
                        payload = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                    text = extract_text(payload)
                    if text:
                        if first_chunk_at is None:
                            first_chunk_at = time.time()
                        chunks.append(text)
                    usage.update(self._normalise_usage(extract_usage(payload)))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ExternalProviderError(f"external API HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise ExternalProviderError(f"external API connection failed: {exc.reason}") from exc
        return ExternalResult(
            text="".join(chunks),
            chunks=chunks,
            usage=usage,
            ttft_ms=round((first_chunk_at - started) * 1000, 2) if first_chunk_at is not None else None,
        )

    @staticmethod
    def _normalise_usage(raw: dict[str, Any]) -> dict[str, int]:
        if not isinstance(raw, dict):
            return {}
        aliases = {
            "prompt_tokens": "input_tokens",
            "input_tokens": "input_tokens",
            "promptTokenCount": "input_tokens",
            "completion_tokens": "output_tokens",
            "output_tokens": "output_tokens",
            "candidatesTokenCount": "output_tokens",
            "total_tokens": "total_tokens",
            "totalTokenCount": "total_tokens",
        }
        normalized: dict[str, int] = {}
        for key, value in raw.items():
            target = aliases.get(key)
            if target is not None and isinstance(value, int):
                normalized[target] = value
        if "total_tokens" not in normalized and {"input_tokens", "output_tokens"} <= normalized.keys():
            normalized["total_tokens"] = normalized["input_tokens"] + normalized["output_tokens"]
        return normalized

    @staticmethod
    def _default_model(provider: str) -> str:
        return {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-latest",
            "google": "gemini-2.0-flash",
        }.get(provider, "")
