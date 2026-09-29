from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import settings
from app.providers.base import ChatMessage

try:  # Optional at source checkout; production installs the official SDKs.
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - exercised when optional SDK is absent
    OpenAI = None

try:
    from anthropic import Anthropic  # type: ignore
except ImportError:  # pragma: no cover
    Anthropic = None

try:
    from google import genai  # type: ignore
except ImportError:  # pragma: no cover
    genai = None


class ExternalProviderError(RuntimeError):
    pass


@dataclass
class ExternalResult:
    text: str
    chunks: list[str]
    usage: dict[str, int] = field(default_factory=dict)
    ttft_ms: float | None = None
    response_id: str | None = None
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ExternalObservationEngine:
    def __init__(self) -> None:
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

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

    def run_unified(
        self,
        provider: str,
        messages: list[ChatMessage],
        params: dict[str, Any],
        on_event: Callable[[dict[str, Any]], None],
        *,
        cancel_event: threading.Event | None = None,
        model_id: str | None = None,
    ) -> str:
        """Stream a cloud request using the provider-neutral event protocol."""
        provider = {"google": "gemini"}.get(provider, provider)
        session_id = uuid.uuid4().hex[:12]
        cancel_event = cancel_event or threading.Event()
        with self._lock:
            self._cancel_events[session_id] = cancel_event
        model = str(model_id or params.get("model") or self._default_model(provider))
        prompt = next((message.content for message in reversed(messages) if message.role == "user"), "")
        started = time.time()
        response_id: str | None = None
        try:
            self._unified_emit(on_event, "generation.started", session_id, {
                "provider": provider, "model": model, "prompt": prompt,
                "inspection_mode": "limited", "local_or_cloud": "local" if provider == "ollama" else "cloud",
                "observable": ["request", "streaming_text", "response_id", "usage", "timing", "stop_reason"],
            })
            self._unified_emit(on_event, "provider.started", session_id, {"provider": provider, "model": model})
            self._unified_emit(on_event, "input.prepared", session_id, {"message_count": len(messages), "exact_tokenization": False})
            self._unified_emit(on_event, "provider.connected", session_id, {"provider": provider, "model": model})
            if provider == "openai":
                result = self._openai_unified(model, messages, params, cancel_event, on_event, session_id)
            elif provider == "anthropic":
                result = self._anthropic_unified(model, messages, params, cancel_event, on_event, session_id)
            elif provider == "gemini":
                result = self._gemini_unified(model, messages, params, cancel_event, on_event, session_id)
            elif provider == "ollama":
                result = self._ollama_unified(model, messages, params, cancel_event, on_event, session_id)
            else:
                raise ExternalProviderError(f"unsupported external provider: {provider}")
            response_id = result.response_id
            if cancel_event.is_set():
                self._unified_emit(on_event, "generation.cancelled", session_id, {"provider": provider, "model": model, "response": result.text})
                return session_id
            if result.usage:
                self._unified_emit(on_event, "provider.usage", session_id, {"provider": provider, "model": model, "usage": result.usage})
            elapsed = (time.time() - started) * 1000
            timings = {"total_ms": round(elapsed, 2), "ttft_ms": result.ttft_ms}
            if result.usage.get("output_tokens"):
                timings["tokens_per_second"] = round(result.usage["output_tokens"] / max(elapsed / 1000, 1e-6), 2)
            normalized = result.metadata.get("telemetry")
            if normalized:
                timings.update(normalized["timing"])
                timings["tokens_per_second"] = normalized["performance"]["end_to_end_tokens_per_second"]
                timings["decode_tokens_per_second"] = normalized["performance"]["decode_tokens_per_second"]
            summary = {
                "telemetry": normalized,
                "session_id": session_id, "prompt": prompt, "model_id": model,
                "provider": provider, "inspection_mode": "limited", "local_or_cloud": "local" if provider == "ollama" else "cloud",
                "status": "complete", "num_output_tokens": result.usage.get("output_tokens"),
                "response": result.text, "usage": result.usage or None, "timings": timings,
                "response_id": response_id, "stop_reason": result.stop_reason,
                "metadata": result.metadata,
            }
            self._unified_emit(on_event, "response.text.completed", session_id, {"provider": provider, "model": model, "response": result.text})
            self._unified_emit(on_event, "generation.completed", session_id, {
                "summary": summary, "response": result.text, "provider": provider,
                "model": model, "inspection_mode": "limited", "usage": result.usage or None,
                "timings": timings, "response_id": response_id, "stop_reason": result.stop_reason,
                "metadata": result.metadata, "telemetry": normalized,
            })
        except Exception as exc:
            self._unified_emit(on_event, "generation.error", session_id, {
                "stage": "external_provider", "provider": provider, "model": model,
                "message": self._sanitize_error(str(exc)),
            })
        finally:
            with self._lock:
                self._cancel_events.pop(session_id, None)
        return session_id

    def _ollama_unified(self, model, messages, params, cancel_event, on_event, session_id) -> ExternalResult:
        from app.providers.ollama import model_info
        from app.providers.registry import LIMITED
        from app.providers.telemetry import telemetry
        url = str(os.getenv("OLLAMA_URL") or os.getenv("BRAINOS_OLLAMA_URL") or settings.ollama_url).rstrip("/")
        info = model_info(model)
        if info["status"] != "READY":
            raise ExternalProviderError(f"Ollama {info['status']}: {info.get('error', model)}")
        options = {"num_predict": int(params.get("max_new_tokens", settings.max_new_tokens_default))}
        if info.get("context_window") is not None and options["num_predict"] >= info["context_window"]:
            raise ExternalProviderError("Requested output leaves no room for input in Ollama's active context. Reduce Max Output Tokens.")
        for key in ("temperature", "top_p", "top_k", "seed", "repeat_penalty", "num_ctx"):
            if key in params:
                options[key] = params[key]
        if "repetition_penalty" in params:
            options["repeat_penalty"] = params["repetition_penalty"]
        if "num_ctx" in options:
            info["context_window"] = options["num_ctx"]
        caps = LIMITED.to_dict()
        sampling = {**options, "max_output_tokens": options["num_predict"], "source": "ollama_request.options"}
        snapshot = telemetry("ollama", model, model=info, capabilities=caps, sampling=sampling,
                             sources={"model": "ollama_api", "tokens": "ollama_api", "timing": "ollama_api / brainos_timer"})
        self._unified_emit(on_event, "telemetry.updated", session_id, snapshot)
        request_body = {"model": model, "messages": [{"role": item.role, "content": item.content} for item in messages],
                        "stream": True, "options": options, "logprobs": True, "top_logprobs": 5}
        request = urllib.request.Request(f"{url}/api/chat", data=json.dumps(request_body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        chunks, logprobs = [], []
        usage, final = {}, {}
        first = None
        # TTFT and end-to-end request latency start when BrainOS submits the
        # generation request. /api/show and /api/ps above are metadata checks,
        # not part of model input prefill or streamed generation.
        request_started_at = time.perf_counter()
        self._unified_emit(on_event, "provider.request_submitted", session_id, {
            "provider": "ollama", "model": model, "options": options,
        })
        try:
            response = urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError as exc:
            # Older Ollama versions can reject the optional logprobs fields.
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 400 and "logprob" in detail.lower():
                request_body.pop("logprobs"); request_body.pop("top_logprobs")
                request.data = json.dumps(request_body).encode()
                response = urllib.request.urlopen(request, timeout=180)
            else:
                raise ExternalProviderError(f"Ollama {'MODEL NOT FOUND' if exc.code == 404 else 'ERROR'}: {detail}") from exc
        with response:
            for raw_line in response:
                if cancel_event.is_set():
                    break
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line)
                if payload.get("error"):
                    raise ExternalProviderError(str(payload["error"]))
                text = str(payload.get("message", {}).get("content", ""))
                if text:
                    if first is None:
                        first = time.perf_counter()
                        self._unified_emit(on_event, "provider.first_content", session_id, {"provider": "ollama", "ttft_ms": (first - request_started_at) * 1000})
                    chunks.append(text)
                    self._unified_emit(on_event, "response.text.delta", session_id, {"provider": "ollama", "model": model, "text": text, "stream_kind": "text_delta"})
                actual_logprobs = payload.get("logprobs") or []
                if actual_logprobs:
                    logprobs.extend(actual_logprobs)
                    self._unified_emit(on_event, "provider.logprobs", session_id, {"provider": "ollama", "tokens": actual_logprobs})
                if payload.get("done"):
                    final = payload
                    break
        if not final and not cancel_event.is_set():
            raise ExternalProviderError("Ollama stream ended before final metrics")
        generation_finished_at = time.perf_counter()
        for source, target in (("prompt_eval_count", "input_tokens"), ("eval_count", "output_tokens"), ("prompt_eval_cached_count", "cached_prompt_tokens")):
            if final.get(source) is not None:
                usage[target] = int(final[source])
        if 'input_tokens' in usage and 'output_tokens' in usage:
            usage['total_tokens'] = usage['input_tokens'] + usage['output_tokens']
        timing = {
            "total_ms": (generation_finished_at - request_started_at) * 1000,
            "ttft_ms": (first - request_started_at) * 1000 if first is not None else None,
        }
        for source, target in (("total_duration", "provider_total_ms"), ("load_duration", "load_ms"), ("prompt_eval_duration", "prompt_eval_ms"), ("eval_duration", "decode_ms")):
            timing[target] = final[source] / 1e6 if final.get(source) is not None else None
        caps['logprobs'] = caps['probabilities'] = bool(logprobs)
        refreshed = model_info(model)
        if refreshed.get('status') == 'READY':
            info.update(refreshed)
        snapshot = telemetry("ollama", model, model=info, capabilities=caps, sampling=sampling,
            tokens={"model_input_tokens": usage.get('input_tokens'), "generated_tokens": usage.get('output_tokens'), "cached_prompt_tokens": usage.get('cached_prompt_tokens')},
            timing=timing, sources={"model": "ollama_api", "tokens": "ollama_api", "timing": "ollama_api", "ttft_ms": "brainos_timer", "total_ms": "brainos_timer", "sampling": "ollama_request.options"})
        self._unified_emit(on_event, "provider.metrics_finalized", session_id, {
            "provider": "ollama", "model": model, "metrics": snapshot["timing"], "tokens": snapshot["tokens"],
        })
        self._unified_emit(on_event, "telemetry.updated", session_id, snapshot)
        return ExternalResult(''.join(chunks), chunks, usage, timing['ttft_ms'], stop_reason=final.get('done_reason'),
            metadata={"telemetry": snapshot, "logprobs": logprobs, "provider_metrics": {key: final.get(key) for key in ('prompt_eval_count', 'eval_count', 'prompt_eval_cached_count', 'total_duration', 'eval_duration', 'load_duration', 'prompt_eval_duration')}})

    def cancel(self, session_id: str | None = None) -> bool:
        with self._lock:
            if session_id is not None and session_id in self._cancel_events:
                self._cancel_events[session_id].set()
                return True
            if session_id is None and self._cancel_events:
                next(iter(self._cancel_events.values())).set()
                return True
        return False

    @staticmethod
    def _unified_emit(on_event: Callable[[dict[str, Any]], None], event_type: str, session_id: str, data: dict[str, Any]) -> None:
        on_event({"type": event_type, "data": data, "ts": time.time(), "session_id": session_id})

    @staticmethod
    def _sanitize_error(message: str) -> str:
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
            secret = os.getenv(key) or getattr(settings, key.lower(), "")
            if secret:
                message = message.replace(secret, "[redacted]")
        return message[:1000]

    def _openai_unified(self, model, messages, params, cancel_event, on_event, session_id) -> ExternalResult:
        key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        if not key:
            raise ExternalProviderError("OPENAI_API_KEY is not configured")
        if OpenAI is not None:
            client = OpenAI(api_key=key)
            input_items = [{"role": m.role, "content": m.content} for m in messages if m.role in {"system", "user", "assistant"}]
            request = {"model": model, "input": input_items, "stream": True, "store": False}
            if "temperature" in params:
                request["temperature"] = float(params["temperature"])
            request["max_output_tokens"] = int(params.get("max_new_tokens", 800))
            stream = client.responses.create(**request)
            return self._consume_openai_sdk_stream(stream, cancel_event, on_event, session_id, model)
        return self._openai_raw_unified(model, messages, params, cancel_event, on_event, session_id)

    def _consume_openai_sdk_stream(self, stream, cancel_event, on_event, session_id, model) -> ExternalResult:
        chunks: list[str] = []
        usage: dict[str, int] = {}
        response_id = None
        stop_reason = None
        started = time.time()
        first = None
        for event in stream:
            if cancel_event.is_set():
                break
            event_type = getattr(event, "type", None) or (event.get("type") if isinstance(event, dict) else "")
            payload = event if isinstance(event, dict) else getattr(event, "model_dump", lambda: {})()
            if event_type == "response.created":
                response = payload.get("response", {}) if isinstance(payload, dict) else {}
                response_id = response.get("id")
                self._unified_emit(on_event, "response.created", session_id, {"provider": "openai", "model": model, "response_id": response_id})
            elif event_type == "response.output_text.delta":
                delta = payload.get("delta", "") if isinstance(payload, dict) else getattr(event, "delta", "")
                if delta:
                    if first is None: first = time.time()
                    chunks.append(delta)
                    self._unified_emit(on_event, "response.text.delta", session_id, {"provider": "openai", "model": model, "text": delta, "stream_kind": "text_delta"})
            elif event_type == "response.completed":
                response = payload.get("response", {}) if isinstance(payload, dict) else {}
                response_id = response.get("id") or response_id
                usage = self._normalise_usage(response.get("usage", {}))
                stop_reason = response.get("incomplete_details", {}).get("reason") if isinstance(response.get("incomplete_details"), dict) else None
                self._unified_emit(on_event, "provider.completed", session_id, {"provider": "openai", "model": model, "response_id": response_id, "stop_reason": stop_reason})
        return ExternalResult("".join(chunks), chunks, usage, round((first - started) * 1000, 2) if first else None, response_id, stop_reason)

    def _openai_raw_unified(self, model, messages, params, cancel_event, on_event, session_id) -> ExternalResult:
        metadata: dict[str, Any] = {}
        def on_payload(payload: dict[str, Any]) -> None:
            response = payload.get("response", {})
            if payload.get("type") == "response.created":
                metadata["response_id"] = response.get("id")
                self._unified_emit(on_event, "response.created", session_id, {"provider": "openai", "model": model, "response_id": metadata.get("response_id")})
            elif payload.get("type") == "response.completed":
                metadata["response_id"] = response.get("id") or metadata.get("response_id")
                incomplete = response.get("incomplete_details") or {}
                metadata["stop_reason"] = incomplete.get("reason") if isinstance(incomplete, dict) else None
                self._unified_emit(on_event, "provider.completed", session_id, {"provider": "openai", "model": model, "response_id": metadata.get("response_id"), "stop_reason": metadata.get("stop_reason")})

        result = self._sse_request(
            "https://api.openai.com/v1/responses",
            {"model": model, "input": [{"role": m.role, "content": m.content} for m in messages], "stream": True, "store": False, "max_output_tokens": int(params.get("max_new_tokens", 800)), "temperature": float(params.get("temperature", 0.7))},
            {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY') or settings.openai_api_key}"},
            lambda payload: payload.get("delta", "") if payload.get("type") == "response.output_text.delta" else "",
            lambda payload: payload.get("response", {}).get("usage", {}) if payload.get("type") == "response.completed" else {},
            on_chunk=lambda text: self._unified_emit(on_event, "response.text.delta", session_id, {"provider": "openai", "model": model, "text": text, "stream_kind": "text_delta"}),
            on_payload=on_payload,
            cancel_event=cancel_event,
        )
        result.response_id = metadata.get("response_id")
        result.stop_reason = metadata.get("stop_reason")
        return result

    def _openai(self, model: str, prompt: str, params: dict[str, Any]) -> ExternalResult:
        key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        if not key:
            raise ExternalProviderError("OPENAI_API_KEY is not configured")
        body = {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "stream": True,
            "store": False,
            "temperature": float(params.get("temperature", 0.7)),
            "max_output_tokens": int(params.get("max_new_tokens", 800)),
        }
        return self._sse_request(
            "https://api.openai.com/v1/responses",
            body,
            {"Authorization": f"Bearer {key}"},
            lambda payload: payload.get("delta", "") or (payload.get("choices") or [{}])[0].get("delta", {}).get("content", ""),
            lambda payload: payload.get("usage") or {},
        )

    def _anthropic(self, model: str, prompt: str, params: dict[str, Any]) -> ExternalResult:
        key = os.getenv("ANTHROPIC_API_KEY") or settings.anthropic_api_key
        if not key:
            raise ExternalProviderError("ANTHROPIC_API_KEY is not configured")
        body = {
            "model": model,
            "max_tokens": int(params.get("max_new_tokens", 800)),
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

    def _anthropic_unified(self, model, messages, params, cancel_event, on_event, session_id) -> ExternalResult:
        key = os.getenv("ANTHROPIC_API_KEY") or settings.anthropic_api_key
        if not key:
            raise ExternalProviderError("ANTHROPIC_API_KEY is not configured")
        if Anthropic is not None:
            client = Anthropic(api_key=key)
            system = "\n".join(m.content for m in messages if m.role == "system") or None
            request = {"model": model, "max_tokens": int(params.get("max_new_tokens", 800)), "messages": [{"role": m.role, "content": m.content} for m in messages if m.role in {"user", "assistant"}], "stream": True}
            if system:
                request["system"] = system
            if "temperature" in params:
                request["temperature"] = float(params["temperature"])
            stream = client.messages.create(**request)
            return self._consume_anthropic_sdk_stream(stream, cancel_event, on_event, session_id, model)
        return self._sse_request(
            "https://api.anthropic.com/v1/messages",
            {"model": model, "max_tokens": int(params.get("max_new_tokens", 800)), "temperature": float(params.get("temperature", 0.7)), "messages": [{"role": m.role, "content": m.content} for m in messages if m.role in {"user", "assistant"}], "stream": True},
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
            lambda payload: payload.get("delta", {}).get("text", "") if payload.get("type") == "content_block_delta" else "",
            lambda payload: payload.get("usage") or payload.get("message", {}).get("usage", {}),
            on_chunk=lambda text: self._unified_emit(on_event, "response.text.delta", session_id, {"provider": "anthropic", "model": model, "text": text, "stream_kind": "text_delta"}),
            cancel_event=cancel_event,
        )

    def _consume_anthropic_sdk_stream(self, stream, cancel_event, on_event, session_id, model) -> ExternalResult:
        chunks: list[str] = []
        usage: dict[str, int] = {}
        started = time.time()
        first = None
        response_id = None
        stop_reason = None
        for event in stream:
            if cancel_event.is_set(): break
            event_type = getattr(event, "type", None)
            if event_type == "message_start":
                message = getattr(event, "message", None)
                response_id = getattr(message, "id", None)
                usage.update(self._normalise_usage(getattr(message, "usage", {}) or {}))
                self._unified_emit(on_event, "response.created", session_id, {"provider": "anthropic", "model": model, "response_id": response_id})
            elif event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                text = getattr(delta, "text", "") if delta is not None else ""
                if text:
                    if first is None: first = time.time()
                    chunks.append(text)
                    self._unified_emit(on_event, "response.text.delta", session_id, {"provider": "anthropic", "model": model, "text": text, "stream_kind": "text_delta"})
            elif event_type == "message_delta":
                delta = getattr(event, "delta", None)
                stop_reason = getattr(delta, "stop_reason", None) if delta is not None else None
                usage.update(self._normalise_usage(getattr(event, "usage", {}) or {}))
        return ExternalResult("".join(chunks), chunks, usage, round((first - started) * 1000, 2) if first else None, response_id, stop_reason)

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
                "maxOutputTokens": int(params.get("max_new_tokens", 800)),
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

    def _gemini_unified(self, model, messages, params, cancel_event, on_event, session_id) -> ExternalResult:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or settings.google_api_key or settings.gemini_api_key
        if not key:
            raise ExternalProviderError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured")
        if genai is not None:
            client = genai.Client(api_key=key)
            contents = [{"role": "user" if m.role == "user" else "model", "parts": [{"text": m.content}]} for m in messages if m.role in {"user", "assistant"}]
            config = {"temperature": float(params.get("temperature", 0.7)), "max_output_tokens": int(params.get("max_new_tokens", 800))}
            stream = client.models.generate_content_stream(model=model, contents=contents, config=config)
            chunks: list[str] = []
            usage: dict[str, int] = {}
            started = time.time(); first = None
            self._unified_emit(on_event, "response.created", session_id, {"provider": "gemini", "model": model})
            for chunk in stream:
                if cancel_event.is_set(): break
                text = getattr(chunk, "text", "") or ""
                if text:
                    if first is None: first = time.time()
                    chunks.append(text)
                    self._unified_emit(on_event, "response.text.delta", session_id, {"provider": "gemini", "model": model, "text": text, "stream_kind": "text_delta"})
                metadata = getattr(chunk, "usage_metadata", None)
                if metadata is not None:
                    usage.update(self._normalise_usage({k: getattr(metadata, k) for k in ("prompt_token_count", "candidates_token_count", "total_token_count") if hasattr(metadata, k)}))
            return ExternalResult("".join(chunks), chunks, usage, round((first - started) * 1000, 2) if first else None)
        query = urllib.parse.urlencode({"alt": "sse", "key": key})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model, safe='')}:streamGenerateContent?{query}"
        return self._sse_request(
            url,
            {"contents": [{"role": "user", "parts": [{"text": m.content}]} for m in messages if m.role in {"user", "assistant"}], "generationConfig": {"temperature": float(params.get("temperature", 0.7)), "maxOutputTokens": int(params.get("max_new_tokens", 800))}},
            {},
            lambda payload: "".join(part.get("text", "") for candidate in payload.get("candidates", []) for part in candidate.get("content", {}).get("parts", [])),
            lambda payload: payload.get("usageMetadata") or {},
            on_chunk=lambda text: self._unified_emit(on_event, "response.text.delta", session_id, {"provider": "gemini", "model": model, "text": text, "stream_kind": "text_delta"}),
            cancel_event=cancel_event,
        )

    def _sse_request(
        self,
        url: str,
        body: dict[str, Any],
        extra_headers: dict[str, str],
        extract_text: Callable[[dict[str, Any]], str],
        extract_usage: Callable[[dict[str, Any]], dict[str, Any]],
        on_chunk: Callable[[str], None] | None = None,
        on_payload: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
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
                    if cancel_event is not None and cancel_event.is_set():
                        break
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
                    if on_payload is not None:
                        on_payload(payload)
                    text = extract_text(payload)
                    if text:
                        if first_chunk_at is None:
                            first_chunk_at = time.time()
                        chunks.append(text)
                        if on_chunk is not None:
                            on_chunk(text)
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
            "prompt_token_count": "input_tokens",
            "completion_tokens": "output_tokens",
            "output_tokens": "output_tokens",
            "candidatesTokenCount": "output_tokens",
            "candidates_token_count": "output_tokens",
            "total_tokens": "total_tokens",
            "totalTokenCount": "total_tokens",
            "total_token_count": "total_tokens",
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
            "gemini": "gemini-2.0-flash",
        }.get(provider, "")
