from __future__ import annotations

import json
import os

import pytest

from app.providers.external import ExternalObservationEngine


class FakeResponse:
    def __init__(self, payload: str | list[str]):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        values = self.payload if isinstance(self.payload, list) else [self.payload]
        return iter(value.encode() for value in values)

    def read(self):
        value = self.payload if isinstance(self.payload, str) else "".join(self.payload)
        return value.encode()


def test_openai_sse_stream_is_converted_to_external_chunks(monkeypatch):
    response = FakeResponse([
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        'data: {"choices":[{"delta":{"content":" world"}}]}\n',
        'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n',
        "data: [DONE]\n",
    ])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("app.providers.external.urllib.request.urlopen", lambda *_args, **_kwargs: response)
    events = []
    ExternalObservationEngine().run("openai", "Say hello", {"max_new_tokens": 4}, events.append)
    assert [e["type"] for e in events] == ["external.started", "external.chunk", "external.chunk", "external.usage", "inference.complete"]
    assert "".join(e["data"]["text"] for e in events if e["type"] == "external.chunk") == "Hello world"
    assert events[-1]["data"]["inspection_mode"] == "limited"
    assert events[-1]["data"]["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


def test_anthropic_sse_stream_is_converted_to_external_chunks(monkeypatch):
    response = FakeResponse([
        'data: {"type":"message_start","message":{}}\n',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Observable"}}\n',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" output"}}\n',
        'data: {"type":"message_delta","usage":{"output_tokens":2}}\n',
    ])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("app.providers.external.urllib.request.urlopen", lambda *_args, **_kwargs: response)
    events = []
    ExternalObservationEngine().run("anthropic", "Say hello", {"max_new_tokens": 4}, events.append)
    assert "Observable output" == "".join(e["data"]["text"] for e in events if e["type"] == "external.chunk")
    assert events[-1]["type"] == "inference.complete"


def test_gemini_stream_is_observation_only(monkeypatch):
    response = FakeResponse([
        'data: {"candidates":[{"content":{"parts":[{"text":"Gemini answer"}]}}]}\n',
        'data: {"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":6}}\n',
    ])
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr("app.providers.external.urllib.request.urlopen", lambda *_args, **_kwargs: response)
    events = []
    ExternalObservationEngine().run("google", "Say hello", {"max_new_tokens": 4}, events.append)
    assert events[1]["type"] == "external.chunk"
    assert events[1]["data"]["text"] == "Gemini answer"
    assert events[-1]["data"]["summary"]["num_output_tokens"] == 2
    assert events[-1]["data"]["summary"]["usage"] == {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}


@pytest.mark.parametrize(
    ("provider", "key"),
    [("openai", ("OPENAI_API_KEY",)), ("anthropic", ("ANTHROPIC_API_KEY",)), ("google", ("GOOGLE_API_KEY", "GEMINI_API_KEY"))],
)
def test_live_external_provider(provider, key):
    if os.getenv("BRAINOS_LIVE_EXTERNAL_TESTS") != "1" or not any(os.getenv(item) for item in key):
        pytest.skip("set BRAINOS_LIVE_EXTERNAL_TESTS=1 and the provider API key to run")
    events = []
    ExternalObservationEngine().run(provider, "Reply with one short sentence.", {"max_new_tokens": 8}, events.append)
    errors = [e for e in events if e["type"] == "system.error"]
    assert not errors, errors
    assert any(e["type"] == "inference.complete" for e in events)
