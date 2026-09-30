from __future__ import annotations

from app.providers.base import ChatMessage
from app.providers.external import ExternalObservationEngine
from app.providers.registry import provider_registry
from app.providers.router import AIProviderRouter


class FakeResponse:
    def __init__(self, lines: list[str]):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(line.encode() for line in self.lines)


def test_registry_has_local_and_cloud_descriptors_without_secrets():
    providers = {item.provider_id: item for item in provider_registry.list()}
    assert set(providers) == {"qwen-local", "openai", "anthropic", "gemini", "ollama"}
    assert providers["qwen-local"].capabilities.hidden_states is True
    assert providers["openai"].capabilities.hidden_states is False
    assert providers["openai"].capabilities.api_usage is True
    assert providers["ollama"].inspection_mode == "deep"
    assert providers["ollama"].capabilities.hidden_states is True
    assert providers["ollama"].capabilities.qkv is True
    assert providers["ollama"].capabilities.attention is True
    assert providers["ollama"].capabilities.embeddings is True
    assert providers["ollama"].capabilities.token_ids is True
    assert providers["ollama"].capabilities.kv_cache is False
    assert "test-key" not in str(providers["openai"].to_dict()).lower()


def test_router_normalizes_google_alias_and_does_not_execute_local_directly():
    router = AIProviderRouter()
    assert router.get_provider_info("google").provider_id == "gemini"
    assert router.get_provider_info("qwen-local").provider_id == "qwen-local"


def test_openai_responses_protocol_is_normalized(monkeypatch):
    monkeypatch.setattr("app.providers.external.OpenAI", None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    response = FakeResponse([
        'data: {"type":"response.created","response":{"id":"resp_test"}}\n',
        'data: {"type":"response.output_text.delta","delta":"Hello"}\n',
        'data: {"type":"response.output_text.delta","delta":" world"}\n',
        'data: {"type":"response.completed","response":{"id":"resp_test","usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}\n',
    ])
    monkeypatch.setattr("app.providers.external.urllib.request.urlopen", lambda *_args, **_kwargs: response)
    events: list[dict] = []
    ExternalObservationEngine().run_unified("openai", [ChatMessage("user", "Say hello")], {"max_new_tokens": 4}, events.append)
    types = [event["type"] for event in events]
    assert types[:4] == ["generation.started", "provider.started", "input.prepared", "provider.connected"]
    assert "response.created" in types
    assert [event["data"]["text"] for event in events if event["type"] == "response.text.delta"] == ["Hello", " world"]
    complete = next(event for event in events if event["type"] == "generation.completed")
    assert complete["data"]["response_id"] == "resp_test"
    assert complete["data"]["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


def test_missing_cloud_key_is_reported_without_leaking_configuration(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    events: list[dict] = []
    ExternalObservationEngine().run_unified("openai", [ChatMessage("user", "hello")], {}, events.append)
    error = next(event for event in events if event["type"] == "generation.error")
    assert "not configured" in error["data"]["message"]
    assert "test-key" not in str(error)


def test_ollama_ndjson_stream_is_normalized(monkeypatch):
    monkeypatch.setattr("app.providers.ollama.model_info", lambda model: {"id": model, "status": "READY"})
    monkeypatch.setattr("app.providers.external.urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse([
        '{"message":{"content":"Hello"},"done":false}\n',
        '{"message":{"content":" Ollama"},"done":true,"prompt_eval_count":3,"eval_count":2}\n',
    ]))
    events: list[dict] = []
    ExternalObservationEngine().run_unified("ollama", [ChatMessage("user", "Say hello")], {"model": "llama3.2"}, events.append)
    assert [event["data"]["text"] for event in events if event["type"] == "response.text.delta"] == ["Hello", " Ollama"]
    complete = next(event for event in events if event["type"] == "generation.completed")
    assert complete["data"]["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
