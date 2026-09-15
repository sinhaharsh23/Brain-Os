from app.events.bus import EventBus
from app.inference.engine import InferenceEngine
from app.providers.registry import PROVIDERS
from app.providers.external import ExternalObservationEngine
from app.models.registry import adapter_for_model
from app.models.adapters import GemmaAdapter, LlamaAdapter, MistralAdapter
from app.models.qwen import QwenAdapter
from app.security import RateLimiter, valid_token
from app.security import validate_generation_params, validate_replay_speed, validate_session_id


def test_replay_controller_seek_and_step():
    import asyncio

    from app.main import ReplayController
    from app.state import state

    class FakeReplay:
        def load_events(self, _session_id):
            return [
                {"type": "dev.log", "data": {"message": "a"}, "ts": 1.0, "session_id": "s"},
                {"type": "dev.log", "data": {"message": "b"}, "ts": 1.1, "session_id": "s"},
                {"type": "dev.log", "data": {"message": "c"}, "ts": 1.2, "session_id": "s"},
            ]

    async def run():
        old_replay = state.replay
        state.replay = FakeReplay()
        controller = ReplayController()
        try:
            await controller.play("s", speed=100)
            await asyncio.sleep(0.01)
            await controller.stop()
            await controller.seek(1)
            assert controller.index == 1
            await controller.step(1)
            assert controller.index == 2
        finally:
            await controller.stop()
            state.replay = old_replay

    asyncio.run(run())


def test_inference_pause_state_is_explicit():
    engine = InferenceEngine(object(), EventBus())
    assert engine.paused is False


def test_inference_pause_and_resume_events(engine):
    import threading

    events = []

    def on_event(event):
        events.append(event.type)
        if event.type == "token.generated" and event.data["step"] == 0:
            engine.pause()
        elif event.type == "inference.paused":
            engine.resume()

    worker = threading.Thread(target=engine.run, args=("Pause between steps", {"max_new_tokens": 3}, on_event))
    worker.start()
    worker.join(timeout=180)
    assert not worker.is_alive()
    assert "inference.paused" in events
    assert "inference.resumed" in events
    engine.pause()
    assert engine.paused is True
    engine.resume()
    assert engine.paused is False
    engine.cancel()
    assert engine.paused is False


def test_provider_capabilities_do_not_claim_external_internals():
    local = next(p for p in PROVIDERS if p.provider_id == "qwen-local")
    external = next(p for p in PROVIDERS if p.provider_id == "openai")
    assert local.inspection_mode == "deep"
    assert local.capabilities.attention is True
    assert external.inspection_mode == "limited"
    assert external.capabilities.hidden_states is False
    assert external.capabilities.qkv is False


def test_external_provider_requires_server_side_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    events = []
    ExternalObservationEngine().run("openai", "hello", {}, lambda event: events.append(event))
    assert events[0]["type"] == "external.started"
    assert events[-1]["type"] == "system.error"
    assert "OPENAI_API_KEY" in events[-1]["data"]["message"]


def test_local_model_registry_selects_architecture_adapters():
    assert adapter_for_model("Qwen/Qwen2.5-0.5B-Instruct") is QwenAdapter
    assert adapter_for_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0") is LlamaAdapter
    assert adapter_for_model("mistralai/Mistral-7B-Instruct-v0.3") is MistralAdapter
    assert adapter_for_model("google/gemma-2-2b-it") is GemmaAdapter
    import pytest

    with pytest.raises(ValueError, match="unsupported model_id"):
        adapter_for_model("unknown/model")


def test_model_catalog_marks_verification_status():
    from app.models.registry import SUPPORTED_MODELS

    statuses = {model["model_id"]: model["verification_status"] for model in SUPPORTED_MODELS}
    assert statuses["Qwen/Qwen2.5-0.5B-Instruct"] == "verified"
    assert statuses["TinyLlama/TinyLlama-1.1B-Chat-v1.0"] == "implemented-unverified"
    assert statuses["mistralai/Mistral-7B-Instruct-v0.3"] == "implemented-unverified"


def test_security_token_comparison_and_rate_limit():
    assert valid_token("secret", "secret") is True
    assert valid_token("wrong", "secret") is False
    limiter = RateLimiter(limit=2, window_seconds=60)
    assert limiter.allow("client") is True
    assert limiter.allow("client") is True
    assert limiter.allow("client") is False
    assert limiter.allow("other") is True


def test_websocket_parameter_validation_rejects_invalid_values():
    import pytest

    with pytest.raises(ValueError):
        validate_generation_params({"temperature": float("nan")}, 128, 512)
    with pytest.raises(ValueError):
        validate_generation_params({"top_p": 0}, 128, 512)
    with pytest.raises(ValueError):
        validate_generation_params({"max_new_tokens": 513}, 128, 512)
    with pytest.raises(ValueError):
        validate_generation_params({"unexpected": 1}, 128, 512)
    with pytest.raises(ValueError):
        validate_replay_speed(0)
    with pytest.raises(ValueError):
        validate_replay_speed(float("inf"))
    with pytest.raises(ValueError):
        validate_session_id("../outside-session")
    assert validate_replay_speed(2) == 2.0


def test_inference_controls_are_owner_scoped():
    engine = InferenceEngine(object(), EventBus())
    first = object()
    second = object()
    engine._owner = first
    assert engine.cancel(second) is False
    assert engine.pause(second) is False
    assert engine.resume(second) is False
    assert engine.paused is False
    assert engine.cancel(first) is True


def test_session_events_are_sent_only_to_bound_connection():
    import asyncio

    from app.events.types import Event
    from app.ws.manager import ConnectionManager

    class FakeSocket:
        def __init__(self):
            self.messages = []

        async def send_json(self, payload):
            self.messages.append(payload)

    async def run():
        manager = ConnectionManager()
        first = FakeSocket()
        second = FakeSocket()
        manager._connections.update({first, second})
        await manager.bind_session("session-a", first)
        assert await manager.bind_session("session-a", second) is False
        await manager.broadcast(Event("token.generated", {"text": "x"}, session_id="session-a"))
        assert len(first.messages) == 1
        assert len(second.messages) == 0
        await manager.broadcast(Event("monitoring.tick", {}, session_id=None))
        assert len(second.messages) == 1

        await manager.disconnect(first)
        await manager.broadcast(Event("token.generated", {"text": "leaked?"}, session_id="session-a"))
        assert len(second.messages) == 1

    asyncio.run(run())


def test_engine_close_removes_hooks():
    class FakeHooks:
        def __init__(self):
            self.removed = False

        def remove_all(self):
            self.removed = True

    engine = InferenceEngine(object(), EventBus())
    hooks = FakeHooks()
    engine._hook_manager = hooks
    engine._hooks_registered = True
    engine._store_ref = {"store": object()}
    engine.close()
    assert hooks.removed is True
    assert engine._hooks_registered is False
    assert engine._closed is True
    assert engine._store_ref["store"] is None


def test_model_switch_unloads_previous_references():
    import asyncio

    from app.main import unload_current_model
    from app.state import state

    class FakeEngine:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeAdapter:
        def __init__(self):
            self.unloaded = False

        def unload(self):
            self.unloaded = True

    old_engine = state.engine
    old_adapter = state.adapter
    engine = FakeEngine()
    adapter = FakeAdapter()
    state.engine = engine
    state.adapter = adapter
    try:
        asyncio.run(unload_current_model())
        assert engine.closed is True
        assert adapter.unloaded is True
        assert state.engine is None
        assert state.adapter is None
    finally:
        state.engine = old_engine
        state.adapter = old_adapter


def test_readiness_requires_loaded_model():
    import asyncio
    import pytest

    from app.api.routes import ready
    from app.state import state

    old_status, old_adapter, old_engine = state.model_status, state.adapter, state.engine
    state.model_status = "loading"
    state.adapter = None
    state.engine = None
    try:
        with pytest.raises(Exception) as exc:
            asyncio.run(ready())
        assert getattr(exc.value, "status_code", None) == 503
    finally:
        state.model_status, state.adapter, state.engine = old_status, old_adapter, old_engine


def test_replay_isolation_rejects_second_connection():
    import asyncio
    import pytest

    from app.main import ReplayController
    from app.state import state

    class FakeReplay:
        def load_events(self, _session_id):
            return []

    class FakeSocket:
        pass

    async def run():
        old_replay = state.replay
        state.replay = FakeReplay()
        state.manager._session_connections.clear()
        first = ReplayController()
        second = ReplayController()
        try:
            await first.play("shared", owner=FakeSocket())
            with pytest.raises(ValueError, match="already active"):
                await second.play("shared", owner=FakeSocket())
        finally:
            await first.stop(release=True)
            await second.stop(release=True)
            state.replay = old_replay

    asyncio.run(run())


def test_engine_close_stops_active_inference(engine):
    import threading

    started = threading.Event()

    def on_event(event):
        if event.type == "step.started":
            started.set()

    worker = threading.Thread(target=engine.run, args=("shutdown while running", {"max_new_tokens": 20}, on_event))
    worker.start()
    assert started.wait(timeout=30)
    engine.close()
    worker.join(timeout=180)
    assert not worker.is_alive()
    assert engine.current_session.status == "cancelled"
