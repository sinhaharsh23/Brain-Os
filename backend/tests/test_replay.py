import asyncio
import json
import tempfile

from app.events.types import Event
from app.inference.engine import InferenceEngine
from app.replay.store import ReplayStore


def test_replay_store_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        store = ReplayStore(d)
        session = {
            "session_id": "abc123",
            "prompt": "hello",
            "model_id": "m",
            "created_at": 1.0,
            "status": "complete",
            "response": "world",
            "num_tokens": 1,
        }
        store.save_session(session)
        store.append_events("abc123", [Event("tokenization.complete", {"n": 1}).to_dict()])
        loaded = store.get_session("abc123")
        assert loaded["summary"]["prompt"] == "hello"
        events = store.load_events("abc123")
        assert events[0]["type"] == "tokenization.complete"
        assert store.list_sessions()[0]["session_id"] == "abc123"


def test_session_record_events(adapter):
    engine = InferenceEngine(adapter, None, max_prompt_tokens=128)
    engine._register_hooks()
    events = []

    async def go():
        loop = asyncio.get_running_loop()

        def on_event(ev):
            events.append(ev)

        await loop.run_in_executor(None, lambda: engine.run("Replay me", {"max_new_tokens": 2}, on_event))

    asyncio.run(go())
    rec = engine.current_session
    assert len(rec.events) == len(events)
    summary = rec.summary()
    assert summary["prompt"] == "Replay me"
    assert summary["status"] == "complete"
    assert summary["num_output_tokens"] == len(rec.output_tokens)
