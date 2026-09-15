import asyncio
import json
import tempfile

import torch

from app.events.types import Event
from app.events.bus import EventBus
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


def test_replay_store_discovers_sessions_after_restart():
    with tempfile.TemporaryDirectory() as d:
        first = ReplayStore(d)
        first.save_session({"session_id": "disk123", "prompt": "persisted", "status": "complete"})
        second = ReplayStore(d)
        sessions = second.list_sessions()
        assert sessions[0]["session_id"] == "disk123"
        assert sessions[0]["prompt"] == "persisted"


def test_tensor_capture_roundtrip():
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as d:
        store = ReplayStore(d)
        session = SimpleNamespace(
            session_id="tensor123",
            store=SimpleNamespace(
                prompt_length=2,
                embeddings=torch.ones(2, 4),
                generated_embeddings=[torch.zeros(4)],
                pca={"coords": [[0.0, 0.0, 0.0]]},
                qkv={(0, "q", 0): torch.ones(2, 4)},
                mlp={(0, "gate_activation", 0): torch.ones(2, 6)},
                hidden={(0, 0): torch.ones(2, 4)},
                attention={(0, 0): torch.ones(2, 2, 2)},
                logits={0: torch.ones(8)},
                steps_total=1,
            ),
        )
        store.save_session({"session_id": "tensor123", "prompt": "x", "tokens": [], "output_tokens": []})
        store.save_capture(session)
        loaded = store.load_capture("tensor123")
        assert loaded is not None
        assert torch.equal(loaded["embeddings"], torch.ones(2, 4))
        assert loaded["qkv"][(0, "q", 0)].shape == (2, 4)


def test_engine_restores_archived_capture():
    engine = InferenceEngine(object(), EventBus())
    summary = {
        "session_id": "restore123",
        "prompt": "restore",
        "model_id": "test",
        "tokens": [{"text": "restore", "id": 1, "position": 0}],
        "output_tokens": [],
        "status": "complete",
    }
    rec = engine.restore_session(summary, {
        "prompt_length": 1,
        "embeddings": torch.ones(1, 3),
        "qkv": {(0, "q", 0): torch.ones(1, 3)},
        "mlp": {},
        "hidden": {},
        "attention": {},
        "logits": {},
    })
    assert rec.session_id == "restore123"
    assert torch.equal(rec.store.embeddings, torch.ones(1, 3))
    assert engine._archived["restore123"] is rec


def test_archived_session_keeps_model_specific_metadata_and_candidates():
    engine = InferenceEngine(object(), EventBus())
    rec = engine.restore_session(
        {
            "session_id": "isolated123",
            "prompt": "isolate",
            "model_id": "model-a",
            "adapter_name": "adapter-a",
            "metadata": {"tokenizer_name": "tokenizer-a"},
            "tokens": [],
            "output_tokens": [],
        },
        {
            "prompt_length": 0,
            "logits": {0: torch.ones(4)},
            "logit_candidates": {0: [{"token_id": 1, "text": "a", "rank": 0}]},
        },
    )
    assert rec.adapter is None
    assert rec.adapter_name == "adapter-a"
    assert rec.metadata["tokenizer_name"] == "tokenizer-a"
    assert rec.store.logit_candidates[0][0]["text"] == "a"


def test_archived_logits_do_not_use_current_model_adapter():
    import asyncio

    from app.api.routes import session_logits
    from app.state import state

    old_engine = state.engine
    engine = InferenceEngine(object(), EventBus())
    engine.restore_session(
        {
            "session_id": "logit-isolated",
            "prompt": "x",
            "model_id": "archived-model",
            "tokens": [],
            "output_tokens": [],
        },
        {
            "prompt_length": 0,
            "logits": {0: torch.ones(4)},
            "logit_candidates": {0: [{"token_id": 2, "text": "archived", "rank": 0}]},
        },
    )
    state.engine = engine
    try:
        response = asyncio.run(session_logits("logit-isolated", step=0, k=1))
        assert response["candidates"][0]["text"] == "archived"
    finally:
        state.engine = old_engine


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
