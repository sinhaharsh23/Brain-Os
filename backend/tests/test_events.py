import asyncio

import pytest

from app.events.bus import EventBus
from app.events.types import Event
from app.inference.engine import InferenceEngine


@pytest.fixture
def bus():
    return EventBus()


def test_bus_publish_subscribe(bus):
    received = []

    async def handler(ev: Event):
        received.append(ev)

    bus.subscribe(handler)
    asyncio.run(bus.publish(Event("test", {"x": 1})))
    assert len(received) == 1
    assert received[0].type == "test"
    bus.unsubscribe(handler)


def test_bus_history(bus):
    asyncio.run(bus.publish(Event("a", {})))
    asyncio.run(bus.publish(Event("b", {})))
    history = bus.history()
    assert [e["type"] for e in history] == ["a", "b"]


def test_engine_emits_event_sequence(adapter):
    engine = InferenceEngine(adapter, EventBus(), max_prompt_tokens=128)
    engine._register_hooks()
    events = []

    async def go():
        loop = asyncio.get_running_loop()

        def on_event(ev):
            events.append(ev)

        await loop.run_in_executor(None, lambda: engine.run("Test event pipeline", {"max_new_tokens": 2}, on_event))

    asyncio.run(go())
    types = [e.type for e in events]
    assert "inference.started" in types
    assert "tokenization.complete" in types
    assert "embeddings.complete" in types
    assert "step.started" in types
    assert "layer.complete" in types
    assert "attention.captured" in types
    assert "qkv.captured" in types
    assert "mlp.captured" in types
    assert "logits.ready" in types
    assert "token.selected" in types
    assert "token.generated" in types
    assert "inference.complete" in types
    assert types.index("inference.started") < types.index("tokenization.complete") < types.index("embeddings.complete")
    first_gen = types.index("token.generated")
    assert "inference.complete" in types[first_gen:]
