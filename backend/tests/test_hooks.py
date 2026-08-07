import asyncio

import pytest
import torch

from app.inference.engine import InferenceEngine


def _run_sync(engine, prompt, params):
    events = []

    async def go():
        loop = asyncio.get_running_loop()

        def on_event(ev):
            events.append(ev)

        await loop.run_in_executor(None, lambda: engine.run(prompt, params, on_event))

    asyncio.run(go())
    return engine.current_session, events


def test_hooks_capture_qkv_shapes(engine):
    rec, _ = _run_sync(engine, "Hello world, this is a test prompt", {"max_new_tokens": 3})
    store = rec.store
    assert (0, "q", 0) in store.qkv
    assert (0, "k", 0) in store.qkv
    assert (0, "v", 0) in store.qkv
    q = store.qkv[(0, "q", 0)]
    assert q.shape[0] == rec.store.prompt_length
    assert q.shape[1] == adapter_hidden(engine)
    assert (engine.adapter.metadata.num_layers - 1, "q", 0) in store.qkv


def adapter_hidden(engine):
    return engine.adapter.metadata.hidden_size


def test_attention_captured_all_layers(engine):
    rec, _ = _run_sync(engine, "What is the capital of France?", {"max_new_tokens": 3})
    store = rec.store
    for layer in range(engine.adapter.metadata.num_layers):
        assert (layer, 0) in store.attention
    m = store.attention[(0, 0)]
    assert m.shape[0] == engine.adapter.metadata.num_attention_heads
    assert m.shape[1] == rec.store.prompt_length
    assert m.shape[2] == rec.store.prompt_length


def test_attention_rows_sum_to_one(engine):
    rec, _ = _run_sync(engine, "The cat sat on the mat", {"max_new_tokens": 2})
    m = rec.store.attention[(0, 0)]
    row_sums = m.sum(dim=-1)
    assert torch_allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


def torch_allclose(a, b, atol):
    import torch

    return torch.allclose(a, b, atol=atol)


def test_mlp_activations_captured(engine):
    rec, _ = _run_sync(engine, "Neural networks learn representations", {"max_new_tokens": 2})
    store = rec.store
    gate = store.mlp.get((0, "gate_activation", 0))
    assert gate is not None
    assert gate.shape == (rec.store.prompt_length, engine.adapter.metadata.intermediate_size)
    up = store.mlp.get((0, "up", 0))
    assert up is not None
    assert up.shape == gate.shape


def test_hidden_states_shape(engine):
    rec, _ = _run_sync(engine, "Hidden state inspection", {"max_new_tokens": 2})
    store = rec.store
    hs = store.hidden[(0, 0)]
    assert hs.shape == (rec.store.prompt_length, engine.adapter.metadata.hidden_size)
    final = store.hidden[(engine.adapter.metadata.num_layers, 0)]
    assert final.shape == hs.shape


def test_qkv_for_generated_position(engine):
    rec, _ = _run_sync(engine, "Position mapping test", {"max_new_tokens": 3})
    store = rec.store
    first_gen_pos = store.prompt_length
    q = store.qkv_for_position(0, "q", first_gen_pos)
    assert q is not None
    assert q.shape == (1, engine.adapter.metadata.hidden_size)
    att = store.attention_for_position(0, first_gen_pos)
    assert att is not None
    assert att.shape[1] == 1
