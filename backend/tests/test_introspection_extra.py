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


def test_residual_stream_capture(engine):
    rec, events = _run_sync(engine, "The quick brown fox", {"max_new_tokens": 2})
    store = rec.store
    # Check that residual data was captured for layer 0 and final layer
    res_0 = store.residual_for_position(0, 0)
    assert res_0 is not None
    assert "input_norm" in res_0
    assert "attn_delta_norm" in res_0
    assert "mlp_delta_norm" in res_0
    assert "output_norm" in res_0
    assert "cosine_similarity" in res_0
    assert -1.0 <= res_0["cosine_similarity"] <= 1.0

    # Verify that residual.captured events were emitted
    res_events = [e for e in events if e.type == "residual.captured"]
    assert len(res_events) >= 1
    assert "layers" in res_events[0].data


def test_logit_lens_capture(engine):
    rec, events = _run_sync(engine, "Artificial intelligence", {"max_new_tokens": 2})
    store = rec.store
    # Verify logit lens entries
    ll = store.logit_lens_for_position(0, 0)
    assert ll is not None
    assert len(ll) > 0
    assert "token_id" in ll[0]
    assert "probability" in ll[0]
    assert "logit" in ll[0]

    # Verify logit_lens.captured events
    ll_events = [e for e in events if e.type == "logit_lens.captured"]
    assert len(ll_events) >= 1


def test_kv_cache_capture(engine):
    rec, events = _run_sync(engine, "Testing KV cache introspection", {"max_new_tokens": 2})
    store = rec.store
    kv = store.kv_cache_for_position(0)
    assert kv is not None
    assert "total_bytes" in kv
    assert kv["total_bytes"] > 0
    assert "layers" in kv
    assert len(kv["layers"]) == engine.adapter.metadata.num_layers
    assert "key_shape" in kv["layers"][0]
    assert "value_shape" in kv["layers"][0]


def test_qkv_head_slicing(engine):
    rec, _ = _run_sync(engine, "GQA head slicing test", {"max_new_tokens": 2})
    store = rec.store
    # Test head 0 and head 1 for Q
    head_q = store.qkv_head_for_position(
        layer=0,
        name="q",
        head=0,
        position=0,
        head_dim=engine.adapter.metadata.head_dim,
        num_heads=engine.adapter.metadata.num_attention_heads,
    )
    assert head_q is not None
    assert head_q["head_dim"] == engine.adapter.metadata.head_dim
    assert len(head_q["values"]) == engine.adapter.metadata.head_dim
    assert "l2_norm" in head_q["stats"]


def test_model_architecture_tree(adapter):
    tree = adapter.get_architecture_tree(max_depth=3)
    assert tree["name"] == "model"
    assert tree["total_params"] > 0
    assert len(tree["children"]) > 0
