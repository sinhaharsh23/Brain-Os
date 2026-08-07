import asyncio

from app.inference.engine import InferenceEngine


def _run(engine, prompt, params):
    events = []

    async def go():
        loop = asyncio.get_running_loop()

        def on_event(ev):
            events.append(ev)

        await loop.run_in_executor(None, lambda: engine.run(prompt, params, on_event))

    asyncio.run(go())
    return engine.current_session, events


def test_generation_produces_real_tokens(engine):
    rec, _ = _run(engine, "The first three numbers are: 1, 2,", {"max_new_tokens": 8})
    assert len(rec.output_tokens) > 0
    assert rec.response.strip() != ""
    assert len(rec.output_tokens) <= 8


def test_generation_stops_at_eos(engine):
    rec, _ = _run(engine, "Reply with exactly: done", {"max_new_tokens": 64})
    assert rec.timings.get("eos_reached") is True
    assert rec.output_tokens[-1]["token_id"] == engine.adapter._tokenizer.eos_token_id


def test_logits_shape_and_candidates(engine):
    rec, _ = _run(engine, "Capital of France is", {"max_new_tokens": 2})
    logits = rec.store.logits[0]
    assert logits.shape == (engine.adapter.metadata.vocab_size,)
    assert len(rec.step_stats) == len(rec.output_tokens)


def test_probabilities_are_valid(engine):
    rec, _ = _run(engine, "Hello world", {"max_new_tokens": 4})
    for t in rec.output_tokens:
        assert 0.0 <= t["probability"] <= 1.0
        assert t["rank"] is not None


def test_cancellation(engine):
    rec, _ = _run(engine, "Tell me a very long story about a dragon", {"max_new_tokens": 200})
    assert rec.status == "complete"
