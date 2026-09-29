"""Native protocol regressions: tensors must exist when the UI is notified."""
import asyncio
import io

import pytest
import torch

from app.api.routes import session_attention, session_embedding
from app.inference import gguf_runtime
from app.state import state


def test_native_embeddings_and_attention_are_available_at_capture_event(monkeypatch, tmp_path):
    lines = [
        'META\t1\t4\t2\t1\t2\t10\t100\t2048\t1000',
        'PROMPT\t2', 'PTOKEN\t0\t1\t61', 'PTOKEN\t1\t2\t62', 'PREFILL',
        'EMBED\t0\t1,2,3,4', 'EMBED\t1\t4,3,2,1',
        'ATTN_FULL\t0\t0\t0\t0.25,0.75',
        'ATTN_FULL\t0\t0\t1\t0.5,0.5',
        'ATTN\t0\t0\t0\t1:0.75,0:0.25', 'STEP_DONE\t0',
        'TOKEN\t0\t3\t63\t1',
        # A bounded decode row retains the FIRST columns, not the last columns.
        'ATTN_FULL\t1\t0\t0\t0.2,0.3',
        'ATTN_FULL\t1\t0\t1\t0.1,0.4',
        'ATTN\t1\t0\t0\t1:0.3,0:0.2', 'STEP_DONE\t1',
        'DONE\t1\t2\t0',
    ]

    class Process:
        stdout = io.StringIO('\n'.join(lines))
        def poll(self):
            return 0

    monkeypatch.setattr(gguf_runtime, '_model_path', lambda _: tmp_path / 'model.gguf')
    monkeypatch.setattr(gguf_runtime, '_compile_runner', lambda: tmp_path / 'runner')
    monkeypatch.setattr(gguf_runtime.subprocess, 'Popen', lambda *a, **kw: Process())
    monkeypatch.setattr(state, 'native_sessions', {})
    seen = []

    def on_event(event):
        seen.append(event['type'])
        sid = event['session_id']
        if event['type'] == 'embeddings.complete':
            result = asyncio.run(session_embedding(sid, position=0))
            assert result['vector'] == [1, 2, 3, 4]
            assert len(result['pca3']) == 3
            assert event['data']['count'] == 2
        if event['type'] == 'attention.captured':
            step = event['data']['step']
            result = asyncio.run(session_attention(sid, layer=0, head=0, position=1 + step, full=True))
            assert result['weights'][0]['token_index'] == 0
            assert result['weights'][1]['weight'] == pytest.approx(0.75 if step == 0 else 0.3)

    sid = gguf_runtime.run_native_gguf('llama3.2', 'ab', {'max_new_tokens': 1}, on_event)
    assert state.native_sessions[sid].status == 'complete', state.native_sessions[sid].errors
    assert seen.count('attention.captured') == 2
    assert 'embeddings.complete' in seen
    assert torch.isfinite(state.native_sessions[sid].store.embeddings).all()
    assert state.native_sessions[sid].store.attention_for_position(0, 0) is None
