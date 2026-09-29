import json
import math
import pytest
import torch
from app.providers.base import ChatMessage
from app.providers.external import ExternalObservationEngine
from app.providers.telemetry import telemetry

class Stream:
    def __init__(self, rows): self.rows = rows
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __iter__(self): return iter(json.dumps(row).encode() for row in self.rows)


def test_ollama_authoritative_metrics_and_options(monkeypatch):
    monkeypatch.setattr('app.providers.ollama.model_info', lambda model: {'id': model, 'status': 'READY', 'architecture': 'llama', 'num_layers': 28})
    clocks = iter([1.0, 1.4, 3.0])
    monkeypatch.setattr('app.providers.external.time.perf_counter', lambda: next(clocks))
    def request(req, **kw):
        payload = json.loads(req.data)
        assert payload['model'] == 'llama3.2'
        assert payload['options'] == {'temperature': 0.3, 'top_p': 0.8, 'top_k': 7, 'num_predict': 150}
        assert payload['logprobs'] is True
        return Stream([
            {'message': {'content': 'One chunk with several words'}, 'done': False},
            {'message': {'content': ''}, 'done': True, 'prompt_eval_count': 32, 'eval_count': 141,
             'prompt_eval_cached_count': 10, 'eval_duration': 2_000_000_000, 'prompt_eval_duration': 100_000_000,
             'load_duration': 200_000_000, 'total_duration': 2_300_000_000},
        ])
    monkeypatch.setattr('app.providers.external.urllib.request.urlopen', request)
    events=[]
    ExternalObservationEngine().run_unified('ollama', [ChatMessage('user','what is ai')], {'model':'llama3.2','max_new_tokens':150,'temperature':0.3,'top_p':0.8,'top_k':7}, events.append)
    complete=next(e for e in events if e['type']=='generation.completed')
    t=complete['data']['telemetry']
    assert t['tokens']['model_input_tokens']==32
    assert t['tokens']['generated_tokens']==141
    assert t['tokens']['sequence_tokens']==173
    assert t['tokens']['cached_prompt_tokens']==10
    assert t['tokens']['user_text_tokens'] is None
    assert t['timing']['ttft_ms']==pytest.approx(400)
    assert t['timing']['provider_total_ms']==2300
    assert t['performance']['decode_tokens_per_second']==70.5
    assert t['model']['num_layers']==28
    assert t['model']['hidden_size'] is None
    assert t['capabilities']['qkv'] is False
    assert t['capabilities']['logprobs'] is False
    assert not any(e['type'].startswith(('layer.', 'token.', 'attention.')) for e in events)


def test_absent_ollama_metrics_remain_unavailable_and_real_logprobs(monkeypatch):
    monkeypatch.setattr('app.providers.ollama.model_info',lambda m:{'id':m,'status':'READY'})
    monkeypatch.setattr('app.providers.external.urllib.request.urlopen',lambda *a,**kw:Stream([
        {'message': {'content':'ok'},'logprobs':[{'token':'ok','logprob':math.log(0.25)}], 'done':True}]))
    events=[]
    ExternalObservationEngine().run_unified('ollama',[ChatMessage('user','hi')],{},events.append)
    t=next(e['data'] for e in reversed(events) if e['type']=='telemetry.updated')
    assert t['tokens']['generated_tokens'] is None
    assert t['tokens']['sequence_tokens'] is None
    assert t['performance']['decode_tokens_per_second'] is None
    assert t['capabilities']['logprobs'] is True
    assert t['timing']['ttft_ms'] is not None
    assert any(e['type']=='provider.logprobs' for e in events)


def test_native_metadata_and_input_counts_are_from_loaded_model(adapter, engine):
    prompt='what is ai'
    tokens, ids, _=adapter.tokenize(prompt)
    expected=adapter._tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True)
    assert ids[0].tolist()==expected
    assert adapter.user_text_token_count(prompt)==len(adapter._tokenizer.encode(prompt,add_special_tokens=False))
    assert len(tokens)==len(expected)
    assert all(token.is_special==(token.id in adapter._tokenizer.all_special_ids) for token in tokens)
    meta=adapter._build_metadata()
    assert meta.num_layers==adapter._model.config.num_hidden_layers
    assert meta.num_params==sum(p.numel() for p in adapter._model.parameters())
    assert meta.dtype==str(next(adapter._model.parameters()).dtype)
    assert meta.device==str(next(adapter._model.parameters()).device)
    events=[]
    engine.run(prompt, {'max_new_tokens':2, 'temperature':0}, events.append)
    rec=engine.current_session
    assert rec is not None
    t=rec.summary()['telemetry']
    assert t['tokens']['model_input_tokens']==ids.shape[1]
    assert t['tokens']['generated_tokens']==len(rec.output_tokens)
    assert t['tokens']['sequence_tokens']==ids.shape[1]+len(rec.output_tokens)
    assert t['tensors']['embeddings']['shape']==[1,ids.shape[1],meta.hidden_size]
    assert '<|im_end|>' not in rec.response
    assert t['timing']['ttft_ms']<=t['timing']['total_ms']


def test_telemetry_does_not_replace_unknown_with_zero():
    t=telemetry('ollama','unknown')
    assert all(value is None for value in t['tokens'].values())
    assert all(value is None for value in t['timing'].values())
