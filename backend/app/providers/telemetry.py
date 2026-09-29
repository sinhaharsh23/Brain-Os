"""Provider-neutral, nullable telemetry. Values are never inferred from text chunks."""
from typing import Any


def telemetry(provider: str, model_id: str, *, model=None, capabilities=None, tokens=None,
              timing=None, sampling=None, tensors=None, sources=None) -> dict[str, Any]:
    timings = {key: None for key in ('ttft_ms', 'total_ms', 'provider_total_ms', 'load_ms', 'prompt_eval_ms', 'decode_ms')}
    timings.update(timing or {})
    counts = {key: None for key in ('user_text_tokens', 'model_input_tokens', 'generated_tokens', 'sequence_tokens', 'cached_prompt_tokens', 'template_history_overhead', 'special_output_tokens')}
    counts.update(tokens or {})
    if counts['model_input_tokens'] is not None and counts['generated_tokens'] is not None:
        counts['sequence_tokens'] = counts['model_input_tokens'] + counts['generated_tokens']
    generated = counts['generated_tokens']
    rate = lambda ms: generated / (ms / 1000) if generated is not None and ms is not None and ms > 0 else None
    info = {key: None for key in ('architecture', 'parameter_count', 'context_window', 'config_dtype', 'runtime_dtype', 'device', 'quantization', 'num_layers', 'hidden_size', 'num_attention_heads', 'num_kv_heads', 'intermediate_size')}
    info.update({'id': model_id, **(model or {})})
    return {
        'provider': {'id': provider, 'name': 'Hugging Face Local' if provider == 'qwen-local' else provider,
                     'runtime': 'pytorch' if provider == 'qwen-local' else provider},
        'model': info, 'capabilities': capabilities or {}, 'tokens': counts, 'timing': timings,
        'performance': {'decode_tokens_per_second': rate(timings['decode_ms']), 'end_to_end_tokens_per_second': rate(timings['total_ms'])},
        'sampling': sampling or {}, 'tensors': tensors or {}, 'sources': sources or {},
    }


def native_telemetry(record) -> dict[str, Any]:
    meta = record.metadata
    counts = dict(meta.get('token_usage', {}))
    counts.update(model_input_tokens=len(record.tokens), generated_tokens=len(record.output_tokens))
    if record.output_tokens and all('is_special' in token for token in record.output_tokens):
        counts['special_output_tokens'] = sum(token['is_special'] for token in record.output_tokens)
    model = {key: meta.get(source) for key, source in {
        'architecture': 'architecture', 'parameter_count': 'num_params', 'context_window': 'context_length',
        'runtime_dtype': 'dtype', 'device': 'device', 'quantization': 'quantization', 'num_layers': 'num_layers',
        'hidden_size': 'hidden_size', 'num_attention_heads': 'num_attention_heads', 'num_kv_heads': 'num_kv_heads', 'intermediate_size': 'intermediate_size',
    }.items()}
    model['config_dtype'] = meta.get('extra', {}).get('config_dtype')
    return telemetry('qwen-local', record.model_id, model=model, capabilities=meta.get('capabilities'), tokens=counts,
        timing={**record.timings, 'prompt_eval_ms': record.timings.get('prefill_ms')}, sampling=meta.get('active_sampling', record.params),
        tensors={'embeddings': meta.get('embedding_tensor'), 'capture_available': bool(record.store.qkv or record.store.embeddings is not None)},
        sources={'tokens': 'hf_tokenizer / generated_ids', 'model': 'model_config / pytorch_parameter',
                 'tensors': 'forward_hook', 'timing': 'brainos_timer', 'runtime': 'psutil / mps_allocator'})
