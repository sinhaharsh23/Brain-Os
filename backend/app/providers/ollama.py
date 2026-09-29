"""Ollama API metadata; no HF tokenizer or native tensor introspection."""
import json
import urllib.request
import urllib.error
from app.providers.registry import _ollama_url, LIMITED


def request_json(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(_ollama_url() + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def model_info(model_id):
    result = {'id': model_id, 'status': 'READY', 'source': 'ollama_api', 'capabilities': LIMITED.to_dict()}
    try:
        show = request_json('/api/show', {'model': model_id})
        info = show.get('model_info', {})
        details = show.get('details', {})
        architecture = info.get('general.architecture') or details.get('family')
        field = lambda suffix: info.get(f'{architecture}.{suffix}') if architecture else None
        checkpoint_context = field('context_length')
        result.update(architecture=architecture, parameter_count=info.get('general.parameter_count'),
            parameter_size=details.get('parameter_size'), quantization=details.get('quantization_level'),
            model_context_window=checkpoint_context, context_window=checkpoint_context, context_window_source='ollama_model_info' if checkpoint_context is not None else None,
            device=None, runtime_dtype=None,
            num_layers=field('block_count'), hidden_size=field('embedding_length'), intermediate_size=field('feed_forward_length'),
            num_attention_heads=field('attention.head_count'), num_kv_heads=field('attention.head_count_kv'))
        # Running context is the usable capacity, not the checkpoint maximum.
        try:
            for running in request_json('/api/ps').get('models', []):
                canonical = lambda name: name if ':' in name else name + ':latest'
                if canonical(running.get('name', running.get('model', ''))) == canonical(model_id):
                    running_context = running.get('context_length')
                    result.update(context_window=running_context if running_context is not None else checkpoint_context,
                                  context_window_source='ollama_ps' if running_context is not None else result.get('context_window_source'),
                                  digest=running.get('digest'),
                                  provider_model_bytes=running.get('size'), provider_vram_bytes=running.get('size_vram'))
                    break
        except (OSError, ValueError):
            pass
    except urllib.error.HTTPError as exc:
        result.update(status='MODEL NOT FOUND' if exc.code == 404 else 'ERROR', error=str(exc))
    except (OSError, ValueError) as exc:
        result.update(status='OFFLINE', error=str(exc))
    return result


def installed_models():
    return [entry['name'] for entry in request_json('/api/tags').get('models', []) if entry.get('name')]
