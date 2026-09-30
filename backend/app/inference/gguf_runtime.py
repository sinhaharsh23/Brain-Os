from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import torch

from app.config import BASE_DIR, settings
from app.events.types import Event
from app.inference.engine import SessionRecord
from app.state import state
from app.instrumentation.stats import pca_projection, pca_project_vector

log = logging.getLogger("brainos.gguf_runtime")
_build_lock = threading.Lock()
_runner: Path | None = None
_active_runs_lock = threading.Lock()
_active_cancellations: dict[str, threading.Event] = {}


def cancel_native_gguf() -> bool:
    with _active_runs_lock:
        if not _active_cancellations:
            return False
        next(reversed(_active_cancellations.values())).set()
        return True


def _model_path(model_id: str) -> Path:
    configured = os.getenv("OLLAMA_GGUF_PATH") or os.getenv("BRAINOS_OLLAMA_GGUF_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
        raise RuntimeError(f"OLLAMA_GGUF_PATH does not exist: {path}")
    ollama = shutil.which("ollama")
    if ollama is None:
        raise RuntimeError("Ollama CLI is required to resolve the installed GGUF model")
    result = subprocess.run([ollama, "show", model_id, "--modelfile"], capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ollama show failed for {model_id}")
    match = re.search(r"^FROM\s+(.+?)\s*$", result.stdout, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Ollama did not return a GGUF file path for {model_id}")
    path = Path(match.group(1).strip('"')).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Ollama GGUF model file not found: {path}")
    return path


def _compile_runner() -> Path:
    global _runner
    with _build_lock:
        source = BASE_DIR / "backend" / "native" / "gguf_runner.cpp"
        if not source.is_file():
            raise RuntimeError(f"native GGUF runner source is missing: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        if _runner is not None and _runner.is_file() and _runner.parent.name == digest:
            return _runner
        brew = shutil.which("brew")
        if brew is None:
            raise RuntimeError("Homebrew is required to build the instrumented llama.cpp runner")
        llama_prefix = subprocess.run([brew, "--prefix", "llama.cpp"], capture_output=True, text=True, timeout=10)
        ggml_prefix = subprocess.run([brew, "--prefix", "ggml"], capture_output=True, text=True, timeout=10)
        if llama_prefix.returncode != 0 or ggml_prefix.returncode != 0:
            raise RuntimeError("Install llama.cpp and ggml with Homebrew to run Ollama deep inspection")
        llama_root = Path(llama_prefix.stdout.strip())
        ggml_root = Path(ggml_prefix.stdout.strip())
        cache = Path(tempfile.gettempdir()) / "brainos-native" / digest
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / "gguf_runner"
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            raise RuntimeError("A C++17 compiler is required to build the instrumented GGUF runner")
        command = [
            compiler, "-std=c++17", "-O2",
            f"-I{llama_root / 'include'}", f"-I{ggml_root / 'include'}",
            f"-L{llama_root / 'lib'}", f"-L{ggml_root / 'lib'}",
            f"-Wl,-rpath,{llama_root / 'lib'}", f"-Wl,-rpath,{ggml_root / 'lib'}",
            str(source), "-lllama", "-lggml", "-lggml-base", "-o", str(binary),
        ]
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=240)
        if compiled.returncode != 0:
            details = (compiled.stderr or compiled.stdout).strip()[-4000:]
            raise RuntimeError(f"failed to build instrumented llama.cpp runner: {details}")
        _runner = binary
        return binary


def run_native_gguf(
    model_id: str,
    prompt: str,
    params: dict[str, Any],
    on_event: Callable[[dict[str, Any]], None],
    messages: list[dict[str, str]] | None = None,
) -> str:
    """Run the selected Ollama GGUF in llama.cpp while collecting real graph tensors."""
    session_id = uuid.uuid4().hex[:12]
    cancel_event = threading.Event()
    with _active_runs_lock:
        _active_cancellations[session_id] = cancel_event
    started = time.perf_counter()
    max_new = max(1, min(int(params.get("max_new_tokens", settings.max_new_tokens_default)), settings.max_new_tokens_limit))
    temperature = max(0.0, min(2.0, float(params.get("temperature", 0.7))))
    top_k = max(0, int(params.get("top_k", 40)))
    top_p = max(0.0, min(1.0, float(params.get("top_p", 0.9))))
    sampling_method = "greedy" if temperature == 0 or top_k == 1 else "temperature+top-k/top-p"
    chat_template = {"available": True, "name": "GGUF bundled llama.cpp chat template", "serialized": "", "message_count": 1, "has_system_message": False, "generation_prompt": True}
    record = SessionRecord(
        session_id, prompt, model_id, dict(params),
        {"adapter_name": "llama.cpp GGUF with graph evaluation capture", "provider": "ollama", "inspection_mode": "deep"},
    )
    record.store.session_id = session_id
    state.native_sessions[session_id] = record
    while len(state.native_sessions) > 5:
        state.native_sessions.pop(next(iter(state.native_sessions)))

    response_bytes = bytearray()
    prompt_tokens: list[dict[str, Any]] = []
    pending: dict[int, dict[str, Any]] = {}
    metadata: dict[str, Any] | None = None
    embedding_rows: dict[int, torch.Tensor] = {}
    output_count = 0
    ttft_ms: float | None = None
    last_token_time = started

    def emit(event_type: str, data: dict[str, Any], ts: float | None = None) -> None:
        event = Event(event_type, data, ts if ts is not None else time.time(), session_id)
        record.add_event(event)
        on_event(event.to_dict())

    emit("inference.started", {
        "prompt": prompt, "params": params, "model_id": model_id, "provider": "ollama",
        "inspection_mode": "deep", "local_or_cloud": "local", "phase": "gguf_runtime",
    })
    try:
        model_path = _model_path(model_id)
        runner = _compile_runner()
    except Exception as exc:
        record.status = "failed"
        record.errors.append(str(exc))
        emit("inference.failed", {"stage": "gguf_runtime", "message": str(exc)})
        with _active_runs_lock:
            _active_cancellations.pop(session_id, None)
        return session_id
    conversation_file = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".brainos-chat")
    for message in messages or [{"role": "user", "content": prompt}]:
        conversation_file.write(f"{message['role']}\t{message['content'].encode('utf-8').hex()}\n")
    conversation_file.flush()
    stderr_file = tempfile.TemporaryFile(mode="w+")
    try:
        process = subprocess.Popen(
            [str(runner), str(model_path), str(max_new), prompt, str(temperature), str(top_k), str(top_p), str(int(params.get("seed", 4294967295))), conversation_file.name],
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as exc:
        conversation_file.close()
        stderr_file.close()
        record.status = "failed"
        record.errors.append(str(exc))
        emit("inference.failed", {"stage": "gguf_runtime", "message": str(exc)})
        with _active_runs_lock:
            _active_cancellations.pop(session_id, None)
        return session_id
    assert process.stdout is not None
    try:
        for raw_line in process.stdout:
            if cancel_event.is_set():
                process.terminate()
                process.wait(timeout=5)
                record.status = "cancelled"
                summary = record.summary()
                summary.update({"status": "cancelled", "inspection_mode": "deep", "local_or_cloud": "local"})
                emit("inference.cancelled", {"summary": summary, "response": record.response, "provider": "ollama", "model": model_id, "num_output_tokens": output_count})
                return session_id
            fields = raw_line.rstrip("\n").split("\t")
            if not fields:
                continue
            kind = fields[0]
            if kind == "META":
                layers, hidden, query_heads, kv_heads, head_dim, vocab_size, num_params, trained_context, model_bytes = map(int, fields[1:10])
                metadata = {
                    "model_id": model_id,
                    "architecture": "GGUF (llama.cpp)",
                    "num_params": num_params,
                    "num_layers": layers,
                    "hidden_size": hidden,
                    "num_attention_heads": query_heads,
                    "num_kv_heads": kv_heads,
                    "head_dim": head_dim,
                    "intermediate_size": 0,
                    "vocab_size": vocab_size,
                    "context_length": min(trained_context, 2048),
                    "max_position_embeddings": trained_context,
                    "activation_function": "SwiGLU",
                    "dtype": "GGUF quantized",
                    "quantization": "GGUF",
                    "device": "Metal (llama.cpp)",
                    "tokenizer_name": model_id,
                    "model_memory_mb": round(model_bytes / (1024 * 1024), 2),
                    "extra": {"runtime": "llama.cpp", "source": str(model_path), "flash_attention": False, "native_context_limit": min(trained_context, 2048), "capture_retention": "prefill query and latest 8 decode steps"},
                    "capabilities": {"attention": True, "qkv": True, "hiddenStates": True, "mlp": True, "embeddings": True, "logits": True, "probabilities": True, "kvCache": False, "tokenization": True},
                }
                record.metadata.update(metadata)
                continue
            if kind == "ARCH":
                if metadata is not None:
                    metadata.update({"architecture": f"{fields[1]} (llama.cpp GGUF)", "intermediate_size": int(fields[2]), "quantization": fields[3], "dtype": fields[3]})
                    record.metadata.update(metadata)
                continue
            if kind == "TEMPLATE":
                chat_template.update({"serialized": bytes.fromhex(fields[1]).decode("utf-8"), "message_count": int(fields[2]), "has_system_message": fields[3] == "1"})
                continue
            if kind == "PROMPT":
                record.store.prompt_length = int(fields[1])
                continue
            if kind == "PTOKEN":
                position, token_id = int(fields[1]), int(fields[2])
                text = bytes.fromhex(fields[3]).decode("utf-8", errors="replace")
                prompt_tokens.append({"position": position, "id": token_id, "text": text, "is_special": text.startswith("<|")})
                continue
            if kind == "PREFILL":
                emit("model.metadata", {"metadata": metadata, "provider": "ollama", "model": model_id, "inspection_mode": "deep"})
                record.tokens = prompt_tokens
                record.store.prompt_length = len(prompt_tokens)
                emit("tokenization.complete", {
                    "tokens": prompt_tokens, "count": len(prompt_tokens), "time_ms": round((time.perf_counter() - started) * 1000, 2),
                    "input_shape": [1, len(prompt_tokens)], "context_length": metadata["context_length"] if metadata else 2048,
                    "context_used": len(prompt_tokens), "context_remaining": max(0, (metadata["context_length"] if metadata else 2048) - len(prompt_tokens)),
                    "chat_template": chat_template,
                })
                continue
            if kind == "EMBED":
                position = int(fields[1])
                vector = torch.tensor([float(value) for value in fields[2].split(",")], dtype=torch.float32)
                if position < record.store.prompt_length:
                    embedding_rows[position] = vector
                elif position == record.store.prompt_length + len(record.store.generated_embeddings):
                    record.store.generated_embeddings.append(vector)
                    token = record.output_tokens[position - record.store.prompt_length]
                    emit("token.embedding", {
                        "position": position, "token_id": token["token_id"], "text": token["text"],
                        "norm": float(vector.norm()), "pca3": pca_project_vector(vector.numpy(), record.store.pca),
                    })
                continue
            if kind == "MODULE":
                step, layer = int(fields[1]), int(fields[2])
                module, phase = fields[3:5]
                emit(f"{module}.{phase}", {"step": step, "layer": layer, "module": module})
                continue
            if kind == "STEP_START":
                step = int(fields[1])
                emit("step.started", {"step": step, "new_token": "(llama.cpp forward pass)"})
                continue
            if kind == "STAT":
                step, layer = int(fields[1]), int(fields[2])
                name = fields[3]
                n = int(fields[4])
                values = list(map(float, fields[5:10]))
                sample = [float(value) for value in fields[10].split(",") if value]
                stats = {"n": n, "min": values[0], "max": values[1], "mean": values[2], "std": values[3], "l2_norm": values[4]}
                slot = pending.setdefault(step, {"qkv": [], "attention": {}, "layers": {}, "mlp": {}, "candidates": []})
                if name in {"q", "k", "v"}:
                    expected = metadata["hidden_size"] if name == "q" else metadata["num_kv_heads"] * metadata["head_dim"]
                    if n != expected:
                        continue
                    slot["qkv"] = [item for item in slot["qkv"] if (item["layer"], item["name"]) != (layer, name)]
                    slot["qkv"].append({"layer": layer, "name": name, "stats": stats})
                elif name == "layer_output":
                    slot["layers"][layer] = {"layer": layer, "hidden_norm": stats["l2_norm"], "hidden_mean": stats["mean"], "hidden_std": stats["std"]}
                elif name == "ffn_activation":
                    top = []
                    if len(fields) > 11:
                        for rank, pair in enumerate(fields[11].split(",")):
                            if pair:
                                index, value = pair.split(":", 1)
                                top.append({"index": int(index), "value": float(value), "rank": rank + 1})
                    slot["mlp"][layer] = {"layer": layer, "top": top, "stats": stats}
                continue
            if kind == "LAYER_DONE":
                step, layer = int(fields[1]), int(fields[2])
                slot = pending.get(step, {})
                item = slot.get("layers", {}).get(layer)
                if item:
                    slot.setdefault("live_layers", set()).add(layer)
                    emit("layer.complete", {**item, "step": step, "is_first": step == 0,
                        "input_shape": [1, 1, metadata["hidden_size"]], "output_shape": [1, 1, metadata["hidden_size"]]})
                continue
            if kind == "ATTN":
                step, layer, head = map(int, fields[1:4])
                links = []
                for pair in fields[4].split(",") if len(fields) > 4 else []:
                    index, weight = pair.split(":", 1)
                    links.append({"head": head, "token_index": int(index), "weight": float(weight)})
                slot = pending.setdefault(step, {"qkv": [], "attention": {}, "layers": {}, "mlp": {}, "candidates": []})
                slot["attention"].setdefault(layer, []).extend(links)
                continue
            if kind == "CAND":
                step, rank, token_id = int(fields[1]), int(fields[2]), int(fields[3])
                logit, probability = float(fields[4]), float(fields[5])
                text = bytes.fromhex(fields[6]).decode("utf-8", errors="replace")
                pending.setdefault(step, {"qkv": [], "attention": {}, "layers": {}, "mlp": [], "candidates": []})["candidates"].append({"token_id": token_id, "rank": rank, "logit": logit, "probability": probability, "text": text})
                continue
            if kind == "STEP_DONE":
                step = int(fields[1])
                slot = pending.get(step, {})
                if step == 0 and len(embedding_rows) == record.store.prompt_length:
                    record.store.embeddings = torch.stack([embedding_rows[i] for i in range(record.store.prompt_length)])
                    pca = pca_projection(record.store.embeddings.numpy(), dims=3)
                    record.store.pca = pca
                    emit("embeddings.complete", {
                        "tokens": [{**token, "norm": float(record.store.embeddings[i].norm()), "pca3": pca["coords"][i]} for i, token in enumerate(prompt_tokens)],
                        "count": len(prompt_tokens), "embedding_dim": int(record.store.embeddings.shape[1]),
                        "explained_variance": pca["explained_variance"], "pca_method": pca["method"], "time_ms": 0.0,
                    })
                for item in slot.get("layers", {}).values():
                    if item["layer"] in slot.get("live_layers", set()):
                        continue
                    sequence = record.store.prompt_length if step == 0 else 1
                    emit("layer.complete", {**item, "step": step, "is_first": step == 0, "input_shape": [1, sequence, metadata["hidden_size"] if metadata else 3072], "output_shape": [1, sequence, metadata["hidden_size"] if metadata else 3072]})
                if slot.get("qkv"):
                    emit("qkv.captured", {"step": step, "layers": slot["qkv"]})
                for layer, mlp in slot.get("mlp", {}).items():
                    width = int(mlp["stats"]["n"])
                    sparse = torch.zeros(width, dtype=torch.float32)
                    for activation in mlp["top"]:
                        sparse[int(activation["index"])] = float(activation["value"])
                    # Older runners may only supply top activations. Never
                    # overwrite a captured full vector with a sparse estimate.
                    if (int(layer), "gate_activation", step) not in record.store.mlp:
                        record.store.mlp[(int(layer), "gate_activation", step)] = sparse.unsqueeze(0)
                    record.metadata.setdefault("native_mlp_stats", {})[f"{layer}:{step}"] = mlp["stats"]
                if slot.get("attention"):
                    query_position = record.store.prompt_length - 1 if step == 0 else record.store.prompt_length + step - 1
                    layers_data = []
                    for layer, links in slot["attention"].items():
                        links.sort(key=lambda item: item["weight"], reverse=True)
                        layers_data.append({"layer": layer, "links": links[:8], "active_position": query_position})
                    emit("attention.captured", {"step": step, "available": True, "layers": layers_data})
                mlp_layers = list(slot.get("mlp", {}).values())
                if mlp_layers:
                    emit("mlp.captured", {"step": step, "layers": mlp_layers})
                if slot.get("candidates"):
                    cand = slot["candidates"]
                    cand.sort(key=lambda item: item["rank"])
                    emit("logits.ready", {"step": step, "candidates": cand, "temperature": temperature, "top_p": top_p, "top_k": top_k})
                record.store.steps_total = max(record.store.steps_total, step + 1)
                # Retain a bounded inspection window, including the prefill
                # query. Streaming full vectors must not grow without bound.
                for tensors in (record.store.qkv, record.store.mlp, record.store.hidden, record.store.attention):
                    for key in list(tensors):
                        if 0 < key[-1] <= step - 8:
                            del tensors[key]
                continue
            if kind == "TOKEN":
                step, token_id = int(fields[1]), int(fields[2])
                text = bytes.fromhex(fields[3]).decode("utf-8", errors="replace")
                probability = float(fields[4]) if len(fields) > 4 else 1.0
                step_candidates = pending.pop(step, {}).get("candidates", [])
                record.store.logit_candidates[step] = step_candidates
                record.store.logit_vocab_sizes[step] = metadata["vocab_size"] if metadata else 0
                if step_candidates:
                    step_candidates.sort(key=lambda item: item["rank"])
                    emit("logits.ready", {"step": step, "candidates": step_candidates, "temperature": temperature, "top_p": top_p, "top_k": top_k, "sampling_method": sampling_method})
                response_bytes.extend(bytes.fromhex(fields[3]))
                position = record.store.prompt_length + step
                now = time.perf_counter()
                latency_ms = round((now - last_token_time) * 1000, 2)
                last_token_time = now
                rank = int(fields[5]) if len(fields) > 5 else 1
                entropy = float(fields[6]) if len(fields) > 6 else None
                token = {"step": step, "token_id": token_id, "text": text, "probability": probability, "rank": rank, "position": position, "time_ms": latency_ms, "entropy": entropy}
                record.output_tokens.append(token)
                record.response = response_bytes.decode("utf-8", errors="replace")
                emit("token.generated", {**token, "output": record.response, "embedding": None, "sampling_method": sampling_method})
                output_count += 1
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000
                continue
            if kind == "VEC":
                step, layer = int(fields[1]), int(fields[2])
                name = fields[3]
                values = [float(value) for value in fields[4].split(",") if value]
                tensor = torch.tensor(values, dtype=torch.float32).reshape(1, -1)
                if name in {"q", "k", "v"}:
                    expected = metadata["hidden_size"] if name == "q" else metadata["num_kv_heads"] * metadata["head_dim"]
                    if len(values) != expected:
                        continue
                    record.store.qkv[(layer, name, step)] = tensor
                    if step == 0:
                        record.store.sequence_starts[0] = max(0, record.store.prompt_length - 1)
                elif name == "attn_out":
                    record.store.qkv[(layer, "o", step)] = tensor
                elif name == "layer_output":
                    record.store.hidden[(layer + 1, step)] = tensor
                elif name in {"ffn_activation", "intermediate", "up"}:
                    record.store.mlp[(layer, "gate_activation" if name == "ffn_activation" else name, step)] = tensor
                if step == 0:
                    record.store.sequence_starts[0] = max(0, record.store.prompt_length - 1)
                continue
            if kind == "ATTN_FULL":
                step, layer, head = map(int, fields[1:4])
                weights = [float(value) for value in fields[4].split(",") if value]
                if step == 0:
                    matrix = record.store.attention.setdefault((layer, step), torch.zeros((metadata["num_attention_heads"], 1, len(weights)), dtype=torch.float32))
                    if matrix.shape[-1] == len(weights) and head < matrix.shape[0]:
                        matrix[head, 0] = torch.tensor(weights, dtype=torch.float32)
                    record.store.attention_starts[step] = max(0, record.store.prompt_length - 1)
                    record.store.attention_column_starts[step] = 0
                else:
                    matrix = record.store.attention.setdefault((layer, step), torch.zeros((metadata["num_attention_heads"], 1, len(weights)), dtype=torch.float32))
                    if matrix.shape[-1] == len(weights) and head < matrix.shape[0]:
                        matrix[head, 0] = torch.tensor(weights, dtype=torch.float32)
                    record.store.attention_column_starts[step] = 0
                continue
            if kind == "DONE":
                generated, input_tokens, decode_result = map(int, fields[1:4])
                if decode_result != 0:
                    raise RuntimeError(f"llama.cpp generation decode failed: {decode_result}")
                duration_ms = (time.perf_counter() - started) * 1000
                record.status = "complete"
                record.timings = {"total_ms": round(duration_ms, 2), "ttft_ms": round(ttft_ms or duration_ms, 2), "tokens_per_second": round(output_count / max(duration_ms / 1000, 0.001), 2)}
                summary = record.summary()
                usage = {"input_tokens": input_tokens, "output_tokens": generated, "total_tokens": input_tokens + generated}
                summary.update({"status": "complete", "inspection_mode": "deep", "local_or_cloud": "local", "num_output_tokens": generated, "usage": usage})
                from app.providers.registry import GGUF_DEEP
                from app.providers.telemetry import telemetry as build_telemetry

                snapshot = build_telemetry(
                    "ollama", model_id,
                    model={
                        "architecture": metadata.get("architecture"), "parameter_count": metadata.get("num_params"),
                        "context_window": metadata.get("context_length"), "runtime_dtype": metadata.get("dtype"),
                        "device": metadata.get("device"), "quantization": metadata.get("quantization"),
                        "num_layers": metadata.get("num_layers"), "hidden_size": metadata.get("hidden_size"),
                        "num_attention_heads": metadata.get("num_attention_heads"), "num_kv_heads": metadata.get("num_kv_heads"),
                        "intermediate_size": metadata.get("intermediate_size"),
                    },
                    capabilities=GGUF_DEEP.to_dict(),
                    tokens={"model_input_tokens": input_tokens, "generated_tokens": generated},
                    timing={"total_ms": record.timings["total_ms"], "ttft_ms": record.timings["ttft_ms"]},
                    sampling=params,
                    tensors={"capture_available": bool(record.store.embeddings is not None or record.store.qkv)},
                    sources={"model": "Ollama installed GGUF", "tokens": "llama.cpp tokenizer / generated token IDs", "tensors": "llama.cpp graph capture", "timing": "BrainOS timer"},
                )
                summary["telemetry"] = snapshot
                emit("telemetry.updated", snapshot)
                emit("inference.complete", {"summary": summary, "response": record.response, "provider": "ollama", "model": model_id, "inspection_mode": "deep", "usage": usage, "num_output_tokens": generated, "timings": record.timings, "telemetry": snapshot})
                return session_id
    except Exception as exc:
        record.status = "failed"
        record.errors.append(str(exc))
        emit("inference.failed", {"stage": "gguf_runtime", "message": str(exc)})
        return session_id
    finally:
        with _active_runs_lock:
            _active_cancellations.pop(session_id, None)
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        conversation_file.close()
        stderr_file.seek(0)
        diagnostic = stderr_file.read()[-2000:].strip()
        stderr_file.close()

    message = "native GGUF runner exited without a completion record"
    if diagnostic:
        message += f": {diagnostic}"
    record.status = "failed"
    record.errors.append(message)
    emit("inference.failed", {"stage": "gguf_runtime", "message": message})
    return session_id
