from __future__ import annotations

import logging
from typing import Any

import psutil

from app.device import hardware_device_report
from app.models.base import ModelAdapter
from app.models.adapters import GemmaAdapter, LlamaAdapter, MistralAdapter
from app.models.qwen import QwenAdapter

log = logging.getLogger("brainos.registry")

ADAPTERS: dict[str, type[ModelAdapter]] = {
    "qwen2": QwenAdapter,
    "llama": LlamaAdapter,
    "mistral": MistralAdapter,
    "gemma": GemmaAdapter,
}

SUPPORTED_MODELS: list[dict[str, Any]] = [
    {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter": "qwen2",
        "params_m": 494,
        "description": "Small instruction-tuned Qwen2.5. Runs on CPU. Recommended default.",
        "recommended": True,
        "verification_status": "verified",
        "verification_note": "Real CPU load, inference, streaming, and tensor capture verified on this machine.",
    },
    {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter": "qwen2",
        "params_m": 1540,
        "description": "Larger Qwen2.5. Needs ~4GB RAM; better quality.",
        "recommended": False,
        "verification_status": "implemented-unverified",
        "verification_note": "Not tested in this session; CPU memory headroom is limited.",
    },
    {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "adapter": "qwen2",
        "params_m": 3055,
        "description": "3B Qwen2.5. Needs ~7GB RAM or a GPU.",
        "recommended": False,
        "verification_status": "implemented-unverified",
        "verification_note": "Checkpoint is cached, but current CPU/RAM capacity is insufficient for a safe verification run.",
    },
    {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "adapter": "qwen2",
        "params_m": 7610,
        "description": "7B Qwen2.5. Requires a GPU or large RAM with quantization.",
        "recommended": False,
        "verification_status": "implemented-unverified",
        "verification_note": "Not suitable for the current 7GB CPU-only machine.",
    },
    {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter": "llama",
        "params_m": 1100,
        "description": "Llama-compatible open model. Requires more RAM than the default.",
        "recommended": False,
        "verification_status": "implemented-unverified",
        "verification_note": "Llama-compatible adapter is implemented, but no TinyLlama checkpoint is present in the current local cache for a safe real-checkpoint run.",
    },
    {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "adapter": "mistral",
        "params_m": 7240,
        "description": "Mistral architecture. GPU or quantized large-memory system recommended.",
        "recommended": False,
        "verification_status": "implemented-unverified",
        "verification_note": "Real Hugging Face adapter implemented; 7B checkpoint exceeds current CPU/RAM budget.",
    },
    {
        "model_id": "google/gemma-2-2b-it",
        "adapter": "gemma",
        "params_m": 2610,
        "description": "Gemma architecture. Hugging Face access approval may be required.",
        "recommended": False,
        "verification_status": "implemented-unverified",
        "verification_note": "Real Hugging Face adapter implemented; checkpoint was not loaded on this constrained machine.",
    },
]


def adapter_for_model(model_id: str) -> type[ModelAdapter]:
    if not isinstance(model_id, str):
        raise ValueError("model_id must be a string")
    for descriptor in SUPPORTED_MODELS:
        if descriptor["model_id"] == model_id:
            return ADAPTERS[descriptor["adapter"]]
    raise ValueError(f"unsupported model_id: {model_id}")


def hardware_report(requested_device: str = "auto", model_device: str | None = None) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    cpu_count = psutil.cpu_count(logical=True)
    cpu_percent = psutil.cpu_percent(interval=None)
    report = hardware_device_report(requested_device, model_device)
    return {
        "cpu": {"count": cpu_count, "percent": cpu_percent, "model": _cpu_name()},
        "ram": {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "used_gb": round(vm.used / (1024**3), 2),
            "percent": vm.percent,
        },
        **report,
    }


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    import subprocess
    if platform.system() == "Darwin":
        try:
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False, timeout=1)
            brand = res.stdout.strip()
            if brand:
                return brand
        except Exception:
            pass
        return platform.processor() or "Apple Silicon"
    return "unknown"


def recommend_model() -> dict[str, Any]:
    report = hardware_report()
    ram_gb = report["ram"]["total_gb"]
    if (
        report["gpu_available"]
        and report["gpu"] is not None
        and report["gpu"]["vram_total_gb"] is not None
        and report["gpu"]["vram_total_gb"] >= 8
    ):
        return {"model_id": "Qwen/Qwen2.5-3B-Instruct", "reason": "GPU with sufficient VRAM available"}
    if ram_gb >= 10:
        return {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "reason": f"{ram_gb:.0f}GB RAM available"}
    return {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "reason": f"{ram_gb:.0f}GB RAM; small CPU-friendly model"}
