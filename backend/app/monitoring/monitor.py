from __future__ import annotations

import time
from typing import Any

import psutil
import torch


class SystemMonitor:
    def __init__(self, adapter_getter) -> None:
        self._adapter_getter = adapter_getter
        self._history: list[dict[str, Any]] = []
        self._max_history = 600

    def snapshot(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        adapter = self._adapter_getter()
        model_loaded = adapter is not None and adapter.is_loaded
        stats: dict[str, Any] = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_total_gb": round(vm.total / (1024**3), 2),
            "ram_used_gb": round(vm.used / (1024**3), 2),
            "ram_available_gb": round(vm.available / (1024**3), 2),
            "ram_percent": vm.percent,
            "process_ram_gb": round(psutil.Process().memory_info().rss / (1024**3), 3),
            "gpu_percent": None,
            "vram_used_gb": None,
            "vram_total_gb": None,
            "accelerator_memory_label": "Unavailable",
            "accelerator_allocated_gb": None,
            "accelerator_total_gb": None,
            "mps_driver_allocated_gb": None,
            "sources": {"cpu_percent": "psutil", "ram": "psutil", "process_ram_gb": "psutil", "accelerator_allocated_gb": "pytorch_allocator", "gpu_percent": "nvml_if_available"},
            "model_loaded": model_loaded,
            "ts": time.time(),
        }
        if torch.cuda.is_available():
            utilization = _gpu_utilization()
            stats["gpu_percent"] = round(utilization, 1) if utilization is not None else None
            stats["vram_used_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
            stats["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            stats["accelerator_memory_label"] = "CUDA VRAM"
            stats["accelerator_allocated_gb"] = stats["vram_used_gb"]
            stats["accelerator_total_gb"] = stats["vram_total_gb"]
        elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            stats["gpu_percent"] = None
            if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                try:
                    stats["accelerator_allocated_gb"] = round(torch.mps.current_allocated_memory() / (1024**3), 3)
                except Exception:
                    stats["accelerator_allocated_gb"] = None
            stats["accelerator_memory_label"] = "MPS allocated memory"
            stats["accelerator_total_gb"] = None
            try:
                stats["mps_driver_allocated_gb"] = round(torch.mps.driver_allocated_memory() / (1024**3), 3)
            except (AttributeError, RuntimeError):
                pass
        self._history.append(stats)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        return stats

    def history(self) -> list[dict[str, Any]]:
        return self._history


def _gpu_utilization() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return float(util.gpu)
    except Exception:
        return None
