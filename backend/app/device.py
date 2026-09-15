from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch


DeviceMode = Literal["auto", "cpu", "cuda", "mps"]
DeviceBackend = Literal["CPU", "CUDA", "ROCm", "MPS"]


@dataclass(frozen=True)
class TorchCapabilities:
    torch_version: str
    cuda_build: str | None
    hip_build: str | None
    cuda_available: bool
    rocm_available: bool
    mps_built: bool = False
    mps_available: bool = False

    @property
    def backend(self) -> DeviceBackend:
        if self.rocm_available:
            return "ROCm"
        if self.cuda_available:
            return "CUDA"
        if self.mps_available:
            return "MPS"
        return "CPU"


@dataclass(frozen=True)
class DeviceSelection:
    requested: DeviceMode
    device: str
    backend: DeviceBackend
    available: bool


@dataclass(frozen=True)
class PhysicalGpu:
    vendor: str
    name: str
    vram_total_gb: float | None
    vram_used_gb: float | None
    pci_slot: str | None = None


def torch_capabilities(torch_module: Any = torch) -> TorchCapabilities:
    version = getattr(torch_module, "version", None)
    cuda_build = getattr(version, "cuda", None)
    hip_build = getattr(version, "hip", None)
    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except (AttributeError, RuntimeError):
        cuda_available = False

    backends = getattr(torch_module, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    try:
        mps_built = bool(mps_backend and mps_backend.is_built())
    except (AttributeError, RuntimeError):
        mps_built = False
    try:
        mps_available = bool(mps_backend and mps_backend.is_available())
    except (AttributeError, RuntimeError):
        mps_available = False

    return TorchCapabilities(
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        cuda_build=str(cuda_build) if cuda_build else None,
        hip_build=str(hip_build) if hip_build else None,
        cuda_available=cuda_available,
        rocm_available=bool(hip_build) and cuda_available,
        mps_built=mps_built,
        mps_available=mps_available,
    )


def resolve_device(requested: DeviceMode = "auto", torch_module: Any = torch) -> DeviceSelection:
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device mode: {requested}")
    capabilities = torch_capabilities(torch_module)
    if requested == "cpu":
        return DeviceSelection(requested, "cpu", "CPU", True)
    if requested == "cuda":
        if capabilities.cuda_available:
            return DeviceSelection(requested, "cuda", capabilities.backend, True)
        return DeviceSelection(requested, "cpu", "CPU", False)
    if requested == "mps":
        if capabilities.mps_available:
            return DeviceSelection(requested, "mps", "MPS", True)
        return DeviceSelection(requested, "cpu", "CPU", False)
    if capabilities.cuda_available:
        return DeviceSelection(requested, "cuda", capabilities.backend, True)
    if capabilities.mps_available:
        return DeviceSelection(requested, "mps", "MPS", True)
    return DeviceSelection(requested, "cpu", "CPU", requested == "auto")


def discover_physical_gpus(capabilities: TorchCapabilities | None = None) -> list[PhysicalGpu]:
    gpus: list[PhysicalGpu] = []
    for device_path in Path("/sys/class/drm").glob("card*/device"):
        vendor_id = _read_text(device_path / "vendor").lower()
        vendor = {"0x1002": "AMD", "0x10de": "NVIDIA", "0x8086": "Intel"}.get(vendor_id)
        if vendor is None:
            continue
        slot = _read_uevent_value(device_path / "uevent", "PCI_SLOT_NAME")
        name = _lspci_name(slot) if slot else None
        total_bytes = _read_int(device_path / "mem_info_vram_total")
        used_bytes = _read_int(device_path / "mem_info_vram_used")
        gpus.append(
            PhysicalGpu(
                vendor=vendor,
                name=name or f"{vendor} GPU",
                vram_total_gb=_gb(total_bytes),
                vram_used_gb=_gb(used_bytes),
                pci_slot=slot,
            )
        )
    if not gpus and capabilities is not None and capabilities.mps_available:
        try:
            import psutil
            vm = psutil.virtual_memory()
            total_gb = round(vm.total / (1024**3), 2)
            used_gb = round(vm.used / (1024**3), 2)
        except Exception:
            total_gb = None
            used_gb = None
        gpus.append(
            PhysicalGpu(
                vendor="Apple",
                name="Apple Silicon GPU (Metal/MPS)",
                vram_total_gb=total_gb,
                vram_used_gb=used_gb,
            )
        )
    return sorted(gpus, key=lambda gpu: gpu.vram_total_gb or 0.0, reverse=True)


def hardware_device_report(
    requested: DeviceMode = "auto",
    model_device: str | None = None,
    torch_module: Any = torch,
) -> dict[str, Any]:
    capabilities = torch_capabilities(torch_module)
    selection = resolve_device(requested, torch_module)
    physical_gpus = discover_physical_gpus(capabilities)
    gpu = physical_gpus[0] if physical_gpus else _torch_gpu(torch_module, capabilities)
    if gpu is not None and capabilities.cuda_available:
        gpu = PhysicalGpu(
            vendor="AMD" if capabilities.rocm_available else "NVIDIA",
            name=_torch_gpu_name(torch_module, gpu.name),
            vram_total_gb=_torch_gpu_total_gb(torch_module, gpu.vram_total_gb),
            vram_used_gb=_torch_gpu_used_gb(torch_module, gpu.vram_used_gb),
            pci_slot=gpu.pci_slot,
        )
    compute_available = capabilities.cuda_available or capabilities.mps_available
    return {
        "gpu": _gpu_dict(gpu, compute_available),
        "gpu_vendor": gpu.vendor if gpu else None,
        "gpu_name": gpu.name if gpu else None,
        "backend": selection.backend,
        "device": selection.device,
        "model_device": model_device or selection.device,
        "requested_device": selection.requested,
        "gpu_available": compute_available,
        "cuda_available": capabilities.cuda_available,
        "rocm_available": capabilities.rocm_available,
        "mps_available": capabilities.mps_available,
        "torch_version": capabilities.torch_version,
        "cuda_version": capabilities.cuda_build,
        "hip_version": capabilities.hip_build,
    }


def _torch_gpu(torch_module: Any, capabilities: TorchCapabilities) -> PhysicalGpu | None:
    if not capabilities.cuda_available:
        return None
    try:
        total = torch_module.cuda.get_device_properties(0).total_memory / (1024**3)
        used = torch_module.cuda.memory_allocated(0) / (1024**3)
        name = torch_module.cuda.get_device_name(0)
        return PhysicalGpu(
            vendor="AMD" if capabilities.rocm_available else "NVIDIA",
            name=str(name),
            vram_total_gb=round(total, 2),
            vram_used_gb=round(used, 2),
        )
    except (AttributeError, RuntimeError, IndexError):
        return None


def _torch_gpu_name(torch_module: Any, fallback: str) -> str:
    try:
        return str(torch_module.cuda.get_device_name(0))
    except (AttributeError, RuntimeError, IndexError):
        return fallback


def _torch_gpu_total_gb(torch_module: Any, fallback: float | None) -> float | None:
    try:
        return round(torch_module.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except (AttributeError, RuntimeError, IndexError):
        return fallback


def _torch_gpu_used_gb(torch_module: Any, fallback: float | None) -> float | None:
    try:
        return round(torch_module.cuda.memory_allocated(0) / (1024**3), 2)
    except (AttributeError, RuntimeError, IndexError):
        return fallback


def _gpu_dict(gpu: PhysicalGpu | None, compute_available: bool) -> dict[str, Any] | None:
    if gpu is None:
        return None
    return {
        "vendor": gpu.vendor,
        "name": gpu.name,
        "vram_total_gb": gpu.vram_total_gb,
        "vram_used_gb": gpu.vram_used_gb,
        "compute_available": compute_available,
        "pci_slot": gpu.pci_slot,
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _read_int(path: Path) -> int | None:
    value = _read_text(path)
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _read_uevent_value(path: Path, key: str) -> str | None:
    for line in _read_text(path).splitlines():
        name, separator, value = line.partition("=")
        if separator and name == key:
            return value
    return None


def _lspci_name(slot: str) -> str | None:
    try:
        result = subprocess.run(
            ["lspci", "-s", slot, "-nn"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = result.stdout.strip()
    if not line:
        return None
    name = line.split(": ", 1)[-1]
    name = re.sub(r"\s+\[[0-9a-fA-F]{4}:[0-9a-fA-F]{4}\]", "", name)
    return re.sub(r"\s+\(rev [^)]*\)$", "", name).strip()


def _gb(value: int | None) -> float | None:
    return round(value / (1024**3), 2) if value is not None else None
