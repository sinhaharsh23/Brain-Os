from __future__ import annotations

from types import SimpleNamespace

from app.device import hardware_device_report, resolve_device, torch_capabilities


class FakeCuda:
    def __init__(self, available: bool):
        self.available = available

    def is_available(self):
        return self.available

    def get_device_name(self, _index):
        return "Fake GPU"

    def get_device_properties(self, _index):
        return SimpleNamespace(total_memory=4 * 1024**3)

    def memory_allocated(self, _index):
        return 256 * 1024**2


class FakeTorch:
    __version__ = "test"

    def __init__(self, *, available: bool, hip: str | None = None):
        self.version = SimpleNamespace(cuda="12.8" if not hip else None, hip=hip)
        self.cuda = FakeCuda(available)


def test_cpu_fallback_when_torch_cannot_access_accelerator():
    torch_module = FakeTorch(available=False)

    capabilities = torch_capabilities(torch_module)
    selection = resolve_device("auto", torch_module)
    explicit_selection = resolve_device("cuda", torch_module)

    assert capabilities.cuda_available is False
    assert capabilities.rocm_available is False
    assert selection.device == "cpu"
    assert selection.backend == "CPU"
    assert explicit_selection.device == "cpu"
    assert explicit_selection.backend == "CPU"
    assert explicit_selection.available is False


def test_cuda_detection_uses_cuda_namespace():
    torch_module = FakeTorch(available=True)

    capabilities = torch_capabilities(torch_module)
    selection = resolve_device("auto", torch_module)

    assert capabilities.cuda_available is True
    assert capabilities.rocm_available is False
    assert selection.device == "cuda"
    assert selection.backend == "CUDA"


def test_rocm_detection_keeps_cuda_device_namespace():
    torch_module = FakeTorch(available=True, hip="7.1")

    capabilities = torch_capabilities(torch_module)
    selection = resolve_device("auto", torch_module)

    assert capabilities.cuda_available is True
    assert capabilities.rocm_available is True
    assert selection.device == "cuda"
    assert selection.backend == "ROCm"


def test_hardware_report_does_not_claim_unavailable_acceleration():
    report = hardware_device_report("auto", model_device="cpu", torch_module=FakeTorch(available=False))

    assert report["backend"] == "CPU"
    assert report["device"] == "cpu"
    assert report["model_device"] == "cpu"
    assert report["gpu_available"] is False
    assert report["cuda_available"] is False
    assert report["rocm_available"] is False


def test_mps_detection_and_fallback():
    class FakeMpsBackend:
        def is_built(self):
            return True

        def is_available(self):
            return True

    class FakeMpsTorch:
        __version__ = "2.8.0"
        backends = SimpleNamespace(mps=FakeMpsBackend())
        cuda = FakeCuda(available=False)

    torch_module = FakeMpsTorch()
    capabilities = torch_capabilities(torch_module)
    selection = resolve_device("auto", torch_module)
    explicit_selection = resolve_device("mps", torch_module)

    assert capabilities.mps_available is True
    assert capabilities.cuda_available is False
    assert selection.device == "mps"
    assert selection.backend == "MPS"
    assert explicit_selection.device == "mps"
    assert explicit_selection.backend == "MPS"
    assert explicit_selection.available is True
