import pytest

from app.monitoring.monitor import SystemMonitor


def test_monitor_snapshot():
    monitor = SystemMonitor(lambda: None)
    snap = monitor.snapshot()
    assert "cpu_percent" in snap
    assert "ram_total_gb" in snap
    assert snap["ram_total_gb"] > 0
    assert 0 <= snap["cpu_percent"] <= 100
    assert snap["model_loaded"] is False


def test_monitor_with_loaded_model(adapter):
    monitor = SystemMonitor(lambda: adapter)
    snap = monitor.snapshot()
    assert snap["model_loaded"] is True
    assert snap["process_ram_gb"] > 0


def test_monitor_history(adapter):
    monitor = SystemMonitor(lambda: adapter)
    monitor.snapshot()
    monitor.snapshot()
    assert len(monitor.history()) == 2
