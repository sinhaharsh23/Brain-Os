from __future__ import annotations

from app.events.bus import EventBus
from app.inference.engine import InferenceEngine
from app.monitoring.monitor import SystemMonitor
from app.replay.store import ReplayStore
from app.ws.manager import ConnectionManager


class AppState:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.manager = ConnectionManager()
        self.replay = ReplayStore("")
        self.adapter = None
        self.engine = None
        self.monitor = SystemMonitor(lambda: self.adapter)
        self.model_status = "not_loaded"

    def wire(self, replay_dir: str) -> None:
        self.replay = ReplayStore(replay_dir)
        self.bus.subscribe(self.manager.broadcast)


state = AppState()
