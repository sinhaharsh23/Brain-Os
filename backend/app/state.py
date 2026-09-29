from __future__ import annotations

from app.events.bus import EventBus
from app.inference.engine import InferenceEngine
from app.monitoring.monitor import SystemMonitor
from app.models.manager import ModelManager
from app.replay.store import ReplayStore
from app.ws.manager import ConnectionManager
from app.persistence import Persistence, migrate_database


class AppState:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.manager = ConnectionManager()
        self.replay = ReplayStore("")
        self.adapter = None
        self.engine = None
        # ModelManager is the single owner of model/tokenizer lifecycle. The
        # adapter/engine aliases remain on AppState because the API and
        # scheduler intentionally consume the currently active runtime.
        self.model_manager = ModelManager()
        self.monitor = SystemMonitor(lambda: self.adapter)
        self.model_status = "not_loaded"
        self.model_error_code: str | None = None
        self.model_error: str | None = None
        self.persistence: Persistence | None = None
        self.scheduler = None
        self.native_sessions: dict[str, Any] = {}

    def wire(
        self,
        replay_dir: str,
        *,
        max_sessions: int = 10,
        max_disk_gb: float = 1.0,
        enable_full_tensor_cache: bool = False,
        capture_steps: int = 16,
        mlp_capture_steps: int = 4,
        logit_top_k: int = 50,
    ) -> None:
        self.replay = ReplayStore(
            replay_dir,
            max_sessions=max_sessions,
            max_disk_gb=max_disk_gb,
            enable_full_tensor_cache=enable_full_tensor_cache,
            capture_steps=capture_steps,
            mlp_capture_steps=mlp_capture_steps,
            logit_top_k=logit_top_k,
        )
        self.bus.subscribe(self.manager.broadcast)

    def wire_persistence(self, database_url: str, *, auto_migrate: bool = True) -> None:
        if auto_migrate:
            migrate_database(database_url)
        self.persistence = Persistence(database_url)


state = AppState()
