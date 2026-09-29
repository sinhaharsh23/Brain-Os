import os
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HF_HOME = str(BASE_DIR / "models" / "hf")
DEFAULT_REPLAY_DIR = str(BASE_DIR / "backend" / "replay_sessions")


class Settings(BaseSettings):
    app_name: str = "BrainOS"
    host: str = "127.0.0.1"
    port: int = 8765
    hf_home: str = Field(default_factory=lambda: os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    default_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    local_model_id: str = Field(default="", validation_alias=AliasChoices("LOCAL_MODEL_ID", "BRAINOS_LOCAL_MODEL_ID"))
    local_model_path: str = Field(default="", validation_alias=AliasChoices("LOCAL_MODEL_PATH", "BRAINOS_LOCAL_MODEL_PATH"))
    local_model_offline: bool = Field(default=True, validation_alias=AliasChoices("LOCAL_MODEL_OFFLINE", "BRAINOS_LOCAL_MODEL_OFFLINE"))
    local_model_allow_download: bool = Field(default=False, validation_alias=AliasChoices("LOCAL_MODEL_ALLOW_DOWNLOAD", "BRAINOS_LOCAL_MODEL_ALLOW_DOWNLOAD"))
    max_new_tokens_default: int = 800
    max_prompt_tokens: int = 1024
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dtype: str = "auto"
    capture_interval_ms: float = 500.0
    capture_limit_tokens: int = 256
    capture_level: Literal["summary", "selected", "full"] = "summary"
    attention_capture: bool = True
    replay_dir: str = Field(default_factory=lambda: DEFAULT_REPLAY_DIR)
    # Full tensor dumps are useful for local debugging, but too large for the
    # normal replay path. Compact replay persistence remains enabled by default.
    enable_full_tensor_cache: bool = False
    max_replay_sessions: int = Field(default=10, ge=1)
    max_replay_disk_gb: float = Field(default=1.0, gt=0)
    replay_capture_steps: int = Field(default=16, ge=1)
    replay_mlp_capture_steps: int = Field(default=4, ge=1)
    replay_logit_top_k: int = Field(default=50, ge=1, le=256)
    cors_origins: str = "http://127.0.0.1:8765,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8765"
    max_prompt_chars: int = 12000
    max_new_tokens_limit: int = 1200
    auth_token: str = ""
    # Observatory screens legitimately fan out metadata/tensor requests while
    # browsing a trace. Keep a protective ceiling without throttling a normal
    # localhost session or its browser regression workflow.
    rate_limit_per_minute: int = 600
    auth_mode: Literal["local", "multi_user"] = "local"
    auth_session_ttl_hours: int = 168
    database_url: str = Field(default_factory=lambda: os.environ.get("BRAINOS_DATABASE_URL", f"sqlite:///{BASE_DIR / 'backend' / 'brainos.db'}"))
    database_auto_migrate: bool = True
    scheduler_max_queue_size: int = 4
    scheduler_max_active: int = 1
    scheduler_run_timeout_s: float = 180.0
    openai_api_key: str = Field(default="", validation_alias=AliasChoices("OPENAI_API_KEY", "BRAINOS_OPENAI_API_KEY"))
    anthropic_api_key: str = Field(default="", validation_alias=AliasChoices("ANTHROPIC_API_KEY", "BRAINOS_ANTHROPIC_API_KEY"))
    google_api_key: str = Field(default="", validation_alias=AliasChoices("GOOGLE_API_KEY", "BRAINOS_GOOGLE_API_KEY"))
    gemini_api_key: str = Field(default="", validation_alias=AliasChoices("GEMINI_API_KEY", "BRAINOS_GEMINI_API_KEY"))
    ollama_url: str = Field(default="http://127.0.0.1:11434", validation_alias=AliasChoices("OLLAMA_URL", "BRAINOS_OLLAMA_URL"))
    ollama_models: str = Field(default="llama3.2", validation_alias=AliasChoices("OLLAMA_MODELS", "BRAINOS_OLLAMA_MODELS"))

    model_config = SettingsConfigDict(
        env_file=[str(BASE_DIR / ".env"), ".env"],
        env_prefix="BRAINOS_",
        extra="ignore",
    )

    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        if self.local_model_id:
            self.default_model = self.local_model_id
        rp = Path(self.replay_dir)
        if not rp.is_absolute():
            if rp.parts and rp.parts[0] == "backend":
                self.replay_dir = str((BASE_DIR / rp).resolve())
            else:
                self.replay_dir = str((BASE_DIR / "backend" / rp).resolve())
        hp = Path(self.hf_home)
        if not hp.is_absolute():
            self.hf_home = str((BASE_DIR / hp).resolve())
        # The local resolver is intentionally offline-first.  A caller must
        # explicitly opt into downloads; HF_HUB_OFFLINE still wins at runtime.
        if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
            self.local_model_offline = True
        return self


settings = Settings()

# Ensure HF_HOME is exported to os.environ for transformers and huggingface_hub
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = settings.hf_home
