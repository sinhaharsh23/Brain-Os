import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "BrainOS"
    host: str = "127.0.0.1"
    port: int = 8765
    hf_home: str = os.environ.get("HF_HOME", str(Path.home() / "BrainOS" / "models" / "hf"))
    default_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens_default: int = 128
    max_prompt_tokens: int = 1024
    device: str = "auto"
    dtype: str = "auto"
    capture_interval_ms: float = 500.0
    replay_dir: str = str(Path.home() / "BrainOS" / "backend" / "replay_sessions")
    enable_full_tensor_cache: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_prefix="BRAINOS_")


settings = Settings()
