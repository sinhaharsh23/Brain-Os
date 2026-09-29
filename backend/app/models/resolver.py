"""Offline-first resolution and validation for local Hugging Face checkpoints.

The resolver is deliberately independent from Transformers.  That is important:
``from_pretrained(repo_id)`` may perform Hub metadata requests before loading a
model, while BrainOS must be able to decide that a checkpoint is installed
without touching the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import BASE_DIR, DEFAULT_HF_HOME


DEFAULT_LOCAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class LocalModelError(RuntimeError):
    """Base class for user-actionable local model errors."""

    code = "LOCAL_MODEL_ERROR"


class LocalModelNotInstalledError(LocalModelError):
    code = "LOCAL_MODEL_NOT_INSTALLED"


class LocalModelIncompleteError(LocalModelError):
    code = "LOCAL_MODEL_INCOMPLETE"


class LocalModelDownloadError(LocalModelError):
    code = "LOCAL_MODEL_DOWNLOAD_FAILED"


@dataclass(frozen=True)
class ResolvedLocalModel:
    model_id: str
    path: Path
    source: str
    network_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "path": str(self.path),
            "source": self.source,
            "network_required": self.network_required,
        }


class LocalModelResolver:
    """Resolve a usable local checkpoint without contacting Hugging Face."""

    def __init__(
        self,
        *,
        settings_obj: Any | None = None,
        project_root: str | Path | None = None,
        model_id: str | None = None,
        model_path: str | Path | None = None,
        hf_home: str | Path | None = None,
        offline: bool | None = None,
        allow_download: bool | None = None,
    ) -> None:
        if settings_obj is None:
            from app.config import settings

            settings_obj = settings
        self.settings = settings_obj
        self.project_root = Path(project_root or BASE_DIR).expanduser().resolve()
        requested_model_id = model_id
        self.model_id = str(
            model_id
            or getattr(settings_obj, "local_model_id", "")
            or getattr(settings_obj, "default_model", "")
            or DEFAULT_LOCAL_MODEL_ID
        )
        configured_path = getattr(settings_obj, "local_model_path", "")
        configured_id = str(getattr(settings_obj, "local_model_id", "") or getattr(settings_obj, "default_model", "") or DEFAULT_LOCAL_MODEL_ID)
        # A path configured for the default model must not accidentally make a
        # later explicit model switch load the old checkpoint.
        self.model_path = model_path if model_path is not None else (configured_path if (requested_model_id is None or self.model_id == configured_id) else "")
        self.offline = bool(
            getattr(settings_obj, "local_model_offline", True) if offline is None else offline
        ) or _env_truthy("HF_HUB_OFFLINE")
        self.allow_download = bool(
            getattr(settings_obj, "local_model_allow_download", False)
            if allow_download is None
            else allow_download
        )
        self.hf_home = Path(
            hf_home
            or os.environ.get("HF_HOME")
            or getattr(settings_obj, "hf_home", "")
            or DEFAULT_HF_HOME
        ).expanduser()
        if not self.hf_home.is_absolute():
            self.hf_home = (self.project_root / self.hf_home).resolve()

    def resolve_model(self) -> ResolvedLocalModel:
        explicit = self._explicit_path()
        if explicit is not None:
            if not explicit.exists():
                raise LocalModelNotInstalledError(self._not_installed_message(explicit))
            self._validate_or_raise(explicit, explicit=True)
            return ResolvedLocalModel(self.model_id, explicit, "configured_path")

        model_as_path = Path(self.model_id).expanduser()
        if model_as_path.exists():
            resolved = model_as_path if model_as_path.is_absolute() else self.project_root / model_as_path
            resolved = resolved.resolve()
            self._validate_or_raise(resolved, explicit=True)
            return ResolvedLocalModel(self.model_id, resolved, "model_id_path")

        incomplete: list[tuple[Path, list[str]]] = []
        for candidate in self._repository_local_candidates():
            if not candidate.exists():
                continue
            missing = self.validate_checkpoint(candidate)
            if not missing:
                return ResolvedLocalModel(self.model_id, candidate.resolve(), "project_local")
            incomplete.append((candidate, missing))

        for candidate in self._cached_snapshot_candidates():
            missing = self.validate_checkpoint(candidate)
            if not missing:
                return ResolvedLocalModel(self.model_id, candidate.resolve(), "hf_cache")
            incomplete.append((candidate, missing))

        if self.allow_download and not self.offline:
            return self._download()

        if incomplete:
            path, missing = incomplete[0]
            raise LocalModelIncompleteError(
                f"LOCAL_MODEL_INCOMPLETE: checkpoint at {path} is missing {', '.join(missing)}. "
                f"Install a complete {self.model_id} checkpoint or run ./scripts/download-local-model.sh."
            )
        raise LocalModelNotInstalledError(self._not_installed_message())

    def validate_checkpoint(self, path: str | Path) -> list[str]:
        """Return missing required artifacts; never performs network access."""
        root = Path(path).expanduser()
        if not root.is_dir():
            return ["checkpoint directory"]

        missing: list[str] = []
        if not (root / "config.json").is_file():
            missing.append("config.json")

        tokenizer_files = (
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "vocab.json",
        )
        if not any((root / name).is_file() for name in tokenizer_files):
            missing.append("tokenizer files (tokenizer.json/tokenizer.model/vocab.json)")
        if (root / "vocab.json").is_file() and not (root / "merges.txt").is_file() and not (root / "tokenizer.json").is_file():
            missing.append("merges.txt for vocab.json tokenizer")

        weight_files = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        if not any((root / name).is_file() for name in weight_files):
            # Some valid checkpoints use a sharded safetensors layout without
            # an index in older exports. Keep validation useful without
            # treating unrelated files as model weights.
            if not list(root.glob("*.safetensors")) and not list(root.glob("*.bin")):
                missing.append("model weights (safetensors or pytorch_model.bin)")
        return missing

    def _explicit_path(self) -> Path | None:
        raw = str(self.model_path or "").strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def _repository_local_candidates(self) -> list[Path]:
        short_name = self.model_id.rstrip("/").split("/")[-1]
        return [
            (self.project_root / "models" / short_name).resolve(),
            (self.project_root / "models" / self.model_id).resolve(),
        ]

    def _cached_snapshot_candidates(self) -> list[Path]:
        if "/" not in self.model_id:
            return []
        owner, name = self.model_id.split("/", 1)
        cache_roots = []
        hub_cache = os.environ.get("HF_HUB_CACHE")
        if hub_cache:
            cache_roots.append(Path(hub_cache).expanduser())
        cache_roots.append(self.hf_home / "hub")
        repo_cache_name = f"models--{owner}--{name}"
        candidates: list[Path] = []
        for root in cache_roots:
            snapshots = root / repo_cache_name / "snapshots"
            if snapshots.is_dir():
                candidates.extend(sorted((p for p in snapshots.iterdir() if p.is_dir()), key=_mtime, reverse=True))
        return candidates

    def _download(self) -> ResolvedLocalModel:
        target = (self.project_root / "models" / self.model_id.rstrip("/").split("/")[-1]).resolve()
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(target),
                cache_dir=str(self.hf_home / "hub"),
                local_files_only=False,
            )
        except Exception as exc:
            raise LocalModelDownloadError(
                f"LOCAL_MODEL_DOWNLOAD_FAILED: unable to download {self.model_id} to {target}: {exc}. "
                "Install the checkpoint while online, then restart BrainOS offline."
            ) from exc
        self._validate_or_raise(target, explicit=True)
        return ResolvedLocalModel(self.model_id, target, "downloaded")

    def _validate_or_raise(self, path: Path, *, explicit: bool = False) -> None:
        missing = self.validate_checkpoint(path)
        if missing:
            code = "LOCAL_MODEL_INCOMPLETE"
            prefix = "configured checkpoint" if explicit else "checkpoint"
            raise LocalModelIncompleteError(
                f"{code}: {prefix} at {path} is missing {', '.join(missing)}. "
                "Run ./scripts/download-local-model.sh or set LOCAL_MODEL_PATH to a complete checkpoint."
            )

    def _not_installed_message(self, path: Path | None = None) -> str:
        location = f" at {path}" if path is not None else ""
        return (
            f"LOCAL_MODEL_NOT_INSTALLED: {self.model_id}{location} is not installed locally. "
            "BrainOS will not contact Hugging Face during startup. "
            "Run ./scripts/download-local-model.sh while online, or set LOCAL_MODEL_PATH to a complete checkpoint."
        )


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
