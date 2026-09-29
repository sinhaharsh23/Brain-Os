from __future__ import annotations

import asyncio
import gc
import logging
from typing import Any

import torch

from app.config import settings
from app.device import resolve_device
from app.models.base import BaseModelAdapter, ModelMetadata
from app.models.registry import adapter_for_model

log = logging.getLogger("brainos.model_manager")


class ModelManager:
    """Central manager owning model lifecycle, memory cleanup, device selection, and metadata."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._adapter: BaseModelAdapter | None = None
        self._status: str = "not_loaded"
        self._last_error: str | None = None
        self._model_id: str | None = None

    @property
    def status(self) -> str:
        return self._status

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def adapter(self) -> BaseModelAdapter | None:
        return self._adapter

    @property
    def is_loaded(self) -> bool:
        return self._adapter is not None and self._adapter.is_loaded

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def get_metadata(self) -> dict[str, Any] | None:
        if self._adapter is not None and self._adapter.is_loaded:
            return self._adapter.metadata.to_dict()
        return None

    def get_architecture(self, max_depth: int = 3) -> dict[str, Any] | None:
        if self._adapter is not None and self._adapter.is_loaded:
            return self._adapter.get_architecture_tree(max_depth=max_depth)
        return None

    async def load_model(
        self,
        model_id: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> BaseModelAdapter:
        async with self._lock:
            resolved_model_id = model_id or settings.default_model
            resolved_device = device or settings.device
            resolved_dtype = dtype or settings.dtype

            if self._adapter is not None and self._adapter.is_loaded and self._model_id == resolved_model_id:
                log.info("Model %s already loaded, reusing instance", resolved_model_id)
                return self._adapter

            self._status = "loading"
            self._last_error = None

            # Unload any existing model first to free memory
            await self._unload_unlocked()

            adapter: BaseModelAdapter | None = None
            try:
                adapter_class = adapter_for_model(resolved_model_id)
                adapter = adapter_class(
                    model_id=resolved_model_id,
                    device=resolved_device,
                    dtype=resolved_dtype,
                )
                await asyncio.to_thread(adapter.load)
                self._adapter = adapter
                self._model_id = resolved_model_id
                self._status = "loaded"
                log.info(
                    "Model %s loaded successfully on %s (dtype: %s, params: %d)",
                    resolved_model_id,
                    adapter.device,
                    adapter.metadata.dtype,
                    adapter.metadata.num_params,
                )
                return adapter
            except Exception as exc:
                self._status = "error"
                self._last_error = str(exc)
                if adapter is not None:
                    try:
                        await asyncio.to_thread(adapter.unload)
                    except Exception:
                        log.exception("Failed to clean up model after load error")
                log.exception("Failed to load model %s: %s", resolved_model_id, exc)
                self.cleanup_memory()
                raise

    async def unload(self) -> None:
        async with self._lock:
            await self._unload_unlocked()

    async def _unload_unlocked(self) -> None:
        if self._adapter is not None:
            log.info("Unloading model %s", self._model_id)
            try:
                await asyncio.to_thread(self._adapter.unload)
            except Exception as exc:
                log.warning("Error during adapter unload: %s", exc)
            self._adapter = None
            self._model_id = None
        self._status = "not_loaded"
        self.cleanup_memory()

    @staticmethod
    def cleanup_memory() -> None:
        """Forces memory release across CPU and accelerator caches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        gc.collect()


# Global singleton instance
model_manager = ModelManager()
