from __future__ import annotations

from typing import Any

import torch


class HookManager:
    def __init__(self, model: torch.nn.Module) -> None:
        self._handles: list[Any] = []
        self._model = model
        existing = getattr(model, "_brainos_hook_manager", None)
        if existing is not None:
            existing.remove_all()
        model._brainos_hook_manager = self

    def register(self, module: torch.nn.Module, fn: Any) -> None:
        self._handles.append(module.register_forward_hook(fn))

    def register_pre_hook(self, module: torch.nn.Module, fn: Any) -> None:
        self._handles.append(module.register_forward_pre_hook(fn))

    def register_layer_hooks(
        self,
        layer: torch.nn.Module,
        layer_index: int,
        on_qkv,
        on_mlp,
        on_layer_out,
        on_residual=None,
        on_module_event=None,
    ) -> None:
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)

        # Residual input hook (x0 before layer)
        if on_residual is not None:
            self.register_pre_hook(layer, _make_layer_in_hook(layer_index, on_residual))

        if attn is not None:
            if on_module_event is not None:
                self.register_pre_hook(attn, _make_module_event_hook("attention", layer_index, on_module_event, "started"))
                self.register(attn, _make_module_event_hook("attention", layer_index, on_module_event, "completed"))
            for name in ("q", "k", "v", "o"):
                projection = getattr(attn, f"{name}_proj", None)
                if projection is not None:
                    self.register(projection, _make_proj_hook(name, layer_index, on_qkv))
            if on_residual is not None:
                self.register(attn, _make_attn_out_hook(layer_index, on_residual))

        if mlp is not None:
            if on_module_event is not None:
                self.register_pre_hook(mlp, _make_module_event_hook("mlp", layer_index, on_module_event, "started"))
                self.register(mlp, _make_module_event_hook("mlp", layer_index, on_module_event, "completed"))
            gate = getattr(mlp, "gate_proj", None)
            up = getattr(mlp, "up_proj", None)
            down = getattr(mlp, "down_proj", None)
            if gate is not None:
                self.register(gate, _make_gate_hook(layer_index, on_mlp, getattr(mlp, "act_fn", None)))
            if up is not None:
                self.register(up, _make_up_hook(layer_index, on_mlp))
            if down is not None:
                self.register(down, _make_down_hook(layer_index, on_mlp))
            if on_residual is not None:
                self.register(mlp, _make_mlp_out_hook(layer_index, on_residual))

        if on_module_event is not None:
            self.register_pre_hook(layer, _make_layer_event_hook(layer_index, on_module_event, "started"))
        self.register(layer, _make_layer_out_hook(layer_index, on_layer_out, on_residual, on_module_event))

    def remove_all(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


def _detach(t: torch.Tensor) -> torch.Tensor:
    return t.detach()


def _make_proj_hook(name: str, layer_index: int, on_qkv):
    def hook(module, args, output):
        on_qkv(layer_index, name, _detach(output))
        return output

    return hook


def _make_gate_hook(layer_index: int, on_mlp, act_fn):
    def hook(module, args, output):
        on_mlp(layer_index, "gate", _detach(output))
        if act_fn is not None:
            try:
                act = act_fn(output)
                on_mlp(layer_index, "gate_activation", _detach(act))
            except Exception:
                pass
        return output

    return hook


def _make_up_hook(layer_index: int, on_mlp):
    def hook(module, args, output):
        on_mlp(layer_index, "up", _detach(output))
        return output

    return hook


def _make_down_hook(layer_index: int, on_mlp):
    def hook(module, args, output):
        on_mlp(layer_index, "down_output", _detach(output))
        return output

    return hook


def _make_layer_in_hook(layer_index: int, on_residual):
    def hook(module, args):
        if args and isinstance(args[0], torch.Tensor):
            on_residual(layer_index, "layer_in", _detach(args[0]))
        return None

    return hook


def _make_attn_out_hook(layer_index: int, on_residual):
    def hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            on_residual(layer_index, "attn_out", _detach(out))
        return output

    return hook


def _make_mlp_out_hook(layer_index: int, on_residual):
    def hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            on_residual(layer_index, "mlp_out", _detach(out))
        return output

    return hook


def _make_layer_out_hook(layer_index: int, on_layer_out, on_residual=None, on_module_event=None):
    def hook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            on_layer_out(layer_index, _detach(out))
            if on_residual is not None:
                on_residual(layer_index, "layer_out", _detach(out))
        if on_module_event is not None:
            on_module_event("layer", layer_index, "completed")
        return output

    return hook


def _make_module_event_hook(category: str, layer_index: int, on_module_event, status: str):
    def hook(module, *args):
        on_module_event(category, layer_index, status)
        return None

    return hook


def _make_layer_event_hook(layer_index: int, on_module_event, status: str):
    def hook(module, *args):
        on_module_event("layer", layer_index, status)
        return None

    return hook
