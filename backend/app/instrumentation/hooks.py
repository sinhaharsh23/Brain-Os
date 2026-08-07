from __future__ import annotations

import inspect
from typing import Any

import torch


class HookManager:
    def __init__(self, model: torch.nn.Module) -> None:
        self._handles: list[Any] = []
        self._model = model

    def register(self, module: torch.nn.Module, fn: Any) -> None:
        self._handles.append(module.register_forward_hook(fn))

    def register_layer_hooks(self, layer, layer_index: int, on_qkv, on_mlp, on_layer_out) -> None:
        attn = layer.self_attn
        mlp = layer.mlp

        self.register(attn.q_proj, _make_proj_hook("q", layer_index, on_qkv))
        self.register(attn.k_proj, _make_proj_hook("k", layer_index, on_qkv))
        self.register(attn.v_proj, _make_proj_hook("v", layer_index, on_qkv))
        self.register(attn.o_proj, _make_proj_hook("o", layer_index, on_qkv))

        self.register(mlp.gate_proj, _make_gate_hook(layer_index, on_mlp))
        self.register(mlp.up_proj, _make_up_hook(layer_index, on_mlp))
        self.register(mlp.down_proj, _make_down_hook(layer_index, on_mlp))

        self.register(layer, _make_layer_out_hook(layer_index, on_layer_out))

    def remove_all(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _detach(t: torch.Tensor) -> torch.Tensor:
    return t.detach()


def _make_proj_hook(name: str, layer_index: int, on_qkv):
    def hook(module, args, output):
        on_qkv(layer_index, name, _detach(output))
        return output

    return hook


def _make_gate_hook(layer_index: int, on_mlp):
    def hook(module, args, output):
        act = torch.nn.functional.silu(_detach(output))
        on_mlp(layer_index, "gate_activation", act)
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


def _make_layer_out_hook(layer_index: int, on_layer_out):
    def hook(module, args, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        on_layer_out(layer_index, _detach(out))
        return output

    return hook


def install_mlp_activation_fn(mlp) -> None:
    for name, child in mlp.named_children():
        if "Activation" in child.__class__.__name__ or isinstance(child, torch.nn.SiLU):
            child.to(torch.float32)
