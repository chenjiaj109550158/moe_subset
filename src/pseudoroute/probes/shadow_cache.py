"""Read-only production-cache view and temporary shadow attention buffers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor

from pseudoroute.models.adapters.tiny import TinyMoEAdapter


def _digest(tensors: tuple[Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReadOnlyLayerCache:
    keys: Tensor
    values: Tensor
    current_hidden: Tensor
    current_pre_rope_query: Tensor


@dataclass(frozen=True)
class ReadOnlyProductionKVCache:
    """Immutable cache view; shadow buffers are never attached to this object."""

    layers: tuple[ReadOnlyLayerCache, ...]
    prefix_length: int
    fingerprint: str

    def assert_unchanged(self) -> None:
        tensors = tuple(
            tensor
            for layer in self.layers
            for tensor in (
                layer.keys,
                layer.values,
                layer.current_hidden,
                layer.current_pre_rope_query,
            )
        )
        if _digest(tensors) != self.fingerprint:
            raise RuntimeError("shadow probe mutated the production KV cache")


def build_read_only_cache(
    adapter: TinyMoEAdapter, prefix_token_ids: Tensor
) -> ReadOnlyProductionKVCache:
    model = adapter.model
    hidden = model.embedding(prefix_token_ids.to(next(model.parameters()).device))
    layers = []
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    with torch.inference_mode():
        for layer_idx in adapter.spec.moe_layer_indices:
            layer = adapter._layer(layer_idx)  # noqa: SLF001 - adapter-owned reference path
            normalized = layer.attention_norm(hidden)
            batch, sequence, hidden_size = normalized.shape
            qkv = layer.attention.qkv(normalized).view(
                batch, sequence, 3, layer.attention.num_heads, layer.attention.head_dim
            )
            query, key, value = (qkv[:, :, index].transpose(1, 2) for index in range(3))
            pre_query = query[:, :, -1].detach().clone()
            if layer.attention.rope is not None:
                query = layer.attention.rope(query, positions)
                key = layer.attention.rope(key, positions)
            layers.append(
                ReadOnlyLayerCache(
                    key.detach().clone(),
                    value.detach().clone(),
                    hidden[:, -1].detach().clone(),
                    pre_query,
                )
            )
            hidden, _, _, _ = layer(hidden, capture_trace=False)
    tensors = tuple(
        tensor
        for cached in layers
        for tensor in (
            cached.keys,
            cached.values,
            cached.current_hidden,
            cached.current_pre_rope_query,
        )
    )
    return ReadOnlyProductionKVCache(
        tuple(layers), int(prefix_token_ids.shape[1]), _digest(tensors)
    )


@dataclass
class ShadowKVBuffer:
    """Probe-local pseudo K/V that can grow without touching production state."""

    keys: list[Tensor]
    values: list[Tensor]

    @classmethod
    def empty(cls) -> ShadowKVBuffer:
        return cls([], [])
