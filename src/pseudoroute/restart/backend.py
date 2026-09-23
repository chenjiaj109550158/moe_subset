"""Shared slot-indexed MoE computation; no policy-specific precision or kernels."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import functional as F


class MoEComputeBackend:
    def __init__(self, name: str = "vllm_fused", *, audit: bool = False) -> None:
        if name not in ("vllm_fused", "python_reference"):
            raise ValueError(f"unsupported backend {name}")
        self.name, self.audit = name, audit
        self.fused: Any = None
        if name == "vllm_fused":
            from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

            self.fused = fused_experts

    def forward(
        self,
        x: Tensor,
        slot_ids: Tensor,
        routing_weights: Tensor,
        gate_up_slots: Tensor,
        down_slots: Tensor,
        workspace: dict[str, Any],
    ) -> Tensor:
        if x.ndim != 2 or not x.is_contiguous() or x.shape[0] == 0:
            raise ValueError("backend requires nonempty contiguous [T,H]")
        if slot_ids.shape != routing_weights.shape or slot_ids.shape[0] != x.shape[0]:
            raise ValueError("invalid route shape")
        valid = (slot_ids >= 0) & (slot_ids < gate_up_slots.shape[0])
        invalid = (~valid) & (routing_weights != 0)
        if self.audit and bool(invalid.any().item()):
            raise ValueError("nonzero contribution has invalid slot")
        if "error" in workspace:
            workspace["error"].logical_or_(invalid.any())
        safe = slot_ids.clamp(0, gate_up_slots.shape[0] - 1).to(torch.int32)
        weights = routing_weights.float() * valid
        if self.name == "vllm_fused":
            return cast(
                Tensor,
                self.fused(
                    x,
                    gate_up_slots,
                    down_slots,
                    weights,
                    safe,
                    inplace=False,
                    activation="silu",
                    apply_router_weight_on_input=False,
                ),
            )
        output = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        # Deliberately retained reference execution, never labeled optimized.
        used = sorted(set(safe[weights != 0].detach().cpu().flatten().tolist()))
        for expert in used:
            token, rank = torch.where((safe == expert) & (weights != 0))
            if token.numel() == 0:
                continue
            gate, up = F.linear(x[token], gate_up_slots[expert]).chunk(2, -1)
            y = F.linear(F.silu(gate) * up, down_slots[expert])
            output.index_add_(0, token, y.float() * weights[token, rank, None])
        return output.to(x.dtype)


def numerical_metrics(actual: Tensor, reference: Tensor) -> dict[str, Any]:
    a, r = actual.double().flatten(), reference.double().flatten()
    diff = a - r
    rmse = diff.square().mean().sqrt()
    rms = r.square().mean().sqrt()
    zero = bool((rms == 0).item())
    return {
        "finite": bool(torch.isfinite(a).all()),
        "max_abs": float(diff.abs().max()),
        "rmse": float(rmse),
        "nrmse": float(rmse / rms) if not zero else None,
        "cosine": float(F.cosine_similarity(a[None], r[None])) if not zero else None,
        "zero_reference": zero,
        "zero_exact": bool(torch.equal(a, r)) if zero else None,
        "worst_flat_index": int(diff.abs().argmax()),
    }
