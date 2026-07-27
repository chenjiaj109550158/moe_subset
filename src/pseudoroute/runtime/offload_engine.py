"""PyTorch-only, batch-1 CUDA expert offload prototype for the tiny MoE."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pseudoroute.models.tiny_moe import ExpertMLP, TinyMoE, TinyMoELayer
from pseudoroute.types import ExpertKey


class ExpertState(StrEnum):
    CPU = "CPU"
    LOADING = "LOADING"
    RESIDENT = "RESIDENT"
    EVICTING = "EVICTING"


@dataclass
class ExpertHandle:
    key: ExpertKey
    up_weight: Tensor
    down_weight: Tensor
    pinned: bool
    byte_size: int
    state: ExpertState = ExpertState.CPU
    slot_idx: int | None = None
    transfer_complete: torch.cuda.Event | None = None


class ExpertSlot(nn.Module):
    def __init__(self, hidden_size: int, expert_hidden_size: int, device: torch.device) -> None:
        super().__init__()
        self.up_weight: Tensor
        self.down_weight: Tensor
        self.register_buffer(
            "up_weight", torch.empty(expert_hidden_size, hidden_size, device=device)
        )
        self.register_buffer(
            "down_weight", torch.empty(hidden_size, expert_hidden_size, device=device)
        )
        self.logical_expert: ExpertKey | None = None
        self.compute_complete: torch.cuda.Event | None = None

    def forward(self, hidden: Tensor) -> Tensor:
        return F.linear(F.silu(F.linear(hidden, self.up_weight)), self.down_weight)


@dataclass(frozen=True)
class TransferRecord:
    key: ExpertKey
    slot_idx: int
    bytes: int
    transfer_ms: float
    exposed_stall_ms: float
    asynchronous: bool


@dataclass(frozen=True)
class RuntimeMetrics:
    h2d_bytes: int
    exposed_stall_ms: float
    transfer_ms: float
    host_elapsed_ms: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    resident_expert_bytes: int
    pinned_cpu_bytes: int
    transfer_records: tuple[TransferRecord, ...]


@dataclass
class _PendingTiming:
    key: ExpertKey
    slot_idx: int
    bytes: int
    transfer_start: torch.cuda.Event
    transfer_end: torch.cuda.Event
    stall_start: torch.cuda.Event
    stall_end: torch.cuda.Event
    asynchronous: bool


class TinyMoEOffloadEngine:
    """CPU expert store with fixed per-layer CUDA slots and explicit dependencies."""

    def __init__(
        self,
        model: TinyMoE,
        *,
        slots_per_layer: int,
        pinned_memory: bool,
        asynchronous: bool,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("M10 offload engine requires CUDA")
        if slots_per_layer < 1 or slots_per_layer >= model.config.num_experts:
            raise ValueError("slot budget must be positive and smaller than full experts")
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise ValueError("dense model must be on a CUDA device")
        self.model = model
        self.device = device
        self.slots_per_layer = slots_per_layer
        self.pinned_memory_requested = pinned_memory
        self.asynchronous = asynchronous
        self.handles: dict[ExpertKey, ExpertHandle] = {}
        self.slots: list[list[ExpertSlot]] = [
            [
                ExpertSlot(model.config.hidden_size, model.config.expert_hidden_size, device)
                for _ in range(slots_per_layer)
            ]
            for _ in range(model.config.num_layers)
        ]
        self.transfer_stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        self._clock = 0
        self._last_used: dict[ExpertKey, int] = {}
        self._pending: list[_PendingTiming] = []
        self._h2d_bytes = 0
        self._host_start = 0.0
        self._extract_cpu_experts()

    def _cpu_tensor(self, tensor: Tensor) -> tuple[Tensor, bool]:
        cpu = tensor.detach().to(device="cpu", copy=True).contiguous()
        if self.pinned_memory_requested:
            try:
                cpu = cpu.pin_memory()
            except RuntimeError:
                return cpu, False
        return cpu, cpu.is_pinned()

    def _extract_cpu_experts(self) -> None:
        for layer_idx, raw_layer in enumerate(self.model.layers):
            layer = cast(TinyMoELayer, raw_layer)
            for expert_idx, module in enumerate(layer.experts):
                expert = cast(ExpertMLP, module)
                up, up_pinned = self._cpu_tensor(expert.up.weight)
                down, down_pinned = self._cpu_tensor(expert.down.weight)
                key = ExpertKey(layer_idx, expert_idx)
                self.handles[key] = ExpertHandle(
                    key,
                    up,
                    down,
                    up_pinned and down_pinned,
                    up.numel() * up.element_size() + down.numel() * down.element_size(),
                )
            # The engine owns the only expert copies; discard the original modules so CPU
            # accounting is not silently doubled and no expert parameter remains on CUDA.
            layer.experts = nn.ModuleList(nn.Identity() for _ in range(len(layer.experts)))

    @property
    def pinned_cpu_bytes(self) -> int:
        return sum(handle.byte_size for handle in self.handles.values() if handle.pinned)

    @property
    def resident_expert_bytes(self) -> int:
        return sum(
            slot.up_weight.numel() * slot.up_weight.element_size()
            + slot.down_weight.numel() * slot.down_weight.element_size()
            for layer_slots in self.slots
            for slot in layer_slots
        )

    def reset_metrics(self) -> None:
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._pending.clear()
        self._h2d_bytes = 0
        self._host_start = time.perf_counter()

    def _choose_slot(self, key: ExpertKey) -> tuple[int, ExpertSlot]:
        layer_slots = self.slots[key.layer_idx]
        for slot_idx, slot in enumerate(layer_slots):
            if slot.logical_expert == key:
                return slot_idx, slot
        for slot_idx, slot in enumerate(layer_slots):
            if slot.logical_expert is None:
                return slot_idx, slot

        def last_used(index: int) -> int:
            logical = layer_slots[index].logical_expert
            if logical is None:
                return -1
            return self._last_used.get(logical, -1)

        slot_idx = min(
            range(len(layer_slots)),
            key=last_used,
        )
        return slot_idx, layer_slots[slot_idx]

    def _load(self, key: ExpertKey) -> ExpertSlot:
        slot_idx, slot = self._choose_slot(key)
        self._clock += 1
        self._last_used[key] = self._clock
        if slot.logical_expert == key:
            return slot
        old_key = slot.logical_expert
        if old_key is not None:
            old = self.handles[old_key]
            old.state = ExpertState.EVICTING
            old.slot_idx = None
            old.state = ExpertState.CPU
        handle = self.handles[key]
        handle.state = ExpertState.LOADING
        transfer_start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        transfer_end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        stall_start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        stall_end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        current = torch.cuda.current_stream(self.device)
        stall_start.record(current)
        with torch.cuda.stream(self.transfer_stream):
            if slot.compute_complete is not None:
                self.transfer_stream.wait_event(slot.compute_complete)
            transfer_start.record(self.transfer_stream)
            slot.up_weight.copy_(handle.up_weight, non_blocking=self.asynchronous)
            slot.down_weight.copy_(handle.down_weight, non_blocking=self.asynchronous)
            transfer_end.record(self.transfer_stream)
        if self.asynchronous:
            current.wait_event(transfer_end)
        else:
            transfer_end.synchronize()
        stall_end.record(current)
        handle.transfer_complete = transfer_end
        handle.state = ExpertState.RESIDENT
        handle.slot_idx = slot_idx
        slot.logical_expert = key
        self._h2d_bytes += handle.byte_size
        self._pending.append(
            _PendingTiming(
                key,
                slot_idx,
                handle.byte_size,
                transfer_start,
                transfer_end,
                stall_start,
                stall_end,
                self.asynchronous,
            )
        )
        return slot

    def preload(self, experts: frozenset[ExpertKey]) -> None:
        by_layer: dict[int, list[ExpertKey]] = {}
        for key in sorted(experts):
            by_layer.setdefault(key.layer_idx, []).append(key)
        if any(len(keys) > self.slots_per_layer for keys in by_layer.values()):
            raise ValueError("subset plan exceeds fixed per-layer slot budget")
        for keys in by_layer.values():
            for key in keys:
                self._load(key)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Tensor,
        *,
        subset_plan: frozenset[ExpertKey] | None = None,
        miss_policy: Literal["lossless_fallback", "hard_commit"] = "lossless_fallback",
    ) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("M10 prototype supports batch size 1 only")
        if input_ids.device != self.device:
            raise ValueError("input IDs must be on the engine CUDA device")
        if subset_plan is not None:
            self.preload(subset_plan)
        hidden = self.model.embedding(input_ids)
        for layer_idx, raw_layer in enumerate(self.model.layers):
            layer = cast(TinyMoELayer, raw_layer)
            attention_output, _, _ = layer.attention(layer.attention_norm(hidden))
            hidden = hidden + attention_output
            router_input = layer.router_norm(hidden)
            scores = layer.router(router_input).softmax(dim=-1)
            if subset_plan is not None and miss_policy == "hard_commit":
                allowed = torch.tensor(
                    [
                        ExpertKey(layer_idx, index) in subset_plan
                        for index in range(scores.shape[-1])
                    ],
                    device=self.device,
                    dtype=torch.bool,
                )
                if int(allowed.sum()) < layer.top_k:
                    raise ValueError("hard-commit subset must contain top-k experts per layer")
                scores = scores.masked_fill(~allowed, 0)
            topk_scores, topk_ids = scores.topk(layer.top_k, dim=-1)
            topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
            combined = torch.zeros_like(hidden)
            for expert_idx in sorted(set(topk_ids.reshape(-1).tolist())):
                key = ExpertKey(layer_idx, int(expert_idx))
                slot = self._load(key)
                output = slot(router_input)
                weights = torch.where(
                    topk_ids == expert_idx, topk_weights, torch.zeros_like(topk_weights)
                ).sum(dim=-1, keepdim=True)
                combined.add_(weights * output)
                complete = torch.cuda.Event()  # type: ignore[no-untyped-call]
                complete.record(torch.cuda.current_stream(self.device))
                slot.compute_complete = complete
            if layer.shared_expert is not None:
                combined.add_(layer.shared_expert(router_input))
            hidden = hidden + combined
        return cast(Tensor, self.model.lm_head(self.model.final_norm(hidden)))

    def finish_metrics(self) -> RuntimeMetrics:
        torch.cuda.synchronize(self.device)
        records = tuple(
            TransferRecord(
                pending.key,
                pending.slot_idx,
                pending.bytes,
                pending.transfer_start.elapsed_time(pending.transfer_end),
                pending.stall_start.elapsed_time(pending.stall_end),
                pending.asynchronous,
            )
            for pending in self._pending
        )
        return RuntimeMetrics(
            self._h2d_bytes,
            sum(record.exposed_stall_ms for record in records),
            sum(record.transfer_ms for record in records),
            (time.perf_counter() - self._host_start) * 1000,
            torch.cuda.max_memory_allocated(self.device),
            torch.cuda.max_memory_reserved(self.device),
            self.resident_expert_bytes,
            self.pinned_cpu_bytes,
            records,
        )
