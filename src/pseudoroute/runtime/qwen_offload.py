"""Pinned-CPU to CUDA expert offloading for native Qwen3-MoE execution."""

from __future__ import annotations

import time
import types
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pseudoroute.benchmark.prefetch import Qwen3MoePrefetchOps


@dataclass(frozen=True)
class QwenTransferBatch:
    phase: str
    experts: int
    bytes: int
    transfer_ms: float
    exposed_stall_ms: float


@dataclass(frozen=True)
class QwenOffloadPhaseMetrics:
    requested_experts: int
    cache_hits: int
    cache_misses: int
    h2d_bytes: int
    transfer_batches: int
    transfer_ms: float
    exposed_stall_ms: float


@dataclass(frozen=True)
class QwenOffloadMetrics:
    h2d_bytes: int
    requested_experts: int
    cache_hits: int
    cache_misses: int
    transfer_batches: int
    transfer_ms: float
    exposed_stall_ms: float
    host_elapsed_ms: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    pinned_cpu_expert_bytes: int
    gpu_expert_slot_capacity_bytes: int
    phase_metrics: dict[str, QwenOffloadPhaseMetrics]
    transfer_records: tuple[QwenTransferBatch, ...]


@dataclass
class _CpuExpertLayer:
    gate_up: Tensor
    down: Tensor
    expert_bytes: int


@dataclass
class _CudaLayerSlots:
    gate_up: Tensor
    down: Tensor
    logical_experts: list[int | None]
    expert_to_slot: dict[int, int]
    compute_complete: list[torch.cuda.Event | None]


@dataclass
class _PendingTransfer:
    phase: str
    experts: int
    bytes: int
    transfer_start: torch.cuda.Event
    transfer_end: torch.cuda.Event
    stall_start: torch.cuda.Event
    stall_end: torch.cuda.Event


@dataclass
class _MutablePhaseMetrics:
    requested_experts: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    h2d_bytes: int = 0
    transfer_batches: int = 0


def deterministic_lru_slot(
    logical_experts: list[int | None],
    last_used: dict[int, int],
    protected: frozenset[int],
) -> int:
    """Choose an empty slot, then the LRU unprotected expert with stable ties."""
    for slot, expert in enumerate(logical_experts):
        if expert is None:
            return slot
    candidates = [
        slot
        for slot, expert in enumerate(logical_experts)
        if expert is not None and expert not in protected
    ]
    if not candidates:
        raise RuntimeError("no unprotected CUDA expert slot is available")
    return min(
        candidates,
        key=lambda slot: (
            last_used.get(cast(int, logical_experts[slot]), -1),
            cast(int, logical_experts[slot]),
            slot,
        ),
    )


def nonzero_expert_ids(top_k_index: Tensor, top_k_weights: Tensor) -> tuple[int, ...]:
    """Return ascending expert IDs with at least one nonzero routed weight."""
    if top_k_index.shape != top_k_weights.shape:
        raise ValueError("Qwen route IDs and weights must have identical shapes")
    ids = top_k_index.detach().reshape(-1).cpu()
    nonzero = top_k_weights.detach().reshape(-1).ne(0).cpu()
    return tuple(sorted({int(value) for value in ids[nonzero].tolist()}))


class QwenExpertOffloadEngine:
    """Replace full Qwen expert parameters with pinned CPU storage and B CUDA slots."""

    def __init__(
        self,
        model: nn.Module,
        *,
        slots_per_layer: int,
        pinned_memory_required: bool = True,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen real offload requires CUDA")
        self.model = model
        self.ops = Qwen3MoePrefetchOps(model)
        if not self.ops.top_k <= slots_per_layer < self.ops.num_experts:
            raise ValueError("Qwen slot budget must be in [native top-k, num experts)")
        parameter = next(model.parameters())
        if parameter.device.type != "cuda":
            raise ValueError("Qwen dense model must already be on CUDA")
        if parameter.dtype != torch.bfloat16:
            raise ValueError("Qwen real offload pilot requires bfloat16 model weights")
        self.device = parameter.device
        self.slots_per_layer = slots_per_layer
        self.pinned_memory_required = pinned_memory_required
        self.transfer_stream = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]
        self.cpu_layers: list[_CpuExpertLayer] = []
        self.cuda_layers: list[_CudaLayerSlots] = []
        self._original_forwards: list[Any] = []
        self._last_used: list[dict[int, int]] = [{} for _ in range(self.ops.num_layers)]
        self._clock = 0
        self._phase = "unscoped"
        self._resident_only = False
        self._pending: list[_PendingTransfer] = []
        self._phase_counts: defaultdict[str, _MutablePhaseMetrics] = defaultdict(
            _MutablePhaseMetrics
        )
        self._host_start = 0.0
        torch.cuda.synchronize(self.device)
        setup_start = time.perf_counter()
        self._extract_cpu_experts()
        torch.cuda.empty_cache()
        self._allocate_cuda_slots()
        torch.cuda.synchronize(self.device)
        self.setup_wall_seconds = time.perf_counter() - setup_start
        self._install_forward_overrides()

    def _pinned_copy(self, source: Tensor) -> Tensor:
        target = torch.empty(
            tuple(source.shape),
            dtype=source.dtype,
            device="cpu",
            pin_memory=True,
        )
        target.copy_(source.detach(), non_blocking=False)
        if self.pinned_memory_required and not target.is_pinned():
            raise RuntimeError("Qwen expert CPU allocation is not pinned")
        return target

    def _extract_cpu_experts(self) -> None:
        for layer in range(self.ops.num_layers):
            experts = cast(Any, self.ops.mlp(layer)).experts
            gate_up_parameter = cast(Tensor, experts.gate_up_proj)
            down_parameter = cast(Tensor, experts.down_proj)
            if gate_up_parameter.device != self.device or down_parameter.device != self.device:
                raise ValueError(f"Qwen layer {layer} expert parameters are not on CUDA")
            gate_up = self._pinned_copy(gate_up_parameter)
            down = self._pinned_copy(down_parameter)
            expert_bytes = int(
                gate_up[0].numel() * gate_up.element_size() + down[0].numel() * down.element_size()
            )
            self.cpu_layers.append(_CpuExpertLayer(gate_up, down, expert_bytes))
            del experts.gate_up_proj
            del experts.down_proj
            experts.register_parameter("gate_up_proj", None)
            experts.register_parameter("down_proj", None)
            experts._pseudoroute_expert_bytes = expert_bytes
            experts._pseudoroute_full_experts_offloaded = True

    def _allocate_cuda_slots(self) -> None:
        for store in self.cpu_layers:
            gate_up = torch.empty(
                (self.slots_per_layer, *store.gate_up.shape[1:]),
                dtype=store.gate_up.dtype,
                device=self.device,
            )
            down = torch.empty(
                (self.slots_per_layer, *store.down.shape[1:]),
                dtype=store.down.dtype,
                device=self.device,
            )
            self.cuda_layers.append(
                _CudaLayerSlots(
                    gate_up,
                    down,
                    [None] * self.slots_per_layer,
                    {},
                    [None] * self.slots_per_layer,
                )
            )

    def _install_forward_overrides(self) -> None:
        for layer in range(self.ops.num_layers):
            experts = cast(Any, self.ops.mlp(layer)).experts
            self._original_forwards.append(experts.forward)

            def replacement(
                _module: nn.Module,
                hidden_states: Tensor,
                top_k_index: Tensor,
                top_k_weights: Tensor,
                *,
                layer_index: int = layer,
            ) -> Tensor:
                return self.execute(layer_index, hidden_states, top_k_index, top_k_weights)

            experts.forward = types.MethodType(replacement, experts)

    @property
    def pinned_cpu_expert_bytes(self) -> int:
        return sum(
            layer.gate_up.numel() * layer.gate_up.element_size()
            + layer.down.numel() * layer.down.element_size()
            for layer in self.cpu_layers
            if layer.gate_up.is_pinned() and layer.down.is_pinned()
        )

    @property
    def full_cpu_expert_bytes(self) -> int:
        return sum(
            layer.gate_up.numel() * layer.gate_up.element_size()
            + layer.down.numel() * layer.down.element_size()
            for layer in self.cpu_layers
        )

    @property
    def gpu_expert_slot_capacity_bytes(self) -> int:
        return sum(
            layer.gate_up.numel() * layer.gate_up.element_size()
            + layer.down.numel() * layer.down.element_size()
            for layer in self.cuda_layers
        )

    @property
    def no_full_expert_parameter_on_cuda(self) -> bool:
        return all(
            cast(Any, self.ops.mlp(layer)).experts.gate_up_proj is None
            and cast(Any, self.ops.mlp(layer)).experts.down_proj is None
            for layer in range(self.ops.num_layers)
        )

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        previous = self._phase
        self._phase = name
        try:
            yield
        finally:
            self._phase = previous

    @contextmanager
    def resident_only(self) -> Iterator[None]:
        previous = self._resident_only
        self._resident_only = True
        try:
            yield
        finally:
            self._resident_only = previous

    def resident_experts(self, layer: int) -> frozenset[int]:
        return frozenset(self.cuda_layers[layer].expert_to_slot)

    def reset_cache(self) -> None:
        torch.cuda.synchronize(self.device)
        for layer, slots in enumerate(self.cuda_layers):
            slots.logical_experts[:] = [None] * self.slots_per_layer
            slots.expert_to_slot.clear()
            slots.compute_complete[:] = [None] * self.slots_per_layer
            self._last_used[layer].clear()
        self._clock = 0

    def reset_metrics(self) -> None:
        torch.cuda.synchronize(self.device)
        self._pending.clear()
        self._phase_counts.clear()
        torch.cuda.reset_peak_memory_stats(self.device)
        self._host_start = time.perf_counter()

    def _touch(self, layer: int, expert: int) -> None:
        self._clock += 1
        self._last_used[layer][expert] = self._clock

    def _load_group(self, layer: int, experts: tuple[int, ...]) -> None:
        slots = self.cuda_layers[layer]
        requested = tuple(sorted(set(experts)))
        counts = self._phase_counts[self._phase]
        counts.requested_experts += len(requested)
        hits = tuple(expert for expert in requested if expert in slots.expert_to_slot)
        misses = tuple(expert for expert in requested if expert not in slots.expert_to_slot)
        counts.cache_hits += len(hits)
        counts.cache_misses += len(misses)
        for expert in hits:
            self._touch(layer, expert)
        if not misses:
            return
        if self._resident_only:
            raise RuntimeError(
                f"resident-only Qwen execution missed layer {layer} experts {misses}"
            )
        protected = frozenset(requested)
        assignments: list[tuple[int, int]] = []
        for expert in misses:
            slot = deterministic_lru_slot(
                slots.logical_experts,
                self._last_used[layer],
                protected,
            )
            old = slots.logical_experts[slot]
            if old is not None:
                del slots.expert_to_slot[old]
            slots.logical_experts[slot] = expert
            slots.expert_to_slot[expert] = slot
            assignments.append((expert, slot))
            self._touch(layer, expert)
        current = torch.cuda.current_stream(self.device)
        transfer_start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        transfer_end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        stall_start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        stall_end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        stall_start.record(current)
        with torch.cuda.stream(self.transfer_stream):
            for _, slot in assignments:
                complete = slots.compute_complete[slot]
                if complete is not None:
                    self.transfer_stream.wait_event(complete)
            transfer_start.record(self.transfer_stream)
            store = self.cpu_layers[layer]
            for expert, slot in assignments:
                slots.gate_up[slot].copy_(store.gate_up[expert], non_blocking=True)
                slots.down[slot].copy_(store.down[expert], non_blocking=True)
            transfer_end.record(self.transfer_stream)
        current.wait_event(transfer_end)
        stall_end.record(current)
        byte_count = sum(self.cpu_layers[layer].expert_bytes for _ in assignments)
        counts.h2d_bytes += byte_count
        counts.transfer_batches += 1
        self._pending.append(
            _PendingTransfer(
                self._phase,
                len(assignments),
                byte_count,
                transfer_start,
                transfer_end,
                stall_start,
                stall_end,
            )
        )

    def preload_subsets(self, subsets: dict[int, tuple[int, ...]]) -> None:
        if set(subsets) != set(range(self.ops.num_layers)):
            raise ValueError("Qwen preload subset must cover every routed layer")
        for layer in range(self.ops.num_layers):
            subset = tuple(sorted(subsets[layer]))
            if len(subset) != self.slots_per_layer or len(set(subset)) != len(subset):
                raise ValueError(f"Qwen layer {layer} preload must contain exactly B experts")
            self._load_group(layer, subset)
            if self.resident_experts(layer) != frozenset(subset):
                raise RuntimeError(f"Qwen layer {layer} resident set differs from preload plan")

    @torch.inference_mode()
    def execute(
        self,
        layer: int,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("Qwen offloaded experts require flat [tokens, hidden] input")
        expert_mask = F.one_hot(top_k_index, num_classes=self.ops.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        used = nonzero_expert_ids(top_k_index, top_k_weights)
        output = torch.zeros_like(hidden_states)
        for start in range(0, len(used), self.slots_per_layer):
            group = used[start : start + self.slots_per_layer]
            self._load_group(layer, group)
            slots = self.cuda_layers[layer]
            for expert in group:
                topk_positions, token_indices = torch.where(expert_mask[expert])
                selected_weights = top_k_weights[token_indices, topk_positions]
                slot = slots.expert_to_slot[expert]
                current = hidden_states[token_indices]
                gate, up = F.linear(current, slots.gate_up[slot]).chunk(2, dim=-1)
                act_fn = cast(Any, self.ops.mlp(layer)).experts.act_fn
                value = F.linear(act_fn(gate) * up, slots.down[slot])
                weighted = value * selected_weights[:, None]
                output.index_add_(0, token_indices, weighted.to(output.dtype))
                complete = torch.cuda.Event()  # type: ignore[no-untyped-call]
                complete.record(torch.cuda.current_stream(self.device))
                slots.compute_complete[slot] = complete
                self._touch(layer, expert)
        return output

    def finish_metrics(self) -> QwenOffloadMetrics:
        torch.cuda.synchronize(self.device)
        records = tuple(
            QwenTransferBatch(
                pending.phase,
                pending.experts,
                pending.bytes,
                pending.transfer_start.elapsed_time(pending.transfer_end),
                pending.stall_start.elapsed_time(pending.stall_end),
            )
            for pending in self._pending
        )
        phases: dict[str, QwenOffloadPhaseMetrics] = {}
        for phase, counts in sorted(self._phase_counts.items()):
            phase_records = [record for record in records if record.phase == phase]
            phases[phase] = QwenOffloadPhaseMetrics(
                counts.requested_experts,
                counts.cache_hits,
                counts.cache_misses,
                counts.h2d_bytes,
                counts.transfer_batches,
                sum(record.transfer_ms for record in phase_records),
                sum(record.exposed_stall_ms for record in phase_records),
            )
        return QwenOffloadMetrics(
            sum(value.h2d_bytes for value in phases.values()),
            sum(value.requested_experts for value in phases.values()),
            sum(value.cache_hits for value in phases.values()),
            sum(value.cache_misses for value in phases.values()),
            sum(value.transfer_batches for value in phases.values()),
            sum(record.transfer_ms for record in records),
            sum(record.exposed_stall_ms for record in records),
            (time.perf_counter() - self._host_start) * 1000,
            torch.cuda.max_memory_allocated(self.device),
            torch.cuda.max_memory_reserved(self.device),
            self.pinned_cpu_expert_bytes,
            self.gpu_expert_slot_capacity_bytes,
            phases,
            records,
        )

    def audit(self) -> dict[str, object]:
        return {
            "cpu_layers": len(self.cpu_layers),
            "cuda_slot_layers": len(self.cuda_layers),
            "slots_per_layer": self.slots_per_layer,
            "all_cpu_sources_on_cpu": all(
                layer.gate_up.device.type == "cpu" and layer.down.device.type == "cpu"
                for layer in self.cpu_layers
            ),
            "all_cpu_sources_pinned": all(
                layer.gate_up.is_pinned() and layer.down.is_pinned() for layer in self.cpu_layers
            ),
            "all_slots_on_cuda": all(
                layer.gate_up.device.type == "cuda" and layer.down.device.type == "cuda"
                for layer in self.cuda_layers
            ),
            "no_full_expert_parameter_on_cuda": self.no_full_expert_parameter_on_cuda,
            "full_cpu_expert_bytes": self.full_cpu_expert_bytes,
            "pinned_cpu_expert_bytes": self.pinned_cpu_expert_bytes,
            "gpu_expert_slot_capacity_bytes": self.gpu_expert_slot_capacity_bytes,
            "resident_fraction": self.slots_per_layer / self.ops.num_experts,
            "setup_wall_seconds_measured": self.setup_wall_seconds,
        }
