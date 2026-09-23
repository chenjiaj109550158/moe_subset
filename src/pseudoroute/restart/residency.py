"""CPU expert backing store and versioned slots with ordered copy/compute lifetimes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from pseudoroute.restart.backend import MoEComputeBackend
from pseudoroute.runtime.qwen_offload import deterministic_lru_slot


@dataclass
class HostLayer:
    gate_up: Tensor
    down: Tensor

    @property
    def expert_bytes(self) -> int:
        return (self.gate_up[0].numel() + self.down[0].numel()) * self.gate_up.element_size()


class ResidencyManager:
    def __init__(
        self,
        stores: list[HostLayer],
        slots: int,
        backend: MoEComputeBackend,
        *,
        device: str = "cuda:0",
        mode: str = "performance",
    ) -> None:
        if mode not in ("audit", "profile", "performance"):
            raise ValueError("invalid instrumentation mode")
        self.device, self.mode, self.backend = torch.device(device), mode, backend
        self.stores, self.slots = stores, slots
        self.copy_stream = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]
        self.weights = [
            (
                torch.empty((slots, *s.gate_up.shape[1:]), device=device, dtype=s.gate_up.dtype),
                torch.empty((slots, *s.down.shape[1:]), device=device, dtype=s.down.dtype),
            )
            for s in stores
        ]
        self.maps = [
            torch.full((s.gate_up.shape[0],), -1, dtype=torch.long, device=device) for s in stores
        ]
        self.done = [torch.cuda.Event() for _ in stores]  # type: ignore[no-untyped-call]
        self.ready = [torch.cuda.Event() for _ in stores]  # type: ignore[no-untyped-call]
        self.error = torch.zeros((), dtype=torch.bool, device=device)
        self.staging = None
        self.host_mode = "full_pinned_cpu"
        if not all(s.gate_up.is_pinned() and s.down.is_pinned() for s in stores):
            self.host_mode = "cpu_resident_with_bounded_pinned_staging"
            s = stores[0]
            self.staging = (
                torch.empty((slots, *s.gate_up.shape[1:]), dtype=s.gate_up.dtype, pin_memory=True),
                torch.empty((slots, *s.down.shape[1:]), dtype=s.down.dtype, pin_memory=True),
            )
        self.reset()

    def reset(self) -> None:
        torch.cuda.synchronize(self.device)
        self.logical: list[list[int | None]] = [[None] * self.slots for _ in self.stores]
        self.lookup: list[dict[int, int]] = [{} for _ in self.stores]
        self.ages: list[dict[int, int]] = [{} for _ in self.stores]
        self.versions = [0] * len(self.stores)
        self.has_compute = [False] * len(self.stores)
        self.has_copy = [False] * len(self.stores)
        self.clock = 0
        for mapping in self.maps:
            mapping.fill_(-1)
        self.error.zero_()
        self.counts: Any = defaultdict(lambda: defaultdict(int))
        self.events: list[Any] = []
        self.trace: list[Any] = []
        self.phase = "prefill"

    def load(self, layer: int, experts: tuple[int, ...], *, asynchronous: bool = False) -> None:
        requested = tuple(sorted(set(experts)))
        if len(requested) > self.slots or any(
            e < 0 or e >= self.maps[layer].numel() for e in requested
        ):
            raise ValueError("invalid residency request")
        misses = [e for e in requested if e not in self.lookup[layer]]
        c = self.counts[self.phase]
        c["requested_experts"] += len(requested)
        c["cache_hits"] += len(requested) - len(misses)
        c["cache_misses"] += len(misses)
        for e in requested:
            self.clock += 1
            self.ages[layer][e] = self.clock
        if not misses:
            return
        assignments = []
        for expert in misses:
            slot = deterministic_lru_slot(
                self.logical[layer], self.ages[layer], frozenset(requested)
            )
            previous = self.logical[layer][slot]
            if previous is not None:
                del self.lookup[layer][previous]
            self.logical[layer][slot] = expert
            self.lookup[layer][expert] = slot
            assignments.append((expert, slot))
        store = self.stores[layer]
        if self.staging is not None:
            # The CPU must not overwrite a staging tensor while DMA still reads it.
            self.copy_stream.synchronize()
            for i, (expert, _) in enumerate(assignments):
                self.staging[0][i].copy_(store.gate_up[expert])
                self.staging[1][i].copy_(store.down[expert])
            c["cpu_staging_bytes"] += len(assignments) * store.expert_bytes
        current = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self.copy_stream):
            if self.has_compute[layer]:
                self.copy_stream.wait_event(self.done[layer])
            start = end = None
            if self.mode == "profile":
                start, end = (
                    torch.cuda.Event(enable_timing=True),  # type: ignore[no-untyped-call]
                    torch.cuda.Event(enable_timing=True),  # type: ignore[no-untyped-call]
                )
                start.record()  # type: ignore[no-untyped-call]
            for i, (expert, slot) in enumerate(assignments):
                gu = store.gate_up[expert] if self.staging is None else self.staging[0][i]
                dw = store.down[expert] if self.staging is None else self.staging[1][i]
                self.weights[layer][0][slot].copy_(gu, non_blocking=True)
                self.weights[layer][1][slot].copy_(dw, non_blocking=True)
            if end is not None:
                end.record()  # type: ignore[no-untyped-call]
                self.events.append((self.phase, start, end))
            self.ready[layer].record()  # type: ignore[no-untyped-call]
        self.has_copy[layer] = True
        self.versions[layer] += 1
        # Mapping mutations are enqueued on the compute stream AFTER all previous readers.
        host_map = [-1] * self.maps[layer].numel()
        for e, s in self.lookup[layer].items():
            host_map[e] = s
        self.maps[layer].copy_(torch.tensor(host_map, device=self.device))
        if not asynchronous:
            current.wait_event(self.ready[layer])
        c["h2d_bytes"] += len(assignments) * store.expert_bytes
        c["transfer_batches"] += 1
        c["async_transfer_batches"] += int(asynchronous)
        if self.mode == "audit":
            self.trace.append(
                {
                    "layer": layer,
                    "version": self.versions[layer],
                    "phase": self.phase,
                    "slot_to_expert": list(self.logical[layer]),
                }
            )

    def execute(
        self, layer: int, x: Tensor, ids: Tensor, weights: Tensor, *, resident: bool = False
    ) -> Tensor:
        if resident:
            groups: list[tuple[int, ...] | None] = [None]
        else:
            # Compact selected IDs are the only routing data required by the CPU miss scheduler.
            selected = ids.detach().cpu().flatten().tolist()
            used = sorted(set(selected))
            groups = [tuple(used[i : i + self.slots]) for i in range(0, len(used), self.slots)]
        output = None
        for group in groups:
            if group is not None:
                self.load(layer, group)
            current = torch.cuda.current_stream(self.device)
            if self.has_copy[layer]:
                current.wait_event(self.ready[layer])
            slot_ids = self.maps[layer][ids]
            selected_weights = weights
            if len(groups) > 1:
                mask = torch.zeros(self.maps[layer].numel(), dtype=torch.bool, device=self.device)
                mask[list(group or ())] = True
                selected_weights = weights * mask[ids]
            value = self.backend.forward(
                x.contiguous(),
                slot_ids,
                selected_weights,
                *self.weights[layer],
                {"error": self.error},
            )
            # One completion per whole group, recorded AFTER all slot reads.
            self.done[layer].record(current)  # type: ignore[no-untyped-call]
            self.has_compute[layer] = True
            self.counts[self.phase]["backend_calls"] += 1
            if len(groups) == 1:
                return value
            output = value.float() if output is None else output + value.float()
        if output is None:
            raise RuntimeError("empty route group")
        return output.to(x.dtype)

    def drain(self) -> None:
        torch.cuda.synchronize(self.device)
        if bool(self.error.item()):
            raise RuntimeError("nonzero production/pseudo contribution referenced a missing slot")

    def metrics(self) -> dict[str, Any]:
        # Caller drains and records wall end BEFORE entering this reporting function.
        return {
            "phases": {k: dict(v) for k, v in self.counts.items()},
            "transfer_ms": sum(s.elapsed_time(e) for _, s, e in self.events)
            if self.mode == "profile"
            else None,
            "h2d_bytes": sum(v["h2d_bytes"] for v in self.counts.values()),
            "host_mode": self.host_mode,
            "slots_per_layer": self.slots,
            "gpu_expert_capacity_bytes": sum(s.expert_bytes * self.slots for s in self.stores),
            "host_expert_bytes": sum(s.expert_bytes * s.gate_up.shape[0] for s in self.stores),
            "slot_version_trace": self.trace if self.mode == "audit" else None,
        }
