"""CPU-first sharded safetensors loader and persistent MoE policy hooks."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pseudoroute.restart.backend import MoEComputeBackend
from pseudoroute.restart.policies import cost_aware_subset, legacy_subset, top_ids
from pseudoroute.restart.residency import HostLayer, ResidencyManager


class PolicyRuntime:
    def __init__(
        self,
        manager: ResidencyManager,
        experts: int,
        top_k: int,
        *,
        d: int = 8,
        mode: str = "performance",
    ) -> None:
        self.manager, self.experts, self.k = manager, experts, top_k
        self.d, self.mode = d, mode
        self.reset("E")

    def reset(self, policy: str) -> None:
        if policy not in ("E", "S", "H", "HC", "P8", "P4", "PW4", "PM4"):
            raise ValueError("unknown policy")
        self.policy, self.kind = policy, "prefill"
        self.history = [
            torch.zeros(self.experts, device=self.manager.device) for _ in self.manager.stores
        ]
        self.window = [torch.zeros_like(h) for h in self.history]
        self.prompt_tail = [torch.zeros_like(h) for h in self.history]
        self.prompt_tail_routes: list[list[Tensor]] = [[] for _ in self.history]
        self.prompt_counts = [0] * len(self.history)
        self.active: dict[int, tuple[int, ...]] = {}
        self.keep: dict[int, Tensor] = {}
        self.pseudo: dict[int, tuple[Tensor, Tensor]] = {}
        self.next_subsets: dict[int, tuple[int, ...]] = {}
        self.audit_routes: list[Any] = []
        self.routing_diagnostics = torch.zeros((len(self.history), 3), device=self.manager.device)
        self.boundary = False
        self.capture_fixed_routes = False
        self.fixed_routes: dict[int, tuple[Tensor, Tensor]] | None = None
        self.captured_routes: dict[int, tuple[Tensor, Tensor]] = {}

    def set_subset(
        self, layer: int, subset: tuple[int, ...], *, asynchronous: bool = False
    ) -> None:
        self.manager.load(layer, subset, asynchronous=asynchronous)
        self.active[layer] = subset
        mask = torch.zeros(self.experts, device=self.manager.device, dtype=torch.bool)
        mask[list(subset)] = True
        self.keep[layer] = mask

    def select(self, layer: int, *, bootstrap: bool = False) -> tuple[int, ...]:
        budget = self.manager.slots
        h = self.history[layer]
        if self.policy in ("S", "H"):
            return top_ids(h, budget)
        if self.policy in ("HC", "PW4"):
            p = self.pseudo[layer][0] if self.policy == "PW4" else None
            return cost_aware_subset(
                h,
                self.active.get(layer, ()),
                torch.full_like(h, float(self.manager.stores[layer].expert_bytes)),
                budget=budget,
                d=self.d,
                pseudo_probabilities=p,
            )
        p, ids = self.pseudo[layer]
        return legacy_subset(
            p, ids, self.prompt_tail[layer] if bootstrap else self.window[layer], budget
        )

    def initialize(self) -> None:
        if self.policy == "E":
            return
        self.manager.phase = "bootstrap_residency"
        for layer in range(len(self.history)):
            self.set_subset(layer, self.select(layer, bootstrap=True))
            self.window[layer].zero_()

    def after_boundary(self) -> None:
        if self.policy in ("H", "HC"):
            self.manager.phase = "boundary_h2d"
            for layer in range(len(self.history)):
                self.set_subset(layer, self.select(layer), asynchronous=True)
        for h in self.window:
            h.zero_()

    def forward(self, layer: int, gate: nn.Module, hidden: Tensor) -> Tensor:
        x = hidden.reshape(-1, hidden.shape[-1])
        raw = gate(x)
        if isinstance(raw, tuple):
            logits, natural_weights, natural_ids = raw
        else:
            logits = raw
            natural_weights, natural_ids = F.softmax(logits, dim=-1, dtype=torch.float32).topk(
                self.k, dim=-1
            )
            natural_weights = (natural_weights / natural_weights.sum(-1, keepdim=True)).to(x.dtype)
        # Exact demand LRU has no predictor state; do not weaken it with unused EMA work.
        if self.policy != "E" and self.kind == "prefill":
            mass = torch.zeros((x.shape[0], self.experts), device=x.device)
            mass.scatter_add_(1, natural_ids, natural_weights.float())
            n = self.prompt_counts[layer]
            self.history[layer].mul_(n).add_(mass.sum(0)).div_(n + x.shape[0])
            self.prompt_counts[layer] += x.shape[0]
            if self.policy.startswith("P"):
                tail = self.prompt_tail_routes[layer]
                tail.append(mass[-8:].clone())
                joined = torch.cat(tail)[-8:]
                self.prompt_tail_routes[layer] = [joined]
                self.prompt_tail[layer] = joined.sum(0)
        elif self.policy not in ("E", "S") and self.kind in ("production", "joint"):
            mass = torch.zeros(self.experts, device=x.device)
            mass.scatter_add_(0, natural_ids[0], natural_weights[0].float())
            if self.policy in ("H", "HC", "PW4"):
                self.history[layer].mul_(0.8).add_(mass, alpha=0.2)
            if self.policy.startswith("P"):
                self.window[layer].add_(mass)
        if self.mode != "performance" and self.kind in ("production", "joint"):
            ready = self.manager.maps[layer][natural_ids[0]] >= 0
            self.routing_diagnostics[layer, 0].add_(ready.float().mean())
            self.routing_diagnostics[layer, 1].add_((natural_weights[0].float() * ready).sum())
            self.routing_diagnostics[layer, 2].add_(1)
        ids, weights = natural_ids, natural_weights
        resident = self.kind not in ("prefill", "bootstrap") and self.policy != "E"
        if resident:
            hard_p = F.softmax(logits.float().masked_fill(~self.keep[layer], -torch.inf), -1)
            hard_w, hard_ids = hard_p.topk(self.k, -1)
            hard_w = (hard_w / hard_w.sum(-1, keepdim=True)).to(x.dtype)
            if self.kind == "joint" and self.policy != "PM4":
                ids = torch.cat((hard_ids[:1], natural_ids[1:]))
                pseudo_w = natural_weights[1:] * self.keep[layer][natural_ids[1:]]
                weights = torch.cat((hard_w[:1], pseudo_w))
            else:
                ids, weights = hard_ids, hard_w
        if self.fixed_routes is not None:
            ids, weights = self.fixed_routes[layer]
        if self.capture_fixed_routes:
            self.captured_routes[layer] = (ids.clone(), weights.clone())
        if self.mode == "audit":
            self.audit_routes.append(
                {
                    "layer": layer,
                    "kind": self.kind,
                    "natural_ids": natural_ids.cpu().tolist(),
                    "executed_ids": ids.cpu().tolist(),
                }
            )
        value = self.manager.execute(layer, x, ids, weights, resident=resident)
        if self.kind in ("bootstrap", "joint"):
            start = int(self.kind == "joint")
            probabilities = F.softmax(logits[start:], dim=-1, dtype=torch.float32)
            self.pseudo[layer] = (probabilities.detach(), natural_ids[start:].detach())
        if self.kind == "joint":
            subset = self.select(layer)
            self.next_subsets[layer] = subset
            previous_phase = self.manager.phase
            self.manager.phase = "joint_prefetch"
            self.set_subset(layer, subset, asynchronous=True)
            self.manager.phase = previous_phase
        return value.reshape(hidden.shape)


class SlotMlp(nn.Module):
    def __init__(self, gate: nn.Module, runtime: PolicyRuntime, layer: int) -> None:
        super().__init__()
        self.gate, self.runtime, self.layer = gate, runtime, layer

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.runtime.forward(self.layer, self.gate, hidden_states)


def strict_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device unavailable: {requested}")
    torch.empty(1, device=device)
    return device


@torch.inference_mode()
def load_cpu_first(
    snapshot: str,
    *,
    slots: int = 32,
    backend: str = "vllm_fused",
    mode: str = "performance",
    device: str = "cuda:0",
    manifest_path: Path | None = None,
) -> tuple[Any, Any, PolicyRuntime]:
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    strict_device(device)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    expected = {
        "num_hidden_layers": 48,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "hidden_size": 2048,
        "moe_intermediate_size": 768,
    }
    if any(getattr(config, k) != v for k, v in expected.items()) or not config.norm_topk_prob:
        raise ValueError("checkpoint architecture does not match pinned adapter")
    started = time.perf_counter()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(  # type: ignore[no-untyped-call]
            config, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False
        )
    stores = []
    pin = True
    try:
        torch.empty(1024, dtype=torch.bfloat16, pin_memory=True)
    except RuntimeError:
        pin = False
    for _ in range(config.num_hidden_layers):
        stores.append(
            HostLayer(
                torch.empty(
                    (config.num_experts, 2 * config.moe_intermediate_size, config.hidden_size),
                    dtype=torch.bfloat16,
                    pin_memory=pin,
                ),
                torch.empty(
                    (config.num_experts, config.hidden_size, config.moe_intermediate_size),
                    dtype=torch.bfloat16,
                    pin_memory=pin,
                ),
            )
        )
    manager = ResidencyManager(
        stores, slots, MoEComputeBackend(backend, audit=mode == "audit"), device=device, mode=mode
    )
    runtime = PolicyRuntime(manager, config.num_experts, config.num_experts_per_tok, mode=mode)
    for i, layer in enumerate(model.model.layers):
        layer.mlp = SlotMlp(layer.mlp.gate, runtime, i)
    index = json.loads((Path(snapshot) / "model.safetensors.index.json").read_text())["weight_map"]
    loaded: set[str] = set()
    dense_expected = set(dict(model.named_parameters()))
    expert_pattern = re.compile(
        r"model.layers.(\d+).mlp.experts.(\d+).(gate_proj|up_proj|down_proj).weight"
    )
    dense_loaded: set[str] = set()
    tensors = []
    for shard in sorted(set(index.values())):
        with safe_open(str(Path(snapshot) / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in index or index[key] != shard or key in loaded:
                    raise ValueError(f"unexpected/duplicate checkpoint key {key}")
                t = f.get_tensor(key)
                if t.dtype != torch.bfloat16:
                    raise ValueError(f"unexpected dtype {key}: {t.dtype}")
                match = expert_pattern.fullmatch(key)
                if match:
                    layer, expert = int(match[1]), int(match[2])
                    part, store = match[3], stores[layer]
                    if part == "down_proj":
                        target = store.down[expert]
                    else:
                        offset = 0 if part == "gate_proj" else config.moe_intermediate_size
                        target = store.gate_up[
                            expert, offset : offset + config.moe_intermediate_size
                        ]
                    if target.shape != t.shape:
                        raise ValueError(f"expert shape mismatch {key}")
                    target.copy_(t)
                    tensors.append(
                        {
                            "key": key,
                            "layer": layer,
                            "expert": expert,
                            "shape": list(t.shape),
                            "dtype": str(t.dtype),
                            "shard": shard,
                            "sample": t.flatten()[[0, t.numel() // 2, t.numel() - 1]]
                            .float()
                            .tolist(),
                        }
                    )
                else:
                    if key not in dense_expected:
                        raise ValueError(f"unrecognized dense tensor {key}")
                    module_name, name = key.rsplit(".", 1)
                    module = model.get_submodule(module_name)
                    old = getattr(module, name)
                    if old.shape != t.shape:
                        raise ValueError(f"dense shape mismatch {key}")
                    setattr(module, name, nn.Parameter(t.to(device), requires_grad=False))
                    dense_loaded.add(key)
                loaded.add(key)
                del t
    if dense_loaded != dense_expected or set(index) != loaded:
        raise ValueError("unloaded checkpoint parameters")
    if len(tensors) != config.num_hidden_layers * config.num_experts * 3:
        raise ValueError("incomplete expert store")
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config, device=torch.device(device))
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("unloaded meta tensor remains")
    if any(".experts." in name for name, _ in model.named_parameters()):
        raise ValueError("full expert parameters were retained")
    model.generation_config = GenerationConfig.from_pretrained(snapshot, local_files_only=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    torch.cuda.synchronize()
    if manifest_path is not None:
        manifest_path.write_text(
            json.dumps(
                {
                    "state": "VERIFIED",
                    "layout": "checkpoint_unpacked_to_gate_then_up",
                    "dense_keys": sorted(dense_loaded),
                    "expert_tensors": tensors,
                    "load_wall_seconds": time.perf_counter() - started,
                    "host_mode": manager.host_mode,
                    "no_full_experts_on_gpu": True,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
                indent=2,
            )
        )
    return model, tokenizer, runtime
