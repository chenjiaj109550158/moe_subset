"""Native nine-position Qwen bridge/pseudo execution with real expert offload."""

from __future__ import annotations

import copy
import hashlib
import types
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from pseudoroute.benchmark.prefetch import (
    NativeRoute,
    Qwen3MoePrefetchOps,
    SubsetRouteRecord,
    masked_route_from_natural,
)
from pseudoroute.benchmark.qwen_pseudo import mass_preserving_subset_route
from pseudoroute.benchmark.subset_closed_loop import (
    _cache_layers,
    _cache_length,
    _cache_mutation_signature,
    _rng_state,
)
from pseudoroute.runtime.qwen_offload import (
    QwenExpertOffloadEngine,
    nonzero_expert_ids,
)


@dataclass
class BridgeOnlyCacheFork:
    cache: object
    production_length: int
    joint_tokens: int
    layer_update_counts: list[int]
    layer_query_lengths: list[int]
    committed: bool = False


@dataclass(frozen=True)
class JointPenultimateForwardResult:
    bridge_logits: Tensor
    bridge_records: tuple[SubsetRouteRecord, ...]
    next_subsets: dict[int, tuple[int, ...]]
    joint_input_token_ids: tuple[int, ...]
    production_cache_length_before: int
    production_cache_length_after: int
    pseudo_kv_positions_committed: int
    cache_layer_updates: tuple[int, ...]
    cache_layer_query_lengths: tuple[int, ...]
    attention_calls: int
    router_calls: int
    expert_calls: int
    async_prefetch_layers: int
    production_rng_unchanged: bool
    production_cache_unchanged_before_commit: bool
    pseudo_topk_ids_sha256: str


def _cpu_route(route: NativeRoute) -> NativeRoute:
    return NativeRoute(
        route.logits.detach().cpu(),
        route.weights.detach().cpu(),
        route.ids.detach().cpu(),
    )


def _slice_route(route: NativeRoute, start: int, stop: int) -> NativeRoute:
    return NativeRoute(
        route.logits[start:stop],
        route.weights[start:stop],
        route.ids[start:stop],
    )


def first_four_history_subset(
    pseudo_probabilities: Tensor,
    pseudo_topk_ids: Tensor,
    completed_window_history: Tensor,
    *,
    budget: int,
) -> tuple[int, ...]:
    """Select the frozen first-four core plus completed-window history fill."""
    if (
        pseudo_probabilities.ndim != 2
        or pseudo_topk_ids.ndim != 2
        or pseudo_probabilities.shape[0] != 8
        or pseudo_topk_ids.shape[0] != 8
        or completed_window_history.ndim != 1
        or pseudo_probabilities.shape[1] != completed_window_history.numel()
    ):
        raise ValueError("joint subset selection received incompatible route tensors")
    if not 8 <= budget <= completed_window_history.numel():
        raise ValueError("joint subset budget is outside expert bounds")
    probabilities = pseudo_probabilities.double().cpu()
    ids = pseudo_topk_ids.long().cpu()
    history = completed_window_history.double().cpu()
    union = {int(expert) for anchor in range(4) for expert in ids[anchor].tolist()}
    first_four = probabilities[:4].sum(dim=0)
    core = tuple(sorted(union, key=lambda expert: (-float(first_four[expert]), expert))[:budget])
    blocked = set(core)
    fill = tuple(
        sorted(
            (expert for expert in range(history.numel()) if expert not in blocked),
            key=lambda expert: (-float(history[expert]), expert),
        )[: budget - len(core)]
    )
    selected = tuple(sorted((*core, *fill)))
    if len(selected) != budget or len(set(selected)) != budget:
        raise RuntimeError("joint selector did not produce exactly B unique experts")
    return selected


def fork_bridge_only_cache(production_cache: object, *, joint_tokens: int) -> BridgeOnlyCacheFork:
    """Fork a populated dynamic cache that returns joint KV but stores bridge KV only."""
    if joint_tokens < 2:
        raise ValueError("joint cache requires one bridge and at least one pseudo token")
    production_layers = _cache_layers(production_cache)
    production_length = _cache_length(production_cache)
    fork = copy.copy(production_cache)
    fork_layers = [copy.copy(layer) for layer in production_layers]
    update_counts = [0] * len(fork_layers)
    query_lengths = [0] * len(fork_layers)
    for layer_index, layer in enumerate(fork_layers):
        if getattr(layer, "sliding_window", None) is not None:
            raise ValueError("joint bridge-only commit does not support sliding cache layers")
        if not isinstance(getattr(layer, "keys", None), Tensor) or not isinstance(
            getattr(layer, "values", None), Tensor
        ):
            raise RuntimeError("joint bridge-only commit requires initialized dynamic KV layers")

        def bridge_only_update(
            self: Any,
            key_states: Tensor,
            value_states: Tensor,
            *args: object,
            _layer_index: int = layer_index,
            **kwargs: object,
        ) -> tuple[Tensor, Tensor]:
            del args, kwargs
            if update_counts[_layer_index] != 0:
                raise RuntimeError("joint cache layer was updated more than once")
            query_length = int(key_states.shape[-2])
            if query_length != joint_tokens or value_states.shape[-2] != joint_tokens:
                raise RuntimeError("joint cache layer received the wrong query length")
            previous_keys = cast(Tensor, self.keys)
            previous_values = cast(Tensor, self.values)
            if (
                previous_keys.shape[-2] != production_length
                or previous_values.shape[-2] != production_length
            ):
                raise RuntimeError("joint cache fork changed its production prefix")
            full_keys = torch.cat((previous_keys, key_states), dim=-2)
            full_values = torch.cat((previous_values, value_states), dim=-2)
            self.keys = torch.cat((previous_keys, key_states[..., :1, :]), dim=-2)
            self.values = torch.cat((previous_values, value_states[..., :1, :]), dim=-2)
            update_counts[_layer_index] += 1
            query_lengths[_layer_index] = query_length
            return full_keys, full_values

        layer.update = types.MethodType(bridge_only_update, layer)
    cast(Any, fork).layers = fork_layers
    if _cache_length(fork) != production_length:
        raise RuntimeError("joint bridge-only cache fork changed sequence length")
    return BridgeOnlyCacheFork(
        cache=fork,
        production_length=production_length,
        joint_tokens=joint_tokens,
        layer_update_counts=update_counts,
        layer_query_lengths=query_lengths,
    )


def commit_bridge_only_cache(
    production_cache: object,
    fork: BridgeOnlyCacheFork,
) -> None:
    """Commit exactly the bridge KV tensors while preserving production layer objects."""
    if fork.committed:
        raise RuntimeError("joint bridge cache was committed more than once")
    production_layers = _cache_layers(production_cache)
    fork_layers = _cache_layers(fork.cache)
    if len(production_layers) != len(fork_layers) or any(
        count != 1 for count in fork.layer_update_counts
    ):
        raise RuntimeError("joint bridge cache did not update every layer exactly once")
    if _cache_length(production_cache) != fork.production_length:
        raise RuntimeError("production cache changed before joint bridge commit")
    if _cache_length(fork.cache) != fork.production_length + 1:
        raise RuntimeError("joint bridge cache did not retain exactly one new position")
    for production_layer, fork_layer in zip(
        production_layers,
        fork_layers,
        strict=True,
    ):
        production_layer.keys = fork_layer.keys
        production_layer.values = fork_layer.values
        if hasattr(production_layer, "cumulative_length") and hasattr(
            fork_layer, "cumulative_length"
        ):
            production_layer.cumulative_length = fork_layer.cumulative_length
    if _cache_length(production_cache) != fork.production_length + 1:
        raise RuntimeError("production cache did not advance by exactly one bridge position")
    fork.committed = True


class JointPenultimateMlpContext(AbstractContextManager["JointPenultimateMlpContext"]):
    """Execute bridge hard routing and pseudo intersection routing in one MLP call."""

    def __init__(
        self,
        ops: Qwen3MoePrefetchOps,
        engine: QwenExpertOffloadEngine,
        *,
        active_subsets: dict[int, tuple[int, ...]],
        history_before_bridge: dict[int, Tensor],
        budget: int,
        horizon: int,
    ) -> None:
        if horizon != 8:
            raise ValueError("joint penultimate v1 requires H=8")
        expected_layers = set(range(ops.num_layers))
        if set(active_subsets) != expected_layers or set(history_before_bridge) != expected_layers:
            raise ValueError("joint penultimate state must cover every routed layer")
        if any(
            len(subset) != budget or len(set(subset)) != budget
            for subset in active_subsets.values()
        ):
            raise ValueError("joint penultimate active subsets must contain exactly B experts")
        self.ops = ops
        self.engine = engine
        self.active_subsets = dict(active_subsets)
        self.history_before_bridge = history_before_bridge
        self.budget = budget
        self.horizon = horizon
        self._original: list[Any] = []
        self._bridge_records: dict[int, SubsetRouteRecord] = {}
        self._next_subsets: dict[int, tuple[int, ...]] = {}
        self._pseudo_digest = hashlib.sha256()
        self.router_calls = 0
        self.expert_calls = 0
        self.async_prefetch_layers = 0

    def _forward(self, layer: int, hidden_states: Tensor) -> Tensor:
        shape = hidden_states.shape
        if tuple(shape[:2]) != (1, self.horizon + 1):
            raise RuntimeError("joint MLP requires one bridge plus eight pseudo positions")
        flat = hidden_states.reshape(-1, shape[-1])
        natural = self.ops.route(layer, flat)
        self.router_calls += 1
        bridge_natural = _slice_route(natural, 0, 1)
        pseudo_natural = _slice_route(natural, 1, self.horizon + 1)
        active = self.active_subsets[layer]
        bridge_executed = masked_route_from_natural(self.ops, bridge_natural, active)
        pseudo_executed = mass_preserving_subset_route(
            self.ops,
            pseudo_natural,
            active,
            "natural_top8_intersection_zero_missing",
        )
        executed = NativeRoute(
            natural.logits,
            torch.cat((bridge_executed.weights, pseudo_executed.weights), dim=0),
            torch.cat((bridge_executed.ids, pseudo_executed.ids), dim=0),
        )
        used = nonzero_expert_ids(executed.ids, executed.weights)
        self.engine.require_resident(layer, used)
        with self.engine.phase("joint_compute"):
            contribution = self.ops.experts(layer, flat, executed)
        self.expert_calls += 1

        bridge_cpu = _cpu_route(bridge_natural)
        executed_cpu = _cpu_route(bridge_executed)
        self._bridge_records[layer] = SubsetRouteRecord(
            layer=layer,
            natural=bridge_cpu,
            executed=executed_cpu,
            allowed=active,
        )
        completed_history = self.history_before_bridge[layer].double().cpu().clone()
        completed_history.scatter_add_(
            0,
            bridge_cpu.ids.reshape(-1).long(),
            bridge_cpu.weights.reshape(-1).double(),
        )
        pseudo_probabilities = pseudo_natural.logits.float().softmax(dim=-1).detach().cpu()
        pseudo_ids = pseudo_natural.ids.detach().cpu()
        next_subset = first_four_history_subset(
            pseudo_probabilities,
            pseudo_ids,
            completed_history,
            budget=self.budget,
        )
        self._next_subsets[layer] = next_subset
        self._pseudo_digest.update(layer.to_bytes(4, "little"))
        self._pseudo_digest.update(pseudo_ids.contiguous().numpy().tobytes())
        with self.engine.phase("joint_prefetch"):
            self.engine.preload_layer_subset_async(layer, next_subset)
        self.async_prefetch_layers += 1
        return contribution.reshape(shape)

    @property
    def bridge_records(self) -> tuple[SubsetRouteRecord, ...]:
        if set(self._bridge_records) != set(range(self.ops.num_layers)):
            raise RuntimeError("joint bridge records are incomplete")
        return tuple(self._bridge_records[layer] for layer in range(self.ops.num_layers))

    @property
    def next_subsets(self) -> dict[int, tuple[int, ...]]:
        if set(self._next_subsets) != set(range(self.ops.num_layers)):
            raise RuntimeError("joint next subsets are incomplete")
        return dict(self._next_subsets)

    @property
    def pseudo_topk_ids_sha256(self) -> str:
        return self._pseudo_digest.hexdigest()

    def __enter__(self) -> JointPenultimateMlpContext:
        for layer_index in range(self.ops.num_layers):
            mlp = self.ops.mlp(layer_index)
            self._original.append(cast(Any, mlp).forward)

            def replacement(
                _module: nn.Module,
                hidden_states: Tensor,
                *,
                layer: int = layer_index,
            ) -> Tensor:
                return self._forward(layer, hidden_states)

            cast(Any, mlp).forward = types.MethodType(replacement, mlp)
        return self

    def __exit__(self, *exc: object) -> None:
        for layer_index, original in enumerate(self._original):
            cast(Any, self.ops.mlp(layer_index)).forward = original
        self._original.clear()


@torch.inference_mode()
def joint_penultimate_forward(
    model: nn.Module,
    ops: Qwen3MoePrefetchOps,
    engine: QwenExpertOffloadEngine,
    *,
    current: Tensor,
    anchor_token_ids: tuple[int, ...],
    production_cache: object,
    active_subsets: dict[int, tuple[int, ...]],
    history_before_bridge: dict[int, Tensor],
    budget: int,
) -> JointPenultimateForwardResult:
    """Run and commit one real bridge plus eight disposable pseudo positions."""
    if tuple(current.shape) != (1, 1) or len(anchor_token_ids) != 8:
        raise ValueError("joint penultimate forward requires [1,1] bridge and eight anchors")
    bridge_token_id = int(current.item())
    if anchor_token_ids[0] != bridge_token_id:
        raise ValueError("joint pseudo content is not seeded by the bridge token")
    joint_ids = torch.tensor(
        [[bridge_token_id, *anchor_token_ids]],
        dtype=torch.long,
        device=current.device,
    )
    production_length = _cache_length(production_cache)
    production_signature = _cache_mutation_signature(production_cache)
    rng_before = _rng_state(current.device)
    cache_fork = fork_bridge_only_cache(production_cache, joint_tokens=joint_ids.shape[1])
    context = JointPenultimateMlpContext(
        ops,
        engine,
        active_subsets=active_subsets,
        history_before_bridge=history_before_bridge,
        budget=budget,
        horizon=8,
    )
    with context:
        output = cast(
            Any,
            model(
                input_ids=joint_ids,
                past_key_values=cache_fork.cache,
                use_cache=True,
                return_dict=True,
            ),
        )
    if output.past_key_values is not cache_fork.cache:
        raise RuntimeError("joint Qwen forward replaced its bridge-only cache object")
    cache_unchanged = (
        _cache_length(production_cache) == production_length
        and _cache_mutation_signature(production_cache) == production_signature
    )
    if not cache_unchanged:
        raise RuntimeError("joint Qwen forward mutated production cache before commit")
    rng_after = _rng_state(current.device)
    rng_unchanged = torch.equal(rng_before[0], rng_after[0]) and torch.equal(
        rng_before[1], rng_after[1]
    )
    if not rng_unchanged:
        raise RuntimeError("joint Qwen forward changed generation RNG")
    logits = cast(Tensor, output.logits)
    if tuple(logits.shape[:2]) != (1, 9):
        raise RuntimeError("joint Qwen forward did not return nine vocabulary positions")
    bridge_logits = logits[:, 0].detach().clone()
    commit_bridge_only_cache(production_cache, cache_fork)
    return JointPenultimateForwardResult(
        bridge_logits=bridge_logits,
        bridge_records=context.bridge_records,
        next_subsets=context.next_subsets,
        joint_input_token_ids=tuple(int(value) for value in joint_ids[0].tolist()),
        production_cache_length_before=production_length,
        production_cache_length_after=_cache_length(production_cache),
        pseudo_kv_positions_committed=0,
        cache_layer_updates=tuple(cache_fork.layer_update_counts),
        cache_layer_query_lengths=tuple(cache_fork.layer_query_lengths),
        attention_calls=ops.num_layers,
        router_calls=context.router_calls,
        expert_calls=context.expert_calls,
        async_prefetch_layers=context.async_prefetch_layers,
        production_rng_unchanged=rng_unchanged,
        production_cache_unchanged_before_commit=cache_unchanged,
        pseudo_topk_ids_sha256=context.pseudo_topk_ids_sha256,
    )
