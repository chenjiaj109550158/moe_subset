"""Architecture-neutral MoE adapter contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from torch import Tensor, nn

from pseudoroute.execution.routing_policy import ExecutedRoute, RoutingPolicy
from pseudoroute.types import ExpertKey, ModelSpec, RouterTrace


class TraceLevel(StrEnum):
    ROUTES_ONLY = "routes_only"
    ROUTER_LOGITS = "router_logits"


@dataclass(frozen=True)
class TraceRequest:
    level: TraceLevel = TraceLevel.ROUTER_LOGITS


@dataclass(frozen=True)
class RouteResult:
    raw_logits: Tensor
    pre_topk_scores: Tensor
    topk_ids: Tensor
    topk_weights: Tensor


@dataclass(frozen=True)
class ForwardResult:
    logits: Tensor
    traces: tuple[RouterTrace, ...]
    kv_cache: object | None = None
    executed_routes: tuple[ExecutedRoute, ...] = ()


@dataclass(frozen=True)
class MoELayerHandle:
    layer_idx: int
    router: nn.Module
    experts: tuple[nn.Module, ...]
    top_k: int
    normalization: nn.Module | None
    combine_semantics: str
    shared_expert: nn.Module | None
    score_function: str
    has_router_bias: bool
    shared_expert_count: int = 0


class MoEModelAdapter(ABC):
    @property
    @abstractmethod
    def spec(self) -> ModelSpec: ...

    @abstractmethod
    def validate_structure(self) -> None: ...

    @abstractmethod
    def iter_moe_layers(self) -> tuple[MoELayerHandle, ...]: ...

    @abstractmethod
    def run_base_forward(
        self,
        input_ids: Tensor,
        *,
        position_ids: Tensor | None = None,
        kv_cache: object | None = None,
        use_cache: bool = False,
        trace_request: TraceRequest | None = None,
    ) -> ForwardResult: ...

    @abstractmethod
    def route_from_state(self, layer_idx: int, router_input: Tensor) -> RouteResult: ...

    @abstractmethod
    def forward_with_policy(
        self,
        input_ids: Tensor,
        policy: RoutingPolicy,
        *,
        allowed_experts: frozenset[ExpertKey] | None = None,
        allowed_experts_by_position: dict[int, frozenset[ExpertKey]] | None = None,
        kv_cache: object | None = None,
        use_cache: bool = False,
    ) -> ForwardResult: ...

    @abstractmethod
    def clone_kv_cache_for_shadow(self, kv_cache: object) -> object: ...

    @abstractmethod
    def get_expert_handle(self, key: ExpertKey) -> nn.Module: ...
