"""Cross-cutting types that enforce information and routing boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class InformationRegime(StrEnum):
    ORACLE = "oracle"
    OFFLINE_TEACHER_FORCED = "offline_teacher_forced"
    ONLINE_PRE_SAMPLE = "online_pre_sample"
    ONLINE_POST_SAMPLE = "online_post_sample"


class MissPolicy(StrEnum):
    LOSSLESS_FALLBACK = "lossless_fallback"
    SUBSTITUTE = "substitute"
    TRUNCATE = "truncate"
    EARLY_TERMINATE_WINDOW = "early_terminate_window"
    CPU_EXECUTE = "cpu_execute"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    architecture: str
    num_layers: int
    moe_layer_indices: tuple[int, ...]
    num_experts_by_layer: dict[int, int]
    top_k_by_layer: dict[int, int]
    hidden_size: int
    uses_rope: bool
    pre_norm: bool
    expert_bytes: dict[ExpertKey, int]
    shared_experts_by_layer: dict[int, int] = field(default_factory=dict)
    routing_semantics_by_layer: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True, order=True)
class ExpertKey:
    layer_idx: int
    expert_idx: int

    def __post_init__(self) -> None:
        if self.layer_idx < 0 or self.expert_idx < 0:
            raise ValueError("expert coordinates must be non-negative")


@dataclass(frozen=True)
class RouterTrace:
    token_position: int
    layer_idx: int
    raw_logits: Tensor
    pre_topk_scores: Tensor
    topk_ids: Tensor
    topk_weights: Tensor

    @property
    def expert_keys(self) -> tuple[ExpertKey, ...]:
        return tuple(
            ExpertKey(self.layer_idx, int(expert_idx))
            for expert_idx in self.topk_ids.reshape(-1).tolist()
        )


@dataclass(frozen=True)
class OfflineFutureTrace:
    """Offline-only future data; never accepted by an online probe."""

    sample_id: str
    start_position: int
    token_ids: Tensor
    router_logits: Tensor


@dataclass(frozen=True)
class DeployableDecodeState:
    """Only information available at the declared online decision boundary."""

    information_regime: InformationRegime
    prefix_token_ids: Tensor
    absolute_position: int
    current_router_logits: tuple[Tensor, ...]
    resident_experts: frozenset[ExpertKey]
    next_token_id: int | None = None
    router_history_topk: tuple[Tensor, ...] = ()
    router_history_weights: tuple[Tensor, ...] = ()

    def __post_init__(self) -> None:
        if self.information_regime not in {
            InformationRegime.ONLINE_PRE_SAMPLE,
            InformationRegime.ONLINE_POST_SAMPLE,
        }:
            raise ValueError("DeployableDecodeState requires an online information regime")
        if (
            self.information_regime is InformationRegime.ONLINE_PRE_SAMPLE
            and self.next_token_id is not None
        ):
            raise ValueError("online_pre_sample cannot contain the next token")
        if len(self.router_history_topk) != len(self.router_history_weights):
            raise ValueError("route history IDs and weights must have equal lengths")


def require_deployable_state(value: object) -> DeployableDecodeState:
    """Reject offline future-bearing objects at every online probe entry point."""
    if not isinstance(value, DeployableDecodeState):
        raise TypeError("online probes require DeployableDecodeState")
    return value
