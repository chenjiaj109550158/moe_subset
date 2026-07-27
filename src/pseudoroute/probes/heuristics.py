"""Training-free and transition-based M6 deployable probes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pseudoroute.probes.base import FutureRoutingProbe, ProbeOutput
from pseudoroute.types import DeployableDecodeState


def _selected_vector(ids: Tensor, weights: Tensor, num_experts: int) -> Tensor:
    vector = torch.zeros(num_experts, dtype=torch.float64, device=weights.device)
    vector.scatter_add_(0, ids.reshape(-1).long(), weights.reshape(-1).double())
    return vector


def _output(name: str, horizons: tuple[int, ...], values: dict[int, Tensor]) -> ProbeOutput:
    return ProbeOutput(
        horizons, None, None, values, None, 0.0, {"probe": name, "online_only": True}
    )


@dataclass(frozen=True)
class CurrentRouteProbe(FutureRoutingProbe):
    @property
    def name(self) -> str:
        return "current_route"

    @property
    def requires_training(self) -> bool:
        return False

    def _predict(self, state: DeployableDecodeState, horizons: tuple[int, ...]) -> ProbeOutput:
        horizon = max(horizons)
        utilities = {}
        for layer, logits in enumerate(state.current_router_logits):
            probabilities = logits.double().softmax(dim=-1).reshape(-1)
            utilities[layer] = probabilities * horizon
        return _output(self.name, horizons, utilities)


@dataclass(frozen=True)
class RollingFrequencyProbe(FutureRoutingProbe):
    lookback: int
    decay: float = 1.0

    @property
    def name(self) -> str:
        return "rolling_frequency"

    @property
    def requires_training(self) -> bool:
        return False

    def _predict(self, state: DeployableDecodeState, horizons: tuple[int, ...]) -> ProbeOutput:
        horizon = max(horizons)
        utilities = {}
        history = state.router_history_topk[-self.lookback :]
        for layer, logits in enumerate(state.current_router_logits):
            score = torch.zeros(logits.shape[-1], dtype=torch.float64, device=logits.device)
            for age, ids_by_layer in enumerate(reversed(history)):
                ids = ids_by_layer[layer].reshape(-1).long().to(logits.device)
                score.scatter_add_(
                    0, ids, torch.full_like(ids, self.decay**age, dtype=torch.float64)
                )
            utilities[layer] = score / score.sum().clamp_min(1e-30) * horizon
        return _output(self.name, horizons, utilities)


@dataclass(frozen=True)
class RollingMassProbe(FutureRoutingProbe):
    lookback: int
    decay: float = 1.0

    @property
    def name(self) -> str:
        return "rolling_mass"

    @property
    def requires_training(self) -> bool:
        return False

    def _predict(self, state: DeployableDecodeState, horizons: tuple[int, ...]) -> ProbeOutput:
        horizon = max(horizons)
        utilities = {}
        pairs = list(zip(state.router_history_topk, state.router_history_weights, strict=True))
        for layer, logits in enumerate(state.current_router_logits):
            score = torch.zeros(logits.shape[-1], dtype=torch.float64, device=logits.device)
            for age, (ids_by_layer, weights_by_layer) in enumerate(
                reversed(pairs[-self.lookback :])
            ):
                score += self.decay**age * _selected_vector(
                    ids_by_layer[layer].to(logits.device),
                    weights_by_layer[layer].to(logits.device),
                    logits.shape[-1],
                )
            utilities[layer] = score / score.sum().clamp_min(1e-30) * horizon
        return _output(self.name, horizons, utilities)


@dataclass(frozen=True)
class MarkovTransitionProbe(FutureRoutingProbe):
    transitions: Tensor  # [layers, experts, experts], row stochastic

    @property
    def name(self) -> str:
        return "markov_transition"

    @property
    def requires_training(self) -> bool:
        return True

    def _predict(self, state: DeployableDecodeState, horizons: tuple[int, ...]) -> ProbeOutput:
        if not state.router_history_topk:
            raise ValueError("Markov probe requires route history")
        horizon = max(horizons)
        utilities = {}
        latest_ids = state.router_history_topk[-1]
        latest_weights = state.router_history_weights[-1]
        for layer, logits in enumerate(state.current_router_logits):
            distribution = _selected_vector(
                latest_ids[layer].to(logits.device),
                latest_weights[layer].to(logits.device),
                logits.shape[-1],
            )
            distribution /= distribution.sum().clamp_min(1e-30)
            transition = self.transitions[layer].to(logits.device).double()
            aggregate = torch.zeros_like(distribution)
            for _ in range(horizon):
                distribution = distribution @ transition
                aggregate += distribution
            utilities[layer] = aggregate
        return _output(self.name, horizons, utilities)
