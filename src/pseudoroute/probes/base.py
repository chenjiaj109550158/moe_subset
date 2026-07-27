"""Leakage-safe deployable future-routing probe contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from pseudoroute.types import DeployableDecodeState, InformationRegime, require_deployable_state


@dataclass(frozen=True)
class ProbeOutput:
    horizons: tuple[int, ...]
    per_horizon_logits: dict[int, Tensor] | None
    per_horizon_probs: dict[int, Tensor] | None
    aggregate_utility: dict[int, Tensor]
    uncertainty: dict[int, Tensor] | None
    estimated_cost_ms: float
    metadata: dict[str, Any]


class FutureRoutingProbe(ABC):
    information_regime = InformationRegime.ONLINE_PRE_SAMPLE

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def requires_training(self) -> bool: ...

    def predict(self, state: DeployableDecodeState, horizons: Sequence[int]) -> ProbeOutput:
        checked = require_deployable_state(state)
        requested = tuple(int(value) for value in horizons)
        if not requested or any(value < 1 for value in requested):
            raise ValueError("probe horizons must be positive")
        return self._predict(checked, requested)

    @abstractmethod
    def _predict(self, state: DeployableDecodeState, horizons: tuple[int, ...]) -> ProbeOutput: ...
