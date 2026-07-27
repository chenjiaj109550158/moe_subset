"""Sample-bounded offline oracle windows."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor


@dataclass(frozen=True)
class OracleSample:
    sample_id: str
    token_ids: Tensor
    topk_ids: Tensor
    topk_weights: Tensor
    router_logits: Tensor | None

    def __post_init__(self) -> None:
        token_count = int(self.token_ids.numel())
        if self.topk_ids.ndim != 3 or self.topk_ids.shape[0] != token_count:
            raise ValueError("topk_ids must have shape [tokens, layers, top_k]")
        if self.topk_weights.shape != self.topk_ids.shape:
            raise ValueError("topk weights must match topk IDs")
        if self.router_logits is not None and (
            self.router_logits.ndim != 3 or self.router_logits.shape[:2] != self.topk_ids.shape[:2]
        ):
            raise ValueError("router_logits must have shape [tokens, layers, experts]")


@dataclass(frozen=True)
class OracleWindow:
    sample_id: str
    start: int
    horizon: int
    topk_ids: Tensor
    topk_weights: Tensor
    router_logits: Tensor | None

    @property
    def end(self) -> int:
        return self.start + self.horizon


def iter_windows(
    samples: tuple[OracleSample, ...],
    horizons: tuple[int, ...],
    *,
    stride: int = 1,
) -> Iterator[OracleWindow]:
    if stride < 1:
        raise ValueError("stride must be positive")
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must be positive")
    for sample in samples:
        token_count = int(sample.token_ids.numel())
        for horizon in horizons:
            for start in range(0, token_count - horizon + 1, stride):
                end = start + horizon
                yield OracleWindow(
                    sample_id=sample.sample_id,
                    start=start,
                    horizon=horizon,
                    topk_ids=sample.topk_ids[start:end],
                    topk_weights=sample.topk_weights[start:end],
                    router_logits=(
                        sample.router_logits[start:end]
                        if sample.router_logits is not None
                        else None
                    ),
                )


def load_trace_samples(root: Path) -> tuple[OracleSample, ...]:
    """Load sample-preserving oracle tensors from a validated M1 trace store."""
    from safetensors.torch import load_file

    from pseudoroute.tracing.validation import validate_trace

    manifest = validate_trace(root)
    if manifest.trace_level != "router_logits":
        raise ValueError("full M2 sweep requires a router_logits trace")
    samples = []
    for shard in manifest.shards:
        tensors = load_file(str(root / shard.path))
        samples.append(
            OracleSample(
                sample_id=shard.sample_id,
                token_ids=tensors["token_ids"],
                topk_ids=tensors["router_topk_ids"],
                topk_weights=tensors["router_topk_weights"],
                router_logits=tensors["router_logits"],
            )
        )
    return tuple(samples)
