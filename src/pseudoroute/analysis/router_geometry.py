"""M5 router subspace, covariance, boundary, entropy, and margin analysis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RouterSVD:
    method: str
    u: Tensor
    singular_values: Tensor
    vh: Tensor

    @property
    def effective_rank(self) -> float:
        energy = self.singular_values.square()
        probabilities = energy / energy.sum().clamp_min(1e-30)
        return float(torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()))


@dataclass(frozen=True)
class CovarianceSketch:
    count: int
    mean: Tensor
    covariance: Tensor
    components: Tensor
    eigenvalues: Tensor


@dataclass(frozen=True)
class SVDMetricRow:
    layer_idx: int
    method: str
    rank: int
    singular_value: float
    cumulative_energy: float
    relative_reconstruction_error: float
    effective_rank: float


@dataclass(frozen=True)
class CoordinateRow:
    sample_id: str
    token_position: int
    layer_idx: int
    coordinate_idx: int
    value: float


@dataclass(frozen=True)
class ExpertGeometryRow:
    layer_idx: int
    expert_idx: int
    router_row_norm: float
    covariance_aware_logit_variance: float
    activation_frequency: float
    mean_router_probability: float


@dataclass(frozen=True)
class BoundaryRow:
    layer_idx: int
    expert_a: int
    expert_b: int
    mean_absolute_distance: float
    near_boundary_frequency: float
    coactivation_frequency: float


@dataclass(frozen=True)
class TokenGeometryRow:
    sample_id: str
    token_position: int
    layer_idx: int
    route_entropy: float
    top1_gap: float
    topk_margin: float
    cumulative_topb_mass: float


def exact_router_svd(weight: Tensor) -> RouterSVD:
    u, singular_values, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    return RouterSVD("exact", u, singular_values, vh)


def randomized_router_svd(
    weight: Tensor, *, rank: int, oversample: int, power_iterations: int, seed: int
) -> RouterSVD:
    matrix = weight.double()
    maximum = min(matrix.shape)
    if not 1 <= rank <= maximum:
        raise ValueError("randomized SVD rank is outside matrix dimensions")
    width = min(maximum, rank + oversample)
    generator = torch.Generator(device=matrix.device).manual_seed(seed)
    omega = torch.randn(
        matrix.shape[1], width, dtype=matrix.dtype, device=matrix.device, generator=generator
    )
    sample = matrix @ omega
    for _ in range(power_iterations):
        sample = matrix @ (matrix.T @ sample)
    q, _ = torch.linalg.qr(sample, mode="reduced")
    small = q.T @ matrix
    small_u, singular_values, vh = torch.linalg.svd(small, full_matrices=False)
    return RouterSVD("randomized", q @ small_u[:, :rank], singular_values[:rank], vh[:rank])


def row_space_coordinates(states: Tensor, vh: Tensor, *, rank: int | None = None) -> Tensor:
    basis = vh if rank is None else vh[:rank]
    return states.double() @ basis.T


def project_to_row_space(states: Tensor, vh: Tensor, *, rank: int | None = None) -> Tensor:
    basis = vh if rank is None else vh[:rank]
    return row_space_coordinates(states, basis) @ basis


def projected_router_logits(
    states: Tensor, weight: Tensor, bias: Tensor | None, vh: Tensor, *, rank: int | None = None
) -> Tensor:
    projected = project_to_row_space(states, vh, rank=rank)
    logits = projected @ weight.double().T
    return logits if bias is None else logits + bias.double()


def relative_reconstruction_error(weight: Tensor, svd: RouterSVD, rank: int) -> float:
    approximation = (svd.u[:, :rank] * svd.singular_values[:rank]) @ svd.vh[:rank]
    return float(
        torch.linalg.vector_norm(weight.double() - approximation)
        / torch.linalg.vector_norm(weight.double())
    )


def empirical_covariance_sketch(states: Tensor, *, rank: int) -> CovarianceSketch:
    values = states.reshape(-1, states.shape[-1]).double()
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = (centered.T @ centered) / max(1, values.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)[: min(rank, values.shape[1])]
    return CovarianceSketch(
        int(values.shape[0]), mean, covariance, eigenvectors[:, order].T, eigenvalues[order]
    )


def covariance_aware_logit_variance(weight: Tensor, covariance: Tensor) -> Tensor:
    transformed = weight.double() @ covariance.double()
    return (transformed * weight.double()).sum(dim=-1)


def topk_margin(logits: Tensor, top_k: int) -> Tensor:
    if not 1 <= top_k < logits.shape[-1]:
        raise ValueError("margin requires 1 <= top_k < num_experts")
    ordered = logits.sort(dim=-1, descending=True).values
    return ordered[..., top_k - 1] - ordered[..., top_k]


def margin_stability_condition(base_logits: Tensor, candidate_logits: Tensor, top_k: int) -> Tensor:
    error = (candidate_logits - base_logits).abs().amax(dim=-1)
    return error < topk_margin(base_logits, top_k) / 2


def topk_is_stable(base_logits: Tensor, candidate_logits: Tensor, top_k: int) -> Tensor:
    base = base_logits.topk(top_k, dim=-1).indices.sort(dim=-1).values
    candidate = candidate_logits.topk(top_k, dim=-1).indices.sort(dim=-1).values
    return (base == candidate).all(dim=-1)


def token_geometry(
    probabilities: Tensor, logits: Tensor, *, top_k: int, top_b: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
    ordered = logits.sort(dim=-1, descending=True).values
    top1_gap = ordered[..., 0] - ordered[..., 1]
    margin = ordered[..., top_k - 1] - ordered[..., top_k]
    cumulative_mass = probabilities.topk(top_b, dim=-1).values.sum(dim=-1)
    return entropy, top1_gap, margin, cumulative_mass


def pairwise_boundaries(
    states: Tensor,
    topk_ids: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    epsilon: float,
) -> list[tuple[int, int, float, float, float]]:
    values = states.reshape(-1, states.shape[-1]).double()
    routes = topk_ids.reshape(values.shape[0], -1)
    rows = []
    for expert_a in range(weight.shape[0]):
        for expert_b in range(expert_a + 1, weight.shape[0]):
            delta = weight[expert_a].double() - weight[expert_b].double()
            offset = 0.0 if bias is None else float(bias[expert_a] - bias[expert_b])
            signed = values @ delta + offset
            distance = signed.abs() / torch.linalg.vector_norm(delta).clamp_min(1e-30)
            active_a = (routes == expert_a).any(dim=-1)
            active_b = (routes == expert_b).any(dim=-1)
            rows.append(
                (
                    expert_a,
                    expert_b,
                    float(distance.mean()),
                    float((distance <= epsilon).double().mean()),
                    float((active_a & active_b).double().mean()),
                )
            )
    return rows


def svd_metric_rows(
    layer_idx: int, weight: Tensor, decompositions: tuple[RouterSVD, ...]
) -> list[SVDMetricRow]:
    rows = []
    total_energy = weight.double().square().sum()
    for decomposition in decompositions:
        cumulative = decomposition.singular_values.square().cumsum(dim=0) / total_energy
        for index, singular_value in enumerate(decomposition.singular_values):
            rank = index + 1
            rows.append(
                SVDMetricRow(
                    layer_idx,
                    decomposition.method,
                    rank,
                    float(singular_value),
                    float(cumulative[index]),
                    relative_reconstruction_error(weight, decomposition, rank),
                    decomposition.effective_rank,
                )
            )
    return rows
