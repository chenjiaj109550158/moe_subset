import torch

from pseudoroute.analysis.router_geometry import (
    exact_router_svd,
    margin_stability_condition,
    projected_router_logits,
    randomized_router_svd,
    relative_reconstruction_error,
    topk_is_stable,
)


def test_exact_row_space_projection_reproduces_router_logits() -> None:
    generator = torch.Generator().manual_seed(8)
    weight = torch.randn(4, 9, generator=generator)
    bias = torch.randn(4, generator=generator)
    states = torch.randn(7, 9, generator=generator)
    decomposition = exact_router_svd(weight)
    projected = projected_router_logits(states, weight, bias, decomposition.vh)
    expected = states.double() @ weight.double().T + bias.double()
    assert torch.allclose(projected, expected, atol=1e-10, rtol=1e-10)


def test_low_rank_reconstruction_error_matches_direct_error() -> None:
    weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    decomposition = exact_router_svd(weight)
    expected = torch.sqrt(torch.tensor(2.0**2 + 1.0**2)) / torch.linalg.vector_norm(weight)
    assert abs(relative_reconstruction_error(weight, decomposition, 2) - float(expected)) < 1e-7
    randomized = randomized_router_svd(weight, rank=2, oversample=2, power_iterations=2, seed=4)
    assert relative_reconstruction_error(weight, randomized, 2) < 0.5


def test_margin_stability_sufficient_condition_and_strict_boundary() -> None:
    base = torch.tensor([[5.0, 4.0, 1.0, 0.0]])
    stable = base + torch.tensor([[-0.4, 0.4, 0.4, -0.4]])
    # k=2 margin is 3; infinity error 0.4 < 1.5, hence guaranteed stable.
    condition = margin_stability_condition(base, stable, 2)
    assert condition.item()
    assert topk_is_stable(base, stable, 2).item()

    boundary = base + torch.tensor([[0.0, -1.5, 1.5, 0.0]])
    assert not margin_stability_condition(base, boundary, 2).item()
    # At equality the sufficient condition is deliberately strict and makes no guarantee.

    unstable = base + torch.tensor([[0.0, -2.0, 2.0, 0.0]])
    assert not margin_stability_condition(base, unstable, 2).item()
    assert not topk_is_stable(base, unstable, 2).item()
