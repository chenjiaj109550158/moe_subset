from pathlib import Path

import torch
from safetensors.torch import save_file

from pseudoroute.benchmark import pseudo_mass_preserving_residual_held_out as held_out


def test_oracle_gap_recovery_is_relative_to_corrected_previous() -> None:
    assert held_out._gap_recovery(0.75, 0.60, 0.90) == 0.5
    assert held_out._gap_recovery(0.60, 0.60, 0.90) == 0.0
    assert held_out._gap_recovery(0.70, 0.80, 0.80) == float("-inf")


def test_previous_audit_requires_static_first_then_own_previous_window(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(held_out, "OUTPUT", tmp_path)  # type: ignore[attr-defined]
    static = {layer: tuple(range(held_out.BUDGET)) for layer in range(held_out.LAYERS)}
    ids = torch.tensor(range(32, 40), dtype=torch.int64).repeat(16, held_out.LAYERS, 1)
    weights = torch.ones(16, held_out.LAYERS, held_out.TOP_K)
    expected_later = torch.tensor(tuple(range(24)) + tuple(range(32, 40)), dtype=torch.int64)
    subsets = torch.empty(2, held_out.LAYERS, held_out.BUDGET, dtype=torch.int64)
    subsets[0] = torch.tensor([static[layer] for layer in range(held_out.LAYERS)])
    subsets[1] = expected_later.repeat(held_out.LAYERS, 1)
    tensor_path = held_out._paths("development_reference", 0, held_out.PREVIOUS.key)[1]
    tensor_path.parent.mkdir(parents=True)
    save_file(
        {
            "boundaries": torch.tensor([0, 8]),
            "subsets": subsets,
            "natural_router_topk_ids": ids,
            "natural_router_topk_weights": weights,
        },
        str(tensor_path),
    )
    row = {"prompt_capture_audit": {"production_rng_unchanged": True}}
    assert held_out._previous_audit(row, tensor_path, static)
    subsets[0, 0, 0] = 127
    save_file(
        {
            "boundaries": torch.tensor([0, 8]),
            "subsets": subsets,
            "natural_router_topk_ids": ids,
            "natural_router_topk_weights": weights,
        },
        str(tensor_path),
    )
    assert not held_out._previous_audit(row, tensor_path, static)
