"""Grouped, leakage-safe M6 predictor dataset construction and storage."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.types import DeployableDecodeState, InformationRegime


@dataclass(frozen=True)
class PredictorRow:
    sample_id: str
    boundary_position: int
    decision_mode: str
    history_length: int
    horizon: int
    split: str


@dataclass(frozen=True)
class PredictorDataset:
    features: Tensor
    targets: Tensor
    rows: tuple[PredictorRow, ...]
    num_layers: int
    num_experts: int
    history_length: int
    position_scale: float
    horizon_scale: float

    def indices(self, split: str) -> Tensor:
        return torch.tensor(
            [index for index, row in enumerate(self.rows) if row.split == split],
            dtype=torch.long,
        )


def grouped_split(
    sample_ids: tuple[str, ...], *, train_fraction: float, validation_fraction: float, seed: int
) -> dict[str, str]:
    if len(set(sample_ids)) < 3:
        raise ValueError("grouped train/validation/test split requires at least three documents")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")
    unique = sorted(set(sample_ids))
    random.Random(seed).shuffle(unique)
    train_count = max(1, int(len(unique) * train_fraction))
    validation_count = max(1, int(len(unique) * validation_fraction))
    if train_count + validation_count >= len(unique):
        train_count = len(unique) - 2
        validation_count = 1
    mapping = {}
    for index, sample_id in enumerate(unique):
        mapping[sample_id] = (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
    return mapping


def state_features(
    state: DeployableDecodeState,
    horizon: int,
    *,
    num_experts: int,
    history_length: int,
    position_scale: float,
    horizon_scale: float,
) -> Tensor:
    current = torch.cat([logits.reshape(-1).double() for logits in state.current_router_logits])
    frequency = torch.zeros(len(state.current_router_logits), num_experts, dtype=torch.float64)
    mass = torch.zeros_like(frequency)
    pairs = list(zip(state.router_history_topk, state.router_history_weights, strict=True))
    for ids_by_layer, weights_by_layer in pairs[-history_length:]:
        for layer in range(len(state.current_router_logits)):
            ids = ids_by_layer[layer].reshape(-1).long().cpu()
            frequency[layer].scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float64))
            mass[layer].scatter_add_(0, ids, weights_by_layer[layer].reshape(-1).double().cpu())
    scalars = torch.tensor(
        [state.absolute_position / position_scale, horizon / horizon_scale],
        dtype=torch.float64,
    )
    return torch.cat((current.cpu(), frequency.reshape(-1), mass.reshape(-1), scalars))


def oracle_mass_target(
    future_ids: Tensor, future_weights: Tensor, *, num_layers: int, num_experts: int
) -> Tensor:
    target = torch.zeros(num_layers, num_experts, dtype=torch.float64)
    for layer in range(num_layers):
        target[layer].scatter_add_(
            0,
            future_ids[:, layer].reshape(-1).long().cpu(),
            future_weights[:, layer].reshape(-1).double().cpu(),
        )
    return target.reshape(-1)


def build_predictor_examples(
    adapter: TinyMoEAdapter,
    documents: tuple[tuple[str, Tensor], ...],
    *,
    horizons: tuple[int, ...],
    history_length: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[PredictorDataset, tuple[DeployableDecodeState, ...]]:
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("horizons must be positive")
    split_by_sample = grouped_split(
        tuple(sample_id for sample_id, _ in documents),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    num_layers = len(adapter.spec.moe_layer_indices)
    num_experts = next(iter(adapter.spec.num_experts_by_layer.values()))
    position_scale = float(max(tokens.numel() for _, tokens in documents))
    horizon_scale = float(max(horizons))
    features = []
    targets = []
    rows = []
    states: list[DeployableDecodeState] = []
    with torch.inference_mode():
        for sample_id, token_ids in documents:
            output = adapter.model(token_ids, capture_trace=True)
            by_key = {(trace.token_position, trace.layer_idx): trace for trace in output.traces}
            token_count = int(token_ids.numel())
            ids_history = []
            weights_history = []
            logits_history = []
            for position in range(token_count):
                traces = [by_key[(position, layer)] for layer in adapter.spec.moe_layer_indices]
                ids_history.append(
                    torch.stack([trace.topk_ids.reshape(-1) for trace in traces]).cpu()
                )
                weights_history.append(
                    torch.stack([trace.topk_weights.reshape(-1) for trace in traces]).cpu()
                )
                logits_history.append(tuple(trace.raw_logits.reshape(-1).cpu() for trace in traces))
            for boundary in range(history_length - 1, token_count - 1):
                state = DeployableDecodeState(
                    information_regime=InformationRegime.ONLINE_PRE_SAMPLE,
                    prefix_token_ids=token_ids[:, : boundary + 1].cpu(),
                    absolute_position=boundary,
                    current_router_logits=logits_history[boundary],
                    resident_experts=frozenset(),
                    router_history_topk=tuple(ids_history[: boundary + 1]),
                    router_history_weights=tuple(weights_history[: boundary + 1]),
                )
                for horizon in horizons:
                    end = boundary + 1 + horizon
                    if end > token_count:
                        continue
                    states.append(state)
                    features.append(
                        state_features(
                            state,
                            horizon,
                            num_experts=num_experts,
                            history_length=history_length,
                            position_scale=position_scale,
                            horizon_scale=horizon_scale,
                        )
                    )
                    future_ids = torch.stack(ids_history[boundary + 1 : end])
                    future_weights = torch.stack(weights_history[boundary + 1 : end])
                    targets.append(
                        oracle_mass_target(
                            future_ids,
                            future_weights,
                            num_layers=num_layers,
                            num_experts=num_experts,
                        )
                    )
                    rows.append(
                        PredictorRow(
                            sample_id,
                            boundary,
                            "online_pre_sample",
                            history_length,
                            horizon,
                            split_by_sample[sample_id],
                        )
                    )
    dataset = PredictorDataset(
        torch.stack(features),
        torch.stack(targets),
        tuple(rows),
        num_layers,
        num_experts,
        history_length,
        position_scale,
        horizon_scale,
    )
    return dataset, tuple(states)


def build_predictor_dataset(
    adapter: TinyMoEAdapter,
    documents: tuple[tuple[str, Tensor], ...],
    *,
    horizons: tuple[int, ...],
    history_length: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> PredictorDataset:
    return build_predictor_examples(
        adapter,
        documents,
        horizons=horizons,
        history_length=history_length,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )[0]


def save_predictor_dataset(root: Path, dataset: PredictorDataset) -> None:
    root.mkdir(parents=True, exist_ok=True)
    save_file(
        {"features": dataset.features.contiguous(), "targets": dataset.targets.contiguous()},
        str(root / "dataset.safetensors"),
    )
    manifest = {
        "schema_version": 1,
        "information_regime": "offline_teacher_forced",
        "num_rows": len(dataset.rows),
        "num_layers": dataset.num_layers,
        "num_experts": dataset.num_experts,
        "history_length": dataset.history_length,
        "position_scale": dataset.position_scale,
        "horizon_scale": dataset.horizon_scale,
        "rows": [asdict(row) for row in dataset.rows],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_predictor_dataset(root: Path) -> PredictorDataset:
    tensors = load_file(str(root / "dataset.safetensors"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return PredictorDataset(
        tensors["features"].double(),
        tensors["targets"].double(),
        tuple(PredictorRow(**row) for row in manifest["rows"]),
        int(manifest["num_layers"]),
        int(manifest["num_experts"]),
        int(manifest["history_length"]),
        float(manifest["position_scale"]),
        float(manifest["horizon_scale"]),
    )
