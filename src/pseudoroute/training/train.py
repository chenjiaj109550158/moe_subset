"""Deterministic M6 baseline predictor training."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import torch
from torch import nn

from pseudoroute.probes.learned import (
    ADEPTStyleRFProbe,
    DirectLinearProbe,
    DirectMLPProbe,
    FeatureSpec,
    LearnedProbe,
)
from pseudoroute.training.dataset import PredictorDataset


@dataclass(frozen=True)
class TrainingCost:
    probe_name: str
    calibration_rows: int
    training_seconds: float
    epochs: int
    approximate_flops: int
    parameter_count: int


def feature_spec(dataset: PredictorDataset) -> FeatureSpec:
    return FeatureSpec(
        dataset.num_layers,
        dataset.num_experts,
        dataset.history_length,
        dataset.position_scale,
        dataset.horizon_scale,
    )


def train_linear(
    dataset: PredictorDataset, *, ridge_lambda: float
) -> tuple[DirectLinearProbe, TrainingCost]:
    indices = dataset.indices("train")
    x = dataset.features[indices].double()
    y = dataset.targets[indices].double()
    started = time.perf_counter()
    mean_x = x.mean(dim=0)
    mean_y = y.mean(dim=0)
    centered_x, centered_y = x - mean_x, y - mean_y
    identity = torch.eye(x.shape[1], dtype=torch.float64)
    weight = (
        torch.linalg.pinv(centered_x.T @ centered_x + ridge_lambda * identity)
        @ centered_x.T
        @ centered_y
    )
    bias = mean_y - mean_x @ weight
    elapsed = time.perf_counter() - started
    probe = DirectLinearProbe(weight, bias, feature_spec(dataset), ridge_lambda)
    parameters = weight.numel() + bias.numel()
    flops = int(2 * x.shape[0] * x.shape[1] * y.shape[1])
    return probe, TrainingCost(probe.name, len(indices), elapsed, 1, flops, parameters)


def train_mlp(
    dataset: PredictorDataset, *, hidden_size: int, epochs: int, learning_rate: float, seed: int
) -> tuple[DirectMLPProbe, TrainingCost]:
    torch.manual_seed(seed)
    indices = dataset.indices("train")
    x = dataset.features[indices].float()
    y = dataset.targets[indices].float()
    model = nn.Sequential(
        nn.Linear(x.shape[1], hidden_size), nn.ReLU(), nn.Linear(hidden_size, y.shape[1])
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    started = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    elapsed = time.perf_counter() - started
    first = model[0]
    second = model[2]
    assert isinstance(first, nn.Linear) and isinstance(second, nn.Linear)
    probe = DirectMLPProbe(
        first.weight.detach().T.contiguous(),
        first.bias.detach().clone(),
        second.weight.detach().T.contiguous(),
        second.bias.detach().clone(),
        feature_spec(dataset),
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    flops = int(epochs * x.shape[0] * 2 * (x.shape[1] * hidden_size + hidden_size * y.shape[1]))
    return probe, TrainingCost(probe.name, len(indices), elapsed, epochs, flops, parameters)


def train_adept_style_rf(
    dataset: PredictorDataset, *, trees: int, feature_candidates: int, seed: int
) -> tuple[ADEPTStyleRFProbe, TrainingCost]:
    indices = dataset.indices("train")
    x = dataset.features[indices].double()
    y = dataset.targets[indices].double()
    randomizer = random.Random(seed)
    chosen_features = []
    thresholds = []
    left_values = []
    right_values = []
    started = time.perf_counter()
    for _ in range(trees):
        candidates = randomizer.sample(range(x.shape[1]), k=min(feature_candidates, x.shape[1]))
        best = None
        for feature in candidates:
            threshold = float(x[:, feature].median())
            left = x[:, feature] <= threshold
            if not bool(left.any()) or bool(left.all()):
                continue
            left_mean, right_mean = y[left].mean(0), y[~left].mean(0)
            error = float(
                (y[left] - left_mean).square().sum() + (y[~left] - right_mean).square().sum()
            )
            if best is None or error < best[0]:
                best = (error, feature, threshold, left_mean, right_mean)
        if best is None:
            feature = candidates[0]
            mean = y.mean(0)
            best = (0.0, feature, float(x[:, feature].mean()), mean, mean)
        _, feature, threshold, left_mean, right_mean = best
        chosen_features.append(feature)
        thresholds.append(threshold)
        left_values.append(left_mean)
        right_values.append(right_mean)
    elapsed = time.perf_counter() - started
    probe = ADEPTStyleRFProbe(
        torch.tensor(chosen_features),
        torch.tensor(thresholds),
        torch.stack(left_values),
        torch.stack(right_values),
        feature_spec(dataset),
    )
    parameters = (
        probe.feature_indices.numel()
        + probe.thresholds.numel()
        + probe.left_values.numel()
        + probe.right_values.numel()
    )
    flops = trees * len(indices) * feature_candidates * y.shape[1]
    return probe, TrainingCost(probe.name, len(indices), elapsed, 1, flops, parameters)


def train_all(
    dataset: PredictorDataset,
    *,
    ridge_lambda: float,
    mlp_hidden: int,
    mlp_epochs: int,
    learning_rate: float,
    rf_trees: int,
    rf_feature_candidates: int,
    seed: int,
) -> tuple[list[LearnedProbe], list[TrainingCost]]:
    trained = [
        train_linear(dataset, ridge_lambda=0.0),
        train_linear(dataset, ridge_lambda=ridge_lambda),
        train_mlp(
            dataset,
            hidden_size=mlp_hidden,
            epochs=mlp_epochs,
            learning_rate=learning_rate,
            seed=seed,
        ),
        train_adept_style_rf(
            dataset, trees=rf_trees, feature_candidates=rf_feature_candidates, seed=seed
        ),
    ]
    return [item[0] for item in trained], [item[1] for item in trained]
