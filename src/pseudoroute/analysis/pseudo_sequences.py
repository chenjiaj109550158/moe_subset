"""Offline-only DapQ-style pseudo-sequence construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor


class FactorialCondition(StrEnum):
    SC_SP = "SC_SP"
    DC_SP = "DC_SP"
    SC_DP = "SC_DP"
    DC_DP = "DC_DP"

    @property
    def same_content(self) -> bool:
        return self in {self.SC_SP, self.SC_DP}

    @property
    def same_position(self) -> bool:
        return self in {self.SC_SP, self.DC_SP}


@dataclass(frozen=True)
class OfflineDocument:
    sample_id: str
    token_ids: Tensor

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 1 or not self.token_ids.numel():
            raise ValueError("offline document tokens must be a non-empty vector")


@dataclass(frozen=True)
class OfflinePseudoSequence:
    """Future-bearing factorial input that is intentionally not an online state."""

    sample_id: str
    condition: FactorialCondition
    boundary: int
    horizon: int
    input_ids: Tensor
    position_ids: Tensor
    true_future_ids: Tensor
    context_swapped: bool
    content_seed: int
    position_offset: int

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.input_ids.shape[0] != 1:
            raise ValueError("factorial input_ids must have shape [1, sequence]")
        if self.position_ids.shape != self.input_ids.shape:
            raise ValueError("position IDs must match factorial input IDs")
        if self.true_future_ids.numel() != self.horizon:
            raise ValueError("future span must equal the declared horizon")
        if self.input_ids.shape[1] != self.boundary + self.horizon:
            raise ValueError("pseudo sequence cannot cross its document boundary")

    @property
    def pseudo_slice(self) -> slice:
        return slice(self.boundary, self.boundary + self.horizon)


def _different_content(true_future: Tensor, *, vocab_size: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randint(0, vocab_size, true_future.shape, generator=generator)
    collision = values == true_future.cpu()
    values[collision] = (values[collision] + 1) % vocab_size
    return values.to(true_future.device)


def build_factorial_sequences(
    documents: tuple[OfflineDocument, ...],
    *,
    boundary: int,
    horizon: int,
    vocab_size: int,
    content_seed: int,
    position_offset: int,
    include_context_swap: bool,
) -> tuple[OfflinePseudoSequence, ...]:
    if boundary < 1 or horizon < 1:
        raise ValueError("boundary and horizon must be positive")
    if position_offset == 0:
        raise ValueError("different-position offset must be nonzero")
    if any(document.token_ids.numel() < boundary + horizon for document in documents):
        raise ValueError("factorial span would cross a document boundary")
    examples = []
    for document_index, document in enumerate(documents):
        true_prefix = document.token_ids[:boundary]
        true_future = document.token_ids[boundary : boundary + horizon]
        different = _different_content(
            true_future, vocab_size=vocab_size, seed=content_seed + document_index
        )
        context_options = [(true_prefix, False)]
        if include_context_swap:
            if len(documents) < 2:
                raise ValueError("context swap requires at least two documents")
            other = documents[(document_index + 1) % len(documents)]
            swapped = other.token_ids[:boundary]
            context_options.append((swapped, True))
        for prefix, context_swapped in context_options:
            for condition in FactorialCondition:
                future = true_future if condition.same_content else different
                tokens = torch.cat((prefix, future)).reshape(1, -1)
                positions = torch.arange(boundary + horizon, device=tokens.device).reshape(1, -1)
                if not condition.same_position:
                    positions[:, boundary:] += position_offset
                examples.append(
                    OfflinePseudoSequence(
                        sample_id=document.sample_id,
                        condition=condition,
                        boundary=boundary,
                        horizon=horizon,
                        input_ids=tokens,
                        position_ids=positions,
                        true_future_ids=true_future.clone(),
                        context_swapped=context_swapped,
                        content_seed=content_seed + document_index,
                        position_offset=position_offset,
                    )
                )
    return tuple(examples)
