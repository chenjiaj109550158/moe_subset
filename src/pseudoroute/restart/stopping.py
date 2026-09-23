"""Request-scoped native stopping with an append-only prompt+continuation buffer."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class RequestStopState:
    def __init__(
        self,
        tokenizer: Any,
        prompt_ids: Tensor,
        eos: int | list[int] | None,
        stop_strings: tuple[str, ...],
        max_new_tokens: int,
    ) -> None:
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("restart stopping supports batch=1")
        if max_new_tokens < 1:
            raise ValueError("positive token capacity required")
        from transformers import StopStringCriteria

        self.criteria = StopStringCriteria(tokenizer, list(stop_strings)) if stop_strings else None
        self.eos = frozenset([eos] if isinstance(eos, int) else (eos or []))
        self.length = prompt_ids.shape[1]
        self.buffer = torch.empty(
            (1, self.length + max_new_tokens), device=prompt_ids.device, dtype=torch.long
        )
        self.buffer[:, : self.length].copy_(prompt_ids)
        self.appended = 0

    def append(self, token: int, scores: Tensor) -> bool:
        if self.length >= self.buffer.shape[1]:
            raise ValueError("request token buffer capacity exceeded")
        self.buffer[0, self.length] = token
        self.length += 1
        self.appended += 1
        if token in self.eos:
            return True
        if self.criteria is None:
            return False
        result = self.criteria(self.buffer[:, : self.length], scores)
        if result.numel() != 1:
            raise RuntimeError("expected one native stop result")
        return bool(result.item())
