"""Adapter-neutral teacher-forced route collection."""

from __future__ import annotations

from torch import Tensor

from pseudoroute.models.base import MoEModelAdapter, TraceLevel, TraceRequest
from pseudoroute.tracing.store import TraceStore


def collect_sample(
    adapter: MoEModelAdapter,
    store: TraceStore,
    *,
    sample_id: str,
    token_ids: Tensor,
    trace_level: TraceLevel,
    is_prompt: bool = True,
) -> bool:
    if sample_id in store.completed_sample_ids:
        return False
    result = adapter.run_base_forward(token_ids, trace_request=TraceRequest(trace_level))
    return store.append_sample(sample_id, token_ids, result.traces, is_prompt=is_prompt)
