"""Calibration-free token continuation from the current policy's known context."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

ContextContinuationMode = Literal[
    "longest_suffix_full_continuation",
    "sampled_unigram_full_continuation",
    "longest_suffix_partial_recent_fill",
]


@dataclass(frozen=True)
class ContextContinuationPlan:
    mode: ContextContinuationMode
    anchor_token_ids: tuple[int, ...]
    baseline_anchor_token_ids: tuple[int, ...]
    known_context_length: int
    current_query_start: int
    current_query_end: int
    matched_suffix_length: int
    match_start: int | None
    match_end: int | None
    eligible_match_count_at_selected_length: int
    copied_context_indices: tuple[int, ...]
    copied_anchor_count: int
    copied_anchor_fraction: float
    changed_anchor_count: int
    changed_anchor_fraction: float
    fallback_used: bool
    partial_recent_fill_count: int
    anchor_one_is_sampled_next: bool
    earlier_match_nonoverlapping: bool
    copied_indices_in_known_context: bool
    future_token_accessed: bool
    tie_break: str
    planning_latency_seconds_measured: float


def _recent_anchors(
    boundary: int,
    prompt_token_ids: tuple[int, ...],
    realized_token_ids: tuple[int, ...],
    *,
    horizon: int,
) -> tuple[int, ...]:
    history = (*prompt_token_ids, *realized_token_ids[:boundary])
    if boundary < 0 or boundary >= len(realized_token_ids):
        raise ValueError("context-continuation boundary is outside the saved trajectory")
    if len(history) < horizon - 1:
        raise ValueError("context-continuation history is shorter than the horizon")
    return (realized_token_ids[boundary], *history[-(horizon - 1) :])


def _eligible_starts(
    context: tuple[int, ...],
    *,
    match_length: int,
    required_successors: int,
) -> tuple[int, ...]:
    query_start = len(context) - match_length
    query = context[query_start:]
    starts = []
    for start in range(query_start - match_length + 1):
        match_end = start + match_length
        if (
            context[start:match_end] == query
            and match_end <= query_start
            and len(context) - match_end >= required_successors
        ):
            starts.append(start)
    return tuple(starts)


def build_context_continuation(
    boundary: int,
    prompt_token_ids: tuple[int, ...],
    realized_token_ids: tuple[int, ...],
    mode: ContextContinuationMode,
    *,
    horizon: int = 8,
) -> ContextContinuationPlan:
    """Build H anchors without reading a token after the sampled-next boundary."""
    started = time.perf_counter()
    if horizon < 2:
        raise ValueError("context continuation requires at least two anchors")
    baseline = _recent_anchors(
        boundary,
        prompt_token_ids,
        realized_token_ids,
        horizon=horizon,
    )
    known = (*prompt_token_ids, *realized_token_ids[: boundary + 1])
    copied_target = horizon - 1
    partial = mode == "longest_suffix_partial_recent_fill"
    required = 1 if partial else copied_target
    lengths = (1,) if mode == "sampled_unigram_full_continuation" else range(len(known) // 2, 0, -1)
    selected_length = 0
    selected_starts: tuple[int, ...] = ()
    for match_length in lengths:
        starts = _eligible_starts(
            known,
            match_length=match_length,
            required_successors=required,
        )
        if starts:
            selected_length = match_length
            selected_starts = starts
            break

    if selected_starts:
        match_start = max(selected_starts)
        match_end = match_start + selected_length
        copied_count = min(copied_target, len(known) - match_end)
        copied_indices = tuple(range(match_end, match_end + copied_count))
        anchors = list(baseline)
        anchors[1 : copied_count + 1] = [known[index] for index in copied_indices]
    else:
        match_start = None
        match_end = None
        copied_count = 0
        copied_indices = ()
        anchors = list(baseline)

    query_start = len(known) - selected_length if selected_length else len(known)
    changed_count = sum(
        anchor != reference for anchor, reference in zip(anchors[1:], baseline[1:], strict=True)
    )
    return ContextContinuationPlan(
        mode=mode,
        anchor_token_ids=tuple(anchors),
        baseline_anchor_token_ids=baseline,
        known_context_length=len(known),
        current_query_start=query_start,
        current_query_end=len(known),
        matched_suffix_length=selected_length,
        match_start=match_start,
        match_end=match_end,
        eligible_match_count_at_selected_length=len(selected_starts),
        copied_context_indices=copied_indices,
        copied_anchor_count=copied_count,
        copied_anchor_fraction=copied_count / copied_target,
        changed_anchor_count=changed_count,
        changed_anchor_fraction=changed_count / copied_target,
        fallback_used=copied_count == 0,
        partial_recent_fill_count=copied_target - copied_count if copied_count else 0,
        anchor_one_is_sampled_next=anchors[0] == realized_token_ids[boundary],
        earlier_match_nonoverlapping=(match_end is None or match_end <= query_start),
        copied_indices_in_known_context=all(0 <= index < len(known) for index in copied_indices),
        future_token_accessed=False,
        tie_break="most_recent_match_start",
        planning_latency_seconds_measured=time.perf_counter() - started,
    )
