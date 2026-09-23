"""Predeclared conservative decision rules, independent of report prose."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binomtest  # type: ignore[import-untyped]


def quality_interval(gains: int, losses: int, n: int, claims: int = 1) -> list[float] | None:
    if n == 0:
        return None
    if min(gains, losses) < 0 or gains + losses > n or claims < 1:
        raise ValueError("invalid paired counts/claims")
    level = 1 - 0.05 / (2 * claims)
    gain = binomtest(gains, n).proportion_ci(confidence_level=level, method="exact")
    loss = binomtest(losses, n).proportion_ci(confidence_level=level, method="exact")
    return [max(-1.0, gain.low - loss.high), min(1.0, gain.high - loss.low)]


def speed_interval(
    paired_ratios: dict[str, list[float]], *, seed: int = 20260922, resamples: int = 10000
) -> tuple[float, list[float]]:
    if not paired_ratios or any(not v or min(v) <= 0 for v in paired_ratios.values()):
        raise ValueError("complete positive paired request TPS ratios required")
    logs = [np.log(v) for _, v in sorted(paired_ratios.items())]
    point = float(np.exp(np.concatenate(logs).mean()))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(resamples):
        chosen = [logs[i] for i in rng.integers(0, len(logs), len(logs))]
        estimates.append(np.exp(np.concatenate([rng.choice(v, len(v)) for v in chosen]).mean()))
    return point, np.quantile(estimates, [0.025, 0.975]).tolist()


def evaluate_decision(metrics: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    d = protocol["decision"]
    gates = {key: "NOT_TESTED" for key in ("V", "Q", "S", "D", "P")}
    result: dict[str, Any] = dict(
        overall="INCONCLUSIVE",
        runtime="BLOCKED",
        window_subset="INCONCLUSIVE",
        pseudo_predictor="NOT_TESTED",
        gates=gates,
    )
    if metrics.get("external_blockers") and not metrics.get("actual_model_runs_completed"):
        result.update(overall="BLOCKED", window_subset="BLOCKED", pseudo_predictor="BLOCKED")
        gates["V"] = "BLOCKED"
        return result
    if metrics.get("correctness_pass") is False:
        gates["V"] = "FAIL"
        result.update(overall="PIVOT_RUNTIME", runtime="NEEDS_WORK")
        return result
    if metrics.get("correctness_pass") and metrics.get("runtime_measurement_valid"):
        result["runtime"] = "VALIDATED"
    if not metrics.get("validity_pass"):
        gates["V"] = "INCONCLUSIVE"
        return result
    result["runtime"] = "VALIDATED"
    gates["V"] = "PASS"
    n, planned = metrics.get("completed_paired_quality_n", 0), metrics.get("planned_quality_n")
    if planned is None or n != planned or n < 64 or not metrics.get("required_pairs_complete"):
        return result
    qci = metrics.get("accuracy_delta_ci")
    sci = metrics.get("decode_speedup_ci")
    speed = metrics.get("decode_speedup")
    if qci:
        gates["Q"] = (
            "PASS"
            if qci[0] >= -d["quality_max_accuracy_drop_fraction"]
            and not metrics.get("new_collapse")
            else "INCONCLUSIVE"
        )
        if qci[1] < -d["quality_max_accuracy_drop_fraction"] or metrics.get("new_collapse"):
            gates["Q"] = "FAIL"
    if sci and speed is not None and metrics.get("timing_valid"):
        gates["S"] = (
            "PASS"
            if speed >= d["practical_decode_speedup_point"]
            and sci[0] > d["required_speedup_ci_lower_exclusive"]
            else "INCONCLUSIVE"
        )
        if sci[1] < 1.05:
            gates["S"] = "FAIL"
    gates["D"] = "PASS" if metrics.get("dynamic_value_confirmed") else "INCONCLUSIVE"
    gates["P"] = "PASS" if metrics.get("pseudo_increment_confirmed") else "INCONCLUSIVE"
    if metrics.get("static_dominance_confirmed"):
        result.update(overall="STATIC_SUFFICIENT_TESTED_SCOPE", window_subset="STATIC_SUFFICIENT")
    elif all(gates[g] == "PASS" for g in ("Q", "S")):
        p95 = metrics.get("p95_latency_ratio")
        if gates["D"] == "PASS" and p95 is not None and p95 <= d["p95_latency_ratio_goal"]:
            result.update(
                overall="GO_WINDOW_AND_PSEUDO"
                if gates["P"] == "PASS"
                else "GO_WINDOW_HISTORY_ONLY",
                window_subset="CONTINUE",
            )
        else:
            result.update(overall="PILOT_PROMISING", window_subset="PROMISING")
    elif metrics.get("sufficient_negative_evidence") and (
        gates["Q"] == "FAIL" or gates["S"] == "FAIL"
    ):
        result.update(overall="NO_GO_TESTED_REGIME", window_subset="STOP_TESTED_REGIME")
    if gates["P"] == "PASS":
        result["pseudo_predictor"] = "CONTINUE"
    elif metrics.get("pseudo_dominated"):
        result["pseudo_predictor"] = "STOP_CURRENT_VARIANT"
    elif metrics.get("pseudo_tested"):
        result["pseudo_predictor"] = "INCONCLUSIVE"
    return result
