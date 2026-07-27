import torch

from pseudoroute.analysis.dapq_factorial import (
    add_position_dominance,
    capture_factorial,
    compare_capture,
    paired_bootstrap,
)
from pseudoroute.analysis.pseudo_sequences import (
    FactorialCondition,
    OfflineDocument,
    build_factorial_sequences,
)
from pseudoroute.config import load_config
from pseudoroute.models.adapters.tiny import TinyMoEAdapter
from pseudoroute.models.tiny_moe import TinyMoE


def adapter() -> TinyMoEAdapter:
    config = load_config("configs/model/tiny_moe.yaml")
    return TinyMoEAdapter(TinyMoE(config.model, seed=config.experiment.seed, device="cpu"))


def test_scsp_capture_exactly_matches_teacher_forced_ground_truth() -> None:
    document = OfflineDocument("sample", torch.tensor([1, 2, 3, 4, 5, 6]))
    sequence = build_factorial_sequences(
        (document,),
        boundary=3,
        horizon=3,
        vocab_size=32,
        content_seed=4,
        position_offset=5,
        include_context_swap=False,
    )[0]
    assert sequence.condition is FactorialCondition.SC_SP
    captured = capture_factorial(adapter(), sequence)
    direct = adapter().model(
        document.token_ids.reshape(1, -1),
        position_ids=torch.arange(6).reshape(1, -1),
        capture_activations=True,
    )
    expected = sorted(
        (record for record in direct.activations if record.token_position >= 3),
        key=lambda record: (record.token_position, record.layer_idx),
    )
    logits = torch.stack([record.router_logits.reshape(-1) for record in expected]).reshape(
        3, 2, -1
    )
    topk = torch.stack([record.topk_ids.reshape(-1) for record in expected]).reshape(3, 2, -1)
    assert torch.equal(captured.router_logits, logits)
    assert torch.equal(captured.topk_ids, topk)


def test_metrics_and_paired_bootstrap_are_deterministic() -> None:
    documents = (
        OfflineDocument("a", torch.tensor([1, 2, 3, 4, 5, 6])),
        OfflineDocument("b", torch.tensor([7, 8, 9, 10, 11, 12])),
    )
    sequences = build_factorial_sequences(
        documents,
        boundary=3,
        horizon=3,
        vocab_size=32,
        content_seed=4,
        position_offset=5,
        include_context_swap=False,
    )
    captures = {
        (item.sample_id, item.condition): capture_factorial(adapter(), item) for item in sequences
    }
    rows = []
    for item in sequences:
        rows.extend(
            compare_capture(
                captures[(item.sample_id, item.condition)],
                captures[(item.sample_id, FactorialCondition.SC_SP)],
                horizons=(1, 2, 3),
                top_k=2,
            )
        )
    rows = add_position_dominance(rows)
    first = paired_bootstrap(rows, bootstrap_samples=50, confidence=0.9, seed=123)
    second = paired_bootstrap(rows, bootstrap_samples=50, confidence=0.9, seed=123)
    assert first == second
    assert any(row.condition == "POSITION_DOMINANCE" for row in first)
    scsp = [row for row in first if row.condition == "SC_SP" and row.metric == "topk_recall"]
    assert scsp and all(row.mean == 1.0 and row.ci_lower == row.ci_upper == 1.0 for row in scsp)
