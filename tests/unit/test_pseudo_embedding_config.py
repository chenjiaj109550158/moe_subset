from pathlib import Path

import pytest

from pseudoroute.benchmark.pseudo_embedding_config import (
    load_pseudo_embedding_config,
    load_sample_manifest,
)

CONFIG = Path("configs/benchmark/pseudo_embedding_qwen_gsm8k_v1.yaml")
MANIFEST = Path("configs/benchmark/pseudo_embedding_qwen_gsm8k_v1_samples.json")


def test_pseudo_embedding_protocol_and_samples_are_frozen() -> None:
    suite = load_pseudo_embedding_config(CONFIG)
    manifest = load_sample_manifest(MANIFEST)
    assert suite.fingerprint() == (
        "a81f36b5ec4a4222ca7a459f9f9c1536d88bef5ba143c151c9e70498157d8cbc"
    )
    assert suite.operating_point.horizon == 8
    assert suite.operating_point.budget_per_layer == 32
    assert suite.default_vectors.unobserved_layer_expert_pairs == 444
    assert [len(partition.rows) for partition in manifest.partitions.values()] == [
        8,
        8,
        4,
        8,
        2,
    ]
    assert suite.accuracy_gate.vanilla_successes == 16
    assert suite.accuracy_gate.minimum_policy_successes == 15
    assert not suite.progress_gate.ranking_uses_task_accuracy


def test_sample_manifest_checksum_and_unknown_config_fields_fail(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "pseudo_embedding_qwen_gsm8k_v1.yaml"
    text = CONFIG.read_text(encoding="utf-8")
    broken.write_text(text + "unknown_field: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_pseudo_embedding_config(broken)
