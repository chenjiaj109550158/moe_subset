from pathlib import Path

import pytest

from pseudoroute.benchmark.pseudo_embedding_analysis_config import (
    load_analysis_config,
)

CONFIG = Path("configs/analysis/pseudo_embedding_calibration_free_analysis_v1.yaml")


def test_calibration_free_analysis_scope_is_frozen() -> None:
    config = load_analysis_config(CONFIG)
    assert config.fingerprint() == (
        "c09d70901bacafabdbacda398390574f7c24f67f634909d39b2084be3e04f898"
    )
    assert config.operating_point.horizon == 8
    assert config.operating_point.budget == 32
    assert len(config.source.samples) == 4
    assert not config.candidate_progress_reference.selection_uses_accuracy
    assert config.calibration_free.learned_parameters == "forbidden"
    assert config.calibration_free.offline_expert_priors == "forbidden"


def test_analysis_config_rejects_unknown_fields(tmp_path: Path) -> None:
    broken = tmp_path / "analysis.yaml"
    broken.write_text(CONFIG.read_text(encoding="utf-8") + "unknown: true\n")
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_analysis_config(broken)
