import csv
from pathlib import Path

import yaml

from pseudoroute.cli import main


def test_m7_calibration_and_common_target_evaluation(tmp_path: Path) -> None:
    tiny = str(Path("configs/model/tiny_moe.yaml").resolve())
    documents = [[(start + step) % 31 + 1 for step in range(5)] for start in range(4)]
    train_config = {
        "schema_version": 1,
        "experiment": {"name": "train", "seed": 5, "information_regime": "offline_teacher_forced"},
        "tiny_model_config": tiny,
        "documents": documents,
        "horizons": [1],
        "history_length": 2,
        "split": {"train_fraction": 0.5, "validation_fraction": 0.25},
        "predictors": {
            "ridge_lambda": 0.01,
            "mlp_hidden": 4,
            "mlp_epochs": 1,
            "learning_rate": 0.01,
            "rf_trees": 2,
            "rf_feature_candidates": 2,
            "markov_smoothing": 0.001,
        },
    }
    train_path = tmp_path / "train.yaml"
    train_path.write_text(yaml.safe_dump(train_config))
    predictors = tmp_path / "predictors"
    assert (
        main(["train-predictor", "--config", str(train_path), "--output-dir", str(predictors)]) == 0
    )
    calibration_config = {
        "schema_version": 1,
        "experiment": {
            "name": "defaults",
            "seed": 5,
            "information_regime": "offline_teacher_forced",
        },
        "tiny_model_config": tiny,
        "documents": documents,
    }
    calibration_path = tmp_path / "defaults.yaml"
    calibration_path.write_text(yaml.safe_dump(calibration_config))
    defaults = tmp_path / "defaults"
    assert (
        main(
            [
                "build-default-vectors",
                "--config",
                str(calibration_path),
                "--output-dir",
                str(defaults),
            ]
        )
        == 0
    )
    evaluation_config = {
        "schema_version": 1,
        "experiment": {"name": "shadow", "seed": 5, "information_regime": "online_post_sample"},
        "training_config": str(train_path),
        "predictor_dataset_dir": str(predictors / "dataset"),
        "default_vectors_dir": str(defaults),
        "anchors": [1],
        "top_b": 1,
        "latency_repetitions": 1,
        "wrong_position_offset": 2,
        "fixed_token_id": 0,
    }
    evaluation_path = tmp_path / "evaluation.yaml"
    evaluation_path.write_text(yaml.safe_dump(evaluation_config))
    output = tmp_path / "evaluation"
    assert (
        main(["evaluate-probe", "--config", str(evaluation_path), "--output-dir", str(output)]) == 0
    )
    with (output / "probe_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert {row["information_regime"] for row in rows} == {"online_post_sample"}
    assert (output / "subset_regret_curves.svg").exists()
