import csv
from pathlib import Path

import yaml

from pseudoroute.cli import main


def test_train_and_evaluate_probe_cli(tmp_path: Path) -> None:
    train = {
        "schema_version": 1,
        "experiment": {"name": "test", "seed": 4, "information_regime": "offline_teacher_forced"},
        "tiny_model_config": str(Path("configs/model/tiny_moe.yaml").resolve()),
        "documents": [[(start + step) % 31 + 1 for step in range(7)] for start in range(6)],
        "horizons": [1, 2],
        "history_length": 2,
        "split": {"train_fraction": 0.5, "validation_fraction": 0.1666667},
        "predictors": {
            "ridge_lambda": 0.01,
            "mlp_hidden": 8,
            "mlp_epochs": 2,
            "learning_rate": 0.01,
            "rf_trees": 3,
            "rf_feature_candidates": 3,
            "markov_smoothing": 0.001,
        },
    }
    train_path = tmp_path / "train.yaml"
    train_path.write_text(yaml.safe_dump(train))
    models = tmp_path / "models"
    assert main(["train-predictor", "--config", str(train_path), "--output-dir", str(models)]) == 0
    evaluation = {
        "schema_version": 1,
        "experiment": {"name": "test", "seed": 4, "information_regime": "online_pre_sample"},
        "training_config": str(train_path),
        "predictor_dir": str(models),
        "top_b": 2,
        "latency_repetitions": 1,
        "rolling_lookback": 2,
        "rolling_decay": 0.9,
    }
    evaluation_path = tmp_path / "evaluation.yaml"
    evaluation_path.write_text(yaml.safe_dump(evaluation))
    output = tmp_path / "evaluation"
    assert (
        main(["evaluate-probe", "--config", str(evaluation_path), "--output-dir", str(output)]) == 0
    )
    with (output / "probe_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {row["information_regime"] for row in rows} == {"online_pre_sample"}
    assert len({row["equal_cost_ceiling_us"] for row in rows}) == 1
    assert (output / "quality_vs_latency.svg").exists()
