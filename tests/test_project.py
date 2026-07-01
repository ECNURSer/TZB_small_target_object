from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from convert_to_yolo import CLASS_NAMES, convert_json, output_stem, write_data_yaml
from experiment_results import append_result, metric_values, write_class_metrics
from train import build_parser, load_config
from tools.compare_models import mean, std
from ultralytics.utils.loss import v8DetectionLoss


def test_model_configs_use_official_yolo26_obb_weights():
    expected_runtime = {
        "n": {"imgsz": 640, "batch": 144, "patience": 100},
        "s": {"imgsz": 640, "batch": 288, "patience": 100},
        "m": {"imgsz": 1024, "batch": 64, "patience": 80},
    }
    for size in "nsm":
        config = load_config(size, 0)
        assert config["task"] == "obb"
        assert config["model"] == f"yolo26{size}-obb.pt"
        assert config["patience"] == expected_runtime[size]["patience"]
        assert config["imgsz"] == expected_runtime[size]["imgsz"]
        assert config["batch"] == expected_runtime[size]["batch"]
        assert config["cos_lr"] is (size == "m")
        assert config["save"] is True
        assert config["save_period"] == 10
        assert config["plot_period"] == 50

    medium = load_config("m", 0)
    assert medium["optimizer"] == "AdamW"
    assert medium["cls_pw"] == 0.25
    assert medium["focal_gamma"] == 1.5
    assert medium["focal_alpha"] == 0.25
    assert medium["mosaic"] == 0.25
    assert medium["flipud"] == 0.5


def test_balanced_focal_classification_loss():
    criterion = v8DetectionLoss.__new__(v8DetectionLoss)
    criterion.bce = nn.BCEWithLogitsLoss(reduction="none")
    criterion.class_weights = torch.tensor([[[1.0, 2.0]]])
    predictions = torch.tensor([[[-6.0, 6.0]]])
    targets = torch.tensor([[[0.0, 1.0]]])

    criterion.focal_gamma = 0.0
    criterion.focal_alpha = 0.25
    weighted_bce = criterion.classification_loss(predictions, targets, 1)

    criterion.focal_gamma = 1.5
    focal = criterion.classification_loss(predictions, targets, 1)
    assert torch.isfinite(focal)
    assert 0 < focal < weighted_bce


def test_training_cli_exposes_stopping_and_checkpoint_controls():
    args = build_parser().parse_args(
        ["--model", "m", "--patience", "12", "--save-period", "3", "--resume", "checkpoint.pt"]
    )
    assert args.model == "m"
    assert args.patience == 12
    assert args.save_period == 3
    assert args.resume == "checkpoint.pt"


def test_convert_json_and_test_path(tmp_path: Path):
    dataset = tmp_path / "dataset"
    fold = dataset / "fold_0"
    fold.mkdir(parents=True)
    image_path = dataset / "sample.tif"
    cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
    payload = {
        "data": [
            {
                "data_path": "sample.tif",
                "lab": CLASS_NAMES[0],
                "points": [[10, 10], [30, 10], [30, 20], [10, 20], [10, 10]],
            }
        ]
    }
    json_path = fold / "train.json"
    json_path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    output = tmp_path / "dataset_yolo" / "fold_0" / "train"
    stats = convert_json(json_path, output, dataset)
    assert stats["images"] == 1
    assert stats["annotations"] == 1
    label = next((output / "labels").glob("*.txt")).read_text().split()
    assert len(label) == 9
    assert all(0 <= float(value) <= 1 for value in label[1:])
    assert output_stem("/data/work1/input path/12.tif") == "_data_work1_input_path_12"

    write_data_yaml(output.parent, has_test=True)
    data = yaml.safe_load((output.parent / "data.yaml").read_text())
    assert data["test"] == "../test/images"


def test_result_table(tmp_path: Path):
    class Metrics:
        results_dict = {"metrics/mAP50(B)": 0.5, "metrics/mAP50-95(B)": 0.3, "fitness": 0.3}

        @staticmethod
        def summary():
            return [{"Class": "vehicle", "mAP50": 0.5, "mAP50-95": 0.3}]

    values = metric_values(Metrics())
    assert values["map50"] == 0.5
    table = tmp_path / "experiments.csv"
    append_result(table, {"stage": "test", **values})
    assert len(table.read_text().splitlines()) == 2
    class_table = tmp_path / "classes.csv"
    write_class_metrics(Metrics(), class_table)
    assert "vehicle" in class_table.read_text()


def test_comparison_statistics_handle_missing_optional_metrics():
    rows = [{"map50_95": "0.4", "inference_ms": ""}, {"map50_95": "0.6", "inference_ms": ""}]
    assert mean(rows, "map50_95") == 0.5
    assert std(rows, "map50_95") > 0
    assert str(mean(rows, "inference_ms")) == "nan"
