from pathlib import Path
import os
import subprocess
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from convert_to_yolo import CLASS_NAMES, convert_json, output_stem, write_data_yaml
from experiment_results import RESULT_FIELDS, append_result, metric_values, write_class_metrics
from ultralytics.nn.modules import LSKNet
from train import build_parser, load_config, validate_resume_checkpoint, write_trainval_only_data
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.loss import v8DetectionLoss


def test_model_configs_use_official_yolo26_obb_weights():
    expected_runtime = {
        "n": {"epochs": 700, "imgsz": 1024, "batch": 64, "patience": 200, "save_period": 50, "cos_lr": True},
        "s": {"epochs": 500, "imgsz": 1024, "batch": 64, "patience": 200, "save_period": 50, "cos_lr": True},
        "m": {"epochs": 1500, "imgsz": 1280, "batch": 96, "patience": 300, "save_period": 50, "cos_lr": True},
    }
    for size in "nsm":
        config = load_config(size, 0)
        assert config["task"] == "obb"
        assert config["model"] == f"yolo26{size}-obb.pt"
        assert config["epochs"] == expected_runtime[size]["epochs"]
        assert config["patience"] == expected_runtime[size]["patience"]
        assert config["imgsz"] == expected_runtime[size]["imgsz"]
        assert config["batch"] == expected_runtime[size]["batch"]
        assert config["cos_lr"] is expected_runtime[size]["cos_lr"]
        assert config["save"] is True
        assert config["save_period"] == expected_runtime[size]["save_period"]
        assert config["plot_period"] == 50

    for size in "nsm":
        config = load_config(size, 0)
        assert config["optimizer"] == "AdamW"
        assert config["cls_pw"] == 0.25
        assert config["focal_gamma"] == 1.5
        assert config["focal_alpha"] == 0.25
        assert config["mosaic"] == 0.25
        assert config["flipud"] == 0.5

    full_m = load_config("m", 0)
    assert full_m["lr0"] == 0.0012
    assert full_m["workers"] == 8
    assert full_m["device"] == "0,1,2,3,4,5,6,7"
    assert full_m["lrf"] == 0.005
    assert full_m["warmup_epochs"] == 5.0
    assert full_m["close_mosaic"] == 150
    assert full_m["degrees"] == 180.0


def test_lsknet_configs_use_p2_head_and_competition_validation_settings():
    for variant, batch in (("lsknet-t", 16), ("lsknet-s", 8)):
        config = load_config(variant, 0)
        assert config["task"] == "obb"
        assert config["model"] == f"configs/models/{variant.replace('-', '_')}_obb.yaml"
        assert config["imgsz"] == 1280
        assert config["batch"] == batch
        assert config["device"] == "4,5,6,7"
        assert config["optimizer"] == "AdamW"
        assert config["conf"] == 0.05
        assert config["iou"] == 0.7
        assert config["max_det"] == 600
        model_yaml = yaml.safe_load((Path(config["model"])).read_text(encoding="utf-8"))
        assert model_yaml["backbone"][0][2] == "LSKNet"
        assert model_yaml["head"][-1][0] == [14, 17, 20, 23]
        assert model_yaml["head"][-1][2] == "OBB"
        assert "huggingface.co/GreatBird/LSKNet" in model_yaml["pretrained_backbone"]


def test_lsknet_backbone_feature_shapes():
    model = LSKNet(
        embed_dims=[8, 16, 24, 32],
        mlp_ratios=[2, 2, 2, 2],
        depths=[1, 1, 1, 1],
        drop_rate=0.0,
        drop_path_rate=0.0,
    ).eval()
    with torch.no_grad():
        features = model(torch.zeros(1, 3, 64, 64))
    assert [tuple(feature.shape) for feature in features] == [
        (1, 8, 16, 16),
        (1, 16, 8, 8),
        (1, 24, 4, 4),
        (1, 32, 2, 2),
    ]


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


def test_nonfinite_tensor_detection():
    assert BaseTrainer._tensors_finite([torch.tensor([1.0]), torch.tensor([2], dtype=torch.int64)])
    assert not BaseTrainer._tensors_finite([torch.tensor([float("nan")])])
    assert not BaseTrainer._tensors_finite([torch.tensor([float("inf")])])


def test_training_cli_exposes_stopping_and_checkpoint_controls():
    args = build_parser().parse_args(
        ["--model", "m", "--patience", "12", "--save-period", "3", "--resume", "checkpoint.pt"]
    )
    assert args.model == "m"
    assert args.patience == 12
    assert args.save_period == 3
    assert args.resume == "checkpoint.pt"
    assert build_parser().parse_args(["--model", "lsknet-t"]).model == "lsknet-t"


def test_ddp_subprocess_can_import_project_trainer_from_an_external_directory(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from lsknet_support import LSKNetOBBTrainer; print(LSKNetOBBTrainer.__name__)",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "LSKNetOBBTrainer"


def test_trainval_only_data_removes_test_split(tmp_path: Path, monkeypatch):
    data_path = tmp_path / "data.yaml"
    data_path.write_text(
        yaml.safe_dump({"path": ".", "train": "train/images", "val": "val/images", "test": "test/images"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("train.YOLO_CONFIG_ROOT", tmp_path / "runtime")
    output = write_trainval_only_data(data_path, fold=3)
    runtime_data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert runtime_data["train"] == "train/images"
    assert runtime_data["val"] == "val/images"
    assert "test" not in runtime_data
    assert Path(runtime_data["path"]).is_absolute()


def test_resume_checkpoint_validation(tmp_path: Path):
    resumable = tmp_path / "epoch10.pt"
    torch.save({"epoch": 10, "optimizer": {"state": {}}}, resumable)
    assert validate_resume_checkpoint(resumable)["epoch"] == 10

    stripped = tmp_path / "last.pt"
    torch.save({"epoch": -1, "optimizer": None}, stripped)
    with __import__("pytest").raises(ValueError, match="epochN.pt"):
        validate_resume_checkpoint(stripped)


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


def test_result_table_migrates_old_header_and_records_competition_metrics(tmp_path: Path):
    table = tmp_path / "experiments.csv"
    table.write_text("timestamp_utc,stage,run_name\nold,test,baseline\n", encoding="utf-8")
    append_result(
        table,
        {
            "stage": "train_val",
            "run_name": "lsknet_t",
            "competition_f1_03": 0.75,
            "competition_conf": 0.3,
        },
    )
    with table.open(encoding="utf-8") as stream:
        rows = list(__import__("csv").DictReader(stream))
    assert list(rows[0]) == RESULT_FIELDS
    assert rows[0]["run_name"] == "baseline"
    assert rows[1]["competition_f1_03"] == "0.75"
