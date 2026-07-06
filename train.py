#!/usr/bin/env python3
"""Train official Ultralytics YOLO26 OBB n/s/m models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_CONFIG_ROOT = PROJECT_ROOT / ".ultralytics"
YOLO_CONFIG_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics_src"))

from ultralytics import YOLO  # noqa: E402

from experiment_results import append_result, metric_values, write_class_metrics  # noqa: E402
from tensorboard_support import enable_tensorboard_scalars  # noqa: E402


def load_config(size: str, fold: int) -> dict:
    """Load a model-size configuration and attach the selected fold dataset."""
    config_path = PROJECT_ROOT / "configs" / f"yolo26{size}_obb.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["data"] = str(PROJECT_ROOT / "dataset_yolo" / f"fold_{fold}" / "data.yaml")
    return config


def write_trainval_only_data(data_path: Path, fold: int) -> Path:
    """Write a runtime data YAML containing train/val only, with no test path."""
    with data_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not data.get("train") or not data.get("val"):
        raise ValueError(f"数据配置必须包含 train 和 val: {data_path}")
    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    data["path"] = str(root)
    data.pop("test", None)
    output = YOLO_CONFIG_ROOT / "train_data" / f"fold_{fold}_trainval.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练官方 YOLO26 OBB 模型")
    parser.add_argument("--model", choices=("n", "s", "m"), default="n", help="模型规模")
    parser.add_argument("--fold", type=int, choices=range(5), default=0, help="交叉验证 fold")
    parser.add_argument("--data", type=Path, help="覆盖默认 fold data.yaml，用于检查或自定义数据")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", help="例如 0 或 0,1,2,3")
    parser.add_argument("--patience", type=int, help="早停等待轮数，0 表示关闭早停")
    parser.add_argument("--save-period", type=int, help="每 N 个 epoch 额外保存 epochN.pt，-1 表示关闭")
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--cache", choices=("ram", "disk"))
    parser.add_argument("--name")
    parser.add_argument("--no-test", action="store_true", help="生成仅含 train/val 的运行时 data.yaml")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="从 checkpoint 恢复；不传路径时使用当前实验的 last.pt",
    )
    parser.add_argument("--results-csv", type=Path, default=PROJECT_ROOT / "results" / "experiments.csv")
    parser.add_argument("--dry-run", action="store_true", help="只检查并打印最终配置")
    return parser


def validate_resume_checkpoint(checkpoint: Path) -> dict:
    """Load checkpoint metadata and reject inference-only or completed checkpoints."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = state.get("epoch") if isinstance(state, dict) else None
    optimizer = state.get("optimizer") if isinstance(state, dict) else None
    if not isinstance(epoch, int) or epoch < 0 or optimizer is None:
        raise ValueError(
            f"断点不可续训: {checkpoint} 缺少有效 epoch/optimizer 状态。"
            "该权重已完成训练或已被 strip；请改用最近的 epochN.pt。"
        )
    return state


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.model, args.fold)
    if args.data is not None:
        config["data"] = str(args.data.expanduser().resolve())
    run_name = args.name or f"yolo26{args.model}_obb_fold{args.fold}"
    config.update(project=str(PROJECT_ROOT / "runs"), name=run_name)

    for key in (
        "epochs",
        "batch",
        "imgsz",
        "workers",
        "device",
        "patience",
        "save_period",
        "fraction",
        "cache",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    data_path = Path(config["data"])
    if not data_path.is_file():
        raise FileNotFoundError(f"数据配置不存在: {data_path}。请先运行 python convert_to_yolo.py --all")
    if args.no_test:
        data_path = write_trainval_only_data(data_path.resolve(), args.fold)
        config["data"] = str(data_path)
        print(f"数据泄漏保护: 已移除 test split，训练仅可访问 train/val: {data_path}")
    if config["patience"] < 0:
        raise ValueError("patience 必须大于或等于 0")
    if config["save_period"] == 0 or config["save_period"] < -1:
        raise ValueError("save_period 必须为 -1 或正整数")

    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(
        f"权重策略: last.pt 每轮覆盖；best.pt 在验证 fitness 提升时覆盖；"
        f"epochN.pt 每 {config['save_period']} 轮保存一次。"
    )
    if args.dry_run:
        print("配置检查通过；dry-run 未加载权重或启动训练。")
        return

    plot_period = int(config.pop("plot_period", 0))
    if plot_period < 0:
        raise ValueError("plot_period 必须大于或等于 0")
    os.environ["YOLO_PLOT_PERIOD"] = str(plot_period)
    enable_tensorboard_scalars()
    if args.resume:
        last_pt = (
            PROJECT_ROOT / "runs" / run_name / "weights" / "last.pt"
            if args.resume == "auto"
            else Path(args.resume).expanduser().resolve()
        )
        if not last_pt.is_file():
            raise FileNotFoundError(f"断点不存在: {last_pt}")
        validate_resume_checkpoint(last_pt)
        model = YOLO(str(last_pt))
        resume_overrides = {
            key: config[key]
            for key in ("imgsz", "batch", "device", "workers", "cache", "patience", "save_period")
            if key in config
        }
        if args.epochs is not None:
            resume_overrides["epochs"] = args.epochs
        metrics = model.train(resume=True, **resume_overrides)
    else:
        model = YOLO(config.pop("model"))
        metrics = model.train(**config)

    save_dir = Path(model.trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    params_m = sum(parameter.numel() for parameter in model.model.parameters()) / 1_000_000
    write_class_metrics(metrics, save_dir / "val_class_metrics.csv")
    append_result(
        args.results_csv.expanduser().resolve(),
        {
            "stage": "train_val",
            "run_name": run_name,
            "model": f"yolo26{args.model}-obb.pt",
            "fold": args.fold,
            "split": "val",
            "epochs": config.get("epochs", "resume"),
            "imgsz": config.get("imgsz", ""),
            "batch": config.get("batch", ""),
            "weights": str(best_pt),
            "params_m": params_m,
            "results_dir": str(save_dir),
            **metric_values(metrics),
        },
    )
    print(f"训练完成: {save_dir}")
    print(f"TensorBoard: tensorboard --logdir {PROJECT_ROOT / 'runs'} --port 6006")


if __name__ == "__main__":
    main()
