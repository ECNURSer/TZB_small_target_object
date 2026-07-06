#!/usr/bin/env python3
"""Evaluate one trained checkpoint on the independent test split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_CONFIG_ROOT = PROJECT_ROOT / ".ultralytics"
YOLO_CONFIG_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics_src"))

from ultralytics import YOLO  # noqa: E402

from experiment_results import append_result, metric_values, write_class_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26 OBB 独立 test 集评估")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--data", type=Path, help="覆盖默认 fold data.yaml")
    parser.add_argument("--model", choices=("n", "s", "m"), required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name")
    parser.add_argument("--results-csv", type=Path, default=PROJECT_ROOT / "results" / "experiments.csv")
    args = parser.parse_args()

    weights = args.weights.expanduser().resolve()
    data = (
        args.data.expanduser().resolve()
        if args.data is not None
        else PROJECT_ROOT / "dataset_yolo" / f"fold_{args.fold}" / "data.yaml"
    )
    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"数据配置不存在: {data}")
    with data.open(encoding="utf-8") as stream:
        data_config = yaml.safe_load(stream)
    if not data_config.get("test"):
        raise ValueError(f"数据配置没有独立 test 路径: {data}")

    run_name = args.name or f"test_yolo26{args.model}_obb_fold{args.fold}"
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(PROJECT_ROOT / "runs" / "test"),
        name=run_name,
        exist_ok=True,
        plots=True,
        save_json=False,
    )
    values = metric_values(metrics)
    append_result(
        args.results_csv.expanduser().resolve(),
        {
            "stage": "test",
            "run_name": run_name,
            "model": f"yolo26{args.model}-obb.pt",
            "fold": args.fold,
            "split": "test",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "weights": str(weights),
            "params_m": sum(parameter.numel() for parameter in model.model.parameters()) / 1_000_000,
            "results_dir": str(PROJECT_ROOT / "runs" / "test" / run_name),
            **values,
        },
    )
    output = PROJECT_ROOT / "runs" / "test" / run_name / "test_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
    write_class_metrics(metrics, output.with_name("test_class_metrics.csv"))
    print(json.dumps(values, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
