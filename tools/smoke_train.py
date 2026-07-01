#!/usr/bin/env python3
"""Run a tiny CPU-only OBB training job to verify project plumbing."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_CONFIG_ROOT = PROJECT_ROOT / ".ultralytics"
YOLO_CONFIG_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics_src"))
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

from tensorboard_support import enable_tensorboard_scalars  # noqa: E402


def build_dataset(root: Path) -> Path:
    """Create two simple train/val images with valid normalized OBB labels."""
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(2):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.rectangle(image, (16, 22), (48, 42), (220, 220, 220), -1)
            cv2.imwrite(str(image_dir / f"sample_{index}.jpg"), image)
            (label_dir / f"sample_{index}.txt").write_text(
                "0 0.250000 0.343750 0.750000 0.343750 0.750000 0.656250 0.250000 0.656250\n",
                encoding="utf-8",
            )
    data = {
        "path": str(root),
        "train": "train/images",
        "val": "val/images",
        "test": "val/images",
        "names": {0: "vehicle"},
    }
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml


def main() -> None:
    dataset_root = PROJECT_ROOT / "runs" / "smoke_dataset"
    output_root = PROJECT_ROOT / "runs" / "smoke"
    run_name = "yolo26n_obb_cpu_smoke"
    data_yaml = build_dataset(dataset_root)
    enable_tensorboard_scalars()
    model = YOLO("yolo26n-obb.yaml")
    model.train(
        data=str(data_yaml),
        epochs=2,
        patience=1,
        save=True,
        save_period=1,
        imgsz=64,
        batch=2,
        workers=0,
        device="cpu",
        amp=False,
        cache=False,
        plots=False,
        val=True,
        project=str(output_root),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    run_dir = output_root / run_name
    weights = run_dir / "weights"
    checks = {
        "last_pt": (weights / "last.pt").is_file(),
        "best_pt": (weights / "best.pt").is_file(),
        "periodic_weights": sorted(path.name for path in weights.glob("epoch*.pt")),
        "tensorboard_events": sorted(path.name for path in run_dir.glob("events.out.tfevents.*")),
        "results_csv": (run_dir / "results.csv").is_file(),
    }
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    if not all((checks["last_pt"], checks["best_pt"], checks["periodic_weights"], checks["tensorboard_events"], checks["results_csv"])):
        raise RuntimeError("训练冒烟测试缺少预期输出")


if __name__ == "__main__":
    main()
