#!/usr/bin/env python3
"""Build a deterministic 72-image subset and run a no-augmentation YOLO26m OBB overfit test."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_CONFIG_ROOT = PROJECT_ROOT / ".ultralytics"
YOLO_CONFIG_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics_src"))
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

from convert_to_yolo import CLASS_NAMES  # noqa: E402
from tensorboard_support import enable_tensorboard_scalars  # noqa: E402


def build_subset(count: int, seed: int) -> tuple[Path, dict]:
    """Create one image/label directory used as both train and val."""
    source = PROJECT_ROOT / "dataset_yolo" / "fold_0" / "train"
    output = PROJECT_ROOT / "dataset_yolo" / f"overfit{count}"
    labels = sorted((source / "labels").glob("*.txt"))
    if len(labels) < count:
        raise ValueError(f"Only {len(labels)} source labels are available")
    random.Random(seed).shuffle(labels)
    selected = labels[:count]

    if output.exists():
        shutil.rmtree(output)
    image_output = output / "images"
    label_output = output / "labels"
    image_output.mkdir(parents=True)
    label_output.mkdir(parents=True)

    object_counts = [0] * len(CLASS_NAMES)
    image_counts = [0] * len(CLASS_NAMES)
    selection = []
    for label in selected:
        image_matches = list((source / "images").glob(f"{label.stem}.*"))
        if len(image_matches) != 1:
            raise ValueError(f"Expected one image for {label.stem}, found {len(image_matches)}")
        image = image_matches[0]
        (image_output / image.name).symlink_to(image.resolve())
        (label_output / label.name).symlink_to(label.resolve())
        classes = [int(line.split()[0]) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        for class_id in classes:
            object_counts[class_id] += 1
        for class_id in set(classes):
            image_counts[class_id] += 1
        selection.append(image.name)

    missing = [CLASS_NAMES[index] for index, value in enumerate(object_counts) if value == 0]
    if missing:
        raise ValueError(f"Subset does not cover all classes: {missing}")
    data = {
        "path": str(output),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }
    data_path = output / "data.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (output / "selection.txt").write_text("\n".join(selection) + "\n", encoding="utf-8")
    stats = {
        "images": count,
        "objects": sum(object_counts),
        "objects_per_class": dict(zip(CLASS_NAMES, object_counts)),
        "images_per_class": dict(zip(CLASS_NAMES, image_counts)),
        "data": str(data_path),
    }
    (output / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data_path, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26m OBB 72-image overfit sanity test")
    parser.add_argument("--count", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="1")
    parser.add_argument("--name", default="yolo26m_obb_overfit72_noaug")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    data_path, stats = build_subset(args.count, args.seed)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if args.prepare_only:
        return

    enable_tensorboard_scalars()
    model = YOLO(str(PROJECT_ROOT / "yolo26m-obb.pt"))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=4,
        device=args.device,
        cache="ram",
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        cos_lr=True,
        amp=True,
        patience=0,
        save=True,
        save_period=-1,
        seed=args.seed,
        deterministic=True,
        box=7.5,
        cls=0.75,
        cls_pw=0.0,
        focal_gamma=0.0,
        dfl=1.5,
        angle=1.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        close_mosaic=0,
        max_det=600,
        plots=True,
        project=str(PROJECT_ROOT / "runs"),
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
