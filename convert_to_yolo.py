#!/usr/bin/env python3
"""Convert the project JSON polygons to Ultralytics YOLO OBB format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_NAMES = [
    "Bus",
    "Cargo Truck",
    "Dump Truck",
    "Excavator",
    "Small Car",
    "Tractor",
    "Trailer",
    "Truck Tractor",
    "Van",
    "other-vehicle",
]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE_CACHE: dict[Path, tuple[int, int]] = {}


def resolve_image_path(raw_path: str, json_path: Path, dataset_root: Path) -> Path | None:
    """Resolve absolute paths and the common relative-path layouts used by the source JSON."""
    raw = Path(raw_path).expanduser()
    candidates = [raw] if raw.is_absolute() else [json_path.parent / raw, dataset_root / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def output_stem(raw_path: str) -> str:
    """Match the YOLO11 project's path-derived output filename exactly."""
    return raw_path.replace("/", "_").replace("\\", "_").replace(".tif", "").replace(" ", "_").replace(":", "_")


def link_or_copy(source: Path, destination: Path, copy_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    if copy_images:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source)


def convert_json(json_path: Path, output_dir: Path, dataset_root: Path, copy_images: bool = False) -> dict:
    """Convert one annotation JSON and return conversion statistics."""
    with json_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    annotations = payload["data"] if isinstance(payload, dict) else payload

    grouped: dict[str, list[dict]] = defaultdict(list)
    for annotation in annotations:
        grouped[annotation["data_path"]].append(annotation)

    image_dir = output_dir / "images"
    label_dir = output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "annotations": 0, "missing_images": 0, "invalid_annotations": 0, "unknown_classes": 0}

    for raw_path, image_annotations in grouped.items():
        source = resolve_image_path(raw_path, json_path, dataset_root)
        if source is None:
            stats["missing_images"] += 1
            continue
        size = IMAGE_SIZE_CACHE.get(source)
        if size is None:
            image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if image is None:
                stats["missing_images"] += 1
                continue
            height, width = image.shape[:2]
            IMAGE_SIZE_CACHE[source] = (height, width)
        else:
            height, width = size
        if width <= 0 or height <= 0:
            stats["missing_images"] += 1
            continue

        stem = output_stem(raw_path)
        link_or_copy(source, image_dir / f"{stem}.tif", copy_images)
        lines: list[str] = []

        for annotation in image_annotations:
            class_id = CLASS_TO_ID.get(annotation.get("lab", ""))
            if class_id is None:
                stats["unknown_classes"] += 1
                continue
            # Match the YOLO11 baseline: direct normalization with no clipping,
            # point reordering, min-area conversion, or annotation filtering.
            points = np.asarray(annotation["points"][:4], dtype=np.float32)
            points[:, 0] /= width
            points[:, 1] /= height
            coordinates = " ".join(f"{value:.6f}" for value in points.reshape(-1))
            lines.append(f"{class_id} {coordinates}")

        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["images"] += 1
        stats["annotations"] += len(lines)
    return stats


def write_data_yaml(fold_dir: Path, has_test: bool) -> None:
    data = {
        "path": str(fold_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }
    if has_test:
        data["test"] = "../test/images"
    with (fold_dir / "data.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="JSON Polygon 转 YOLO OBB 八点格式")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--all", action="store_true", help="转换 fold 0-4")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "dataset_yolo")
    parser.add_argument("--copy-images", action="store_true", help="复制图像而不是创建绝对符号链接")
    args = parser.parse_args()

    dataset_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"原始数据目录不存在: {dataset_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    test_json = dataset_root / "test.json"
    has_test = test_json.is_file()
    if has_test:
        stats = convert_json(test_json, output_root / "test", dataset_root, args.copy_images)
        print(f"test: {stats}")

    folds = range(5) if args.all else [args.fold if args.fold is not None else 0]
    for fold in folds:
        source_fold = dataset_root / f"fold_{fold}"
        train_json = source_fold / "train.json"
        val_json = source_fold / "val.json"
        if not train_json.is_file() or not val_json.is_file():
            print(f"跳过 fold {fold}: 缺少 {train_json} 或 {val_json}")
            continue
        target_fold = output_root / f"fold_{fold}"
        train_stats = convert_json(train_json, target_fold / "train", dataset_root, args.copy_images)
        val_stats = convert_json(val_json, target_fold / "val", dataset_root, args.copy_images)
        write_data_yaml(target_fold, has_test)
        print(f"fold {fold} train: {train_stats}")
        print(f"fold {fold} val:   {val_stats}")


if __name__ == "__main__":
    main()
