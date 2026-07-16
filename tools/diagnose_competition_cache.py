#!/usr/bin/env python3
"""Diagnose competition F1@0.3 by class and pixel scale from a prediction cache."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_CONFIG_ROOT = PROJECT_ROOT / ".ultralytics"
YOLO_CONFIG_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(PROJECT_ROOT))

from competition_scoring import MatchRecord, ObjectAnnotation, best_confidence, load_yolo_ground_truth, score_records  # noqa: E402


@dataclass(frozen=True)
class DiagnosticRecord:
    confidence: float
    class_id: int
    true_positive: bool
    gt_scale_bin: str | None
    pred_scale_bin: str


SCALE_BINS = (
    (0.0, 16.0, "lt16"),
    (16.0, 32.0, "16_32"),
    (32.0, 64.0, "32_64"),
    (64.0, 128.0, "64_128"),
    (128.0, math.inf, "ge128"),
)


def polygon_area(points: tuple[tuple[float, float], ...] | list[list[float]]) -> float:
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return abs(float(cv2.contourArea(contour)))


def scale_bin(points: tuple[tuple[float, float], ...] | list[list[float]]) -> str:
    scale = math.sqrt(max(polygon_area(points), 0.0))
    for low, high, name in SCALE_BINS:
        if low <= scale < high:
            return name
    raise AssertionError(f"unreachable scale bin: {scale}")


def resolve_split(data_file: Path, split: str) -> tuple[list[Path], Path]:
    data = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_file.parent))
    if not root.is_absolute():
        root = (data_file.parent / root).resolve()
    split_value = data.get(split)
    if not isinstance(split_value, str):
        raise ValueError(f"Data YAML does not define split={split}: {data_file}")
    image_dir = Path(split_value)
    if not image_dir.is_absolute():
        image_dir = (root / image_dir).resolve()
    label_dir = image_dir.parent / "labels"
    return sorted(path for path in image_dir.iterdir() if path.is_file()), label_dir


def match_image(
    predictions: list[ObjectAnnotation],
    ground_truths: list[ObjectAnnotation],
    iou_threshold: float,
) -> tuple[list[DiagnosticRecord], dict[int, int], dict[str, int], dict[tuple[int, str], int]]:
    gt_by_class: dict[int, list[ObjectAnnotation]] = defaultdict(list)
    for target in ground_truths:
        gt_by_class[target.class_id].append(target)
    matched = {class_id: np.zeros(len(items), dtype=bool) for class_id, items in gt_by_class.items()}
    target_bounds = {
        class_id: np.asarray(
            [
                (
                    min(point[0] for point in target.polygon),
                    min(point[1] for point in target.polygon),
                    max(point[0] for point in target.polygon),
                    max(point[1] for point in target.polygon),
                )
                for target in items
            ],
            dtype=np.float32,
        )
        for class_id, items in gt_by_class.items()
    }
    target_scale_bins = {
        class_id: [scale_bin(target.polygon) for target in items] for class_id, items in gt_by_class.items()
    }
    gt_by_scale: dict[str, int] = defaultdict(int)
    gt_by_class_scale: dict[tuple[int, str], int] = defaultdict(int)
    gt_by_class_count: dict[int, int] = defaultdict(int)
    for class_id, items in gt_by_class.items():
        gt_by_class_count[class_id] += len(items)
        for item in items:
            bin_name = scale_bin(item.polygon)
            gt_by_scale[bin_name] += 1
            gt_by_class_scale[(class_id, bin_name)] += 1

    records: list[DiagnosticRecord] = []
    ordered = sorted(enumerate(predictions), key=lambda item: (-item[1].confidence, item[0]))
    for _, prediction in ordered:
        targets = gt_by_class.get(prediction.class_id, [])
        available = np.flatnonzero(~matched.get(prediction.class_id, np.empty(0, dtype=bool)))
        is_tp = False
        gt_bin = None
        if len(available):
            points = np.asarray(prediction.polygon, dtype=np.float32)
            px1, py1 = points.min(axis=0)
            px2, py2 = points.max(axis=0)
            bounds = target_bounds[prediction.class_id][available]
            overlaps = (bounds[:, 0] < px2) & (bounds[:, 2] > px1) & (bounds[:, 1] < py2) & (bounds[:, 3] > py1)
            available = available[overlaps]
        if len(available):
            ious = np.asarray(
                [
                    cv2.intersectConvexConvex(
                        cv2.convexHull(np.asarray(prediction.polygon, dtype=np.float32).reshape(-1, 2)),
                        cv2.convexHull(np.asarray(targets[index].polygon, dtype=np.float32).reshape(-1, 2)),
                    )[0]
                    for index in available
                ],
                dtype=np.float32,
            )
            pred_area = polygon_area(prediction.polygon)
            target_areas = np.asarray([polygon_area(targets[index].polygon) for index in available], dtype=np.float32)
            unions = pred_area + target_areas - ious
            ious = np.where(unions > 0, ious / unions, 0.0)
            best_position = int(ious.argmax())
            if float(ious[best_position]) >= iou_threshold:
                target_index = int(available[best_position])
                matched[prediction.class_id][target_index] = True
                is_tp = True
                gt_bin = target_scale_bins[prediction.class_id][target_index]
        records.append(
            DiagnosticRecord(
                confidence=prediction.confidence,
                class_id=prediction.class_id,
                true_positive=is_tp,
                gt_scale_bin=gt_bin,
                pred_scale_bin=scale_bin(prediction.polygon),
            )
        )
    return records, dict(gt_by_class_count), dict(gt_by_scale), dict(gt_by_class_scale)


def score_bucket(records: list[DiagnosticRecord], total_gt: int, confidence: float) -> dict[str, float | int]:
    selected = [record for record in records if record.confidence >= confidence]
    tp = sum(record.true_positive for record in selected)
    fp = len(selected) - tp
    fn = max(total_gt - tp, 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"gt": total_gt, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="按类别和像素尺度诊断比赛 F1@0.3")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--fixed-conf", type=float)
    args = parser.parse_args()

    payload = json.loads(args.cache.read_text(encoding="utf-8"))
    image_paths, label_dir = resolve_split(args.data, args.split)
    expected_ids = [path.stem for path in image_paths]
    cached_ids = [image["image_id"] for image in payload["images"]]
    if expected_ids != cached_ids:
        raise ValueError("cache image IDs do not match the selected split")

    all_records: list[DiagnosticRecord] = []
    total_gt_by_class: dict[int, int] = defaultdict(int)
    total_gt_by_scale: dict[str, int] = defaultdict(int)
    total_gt_by_class_scale: dict[tuple[int, str], int] = defaultdict(int)
    for image in payload["images"]:
        predictions = [
            ObjectAnnotation(
                class_id=int(item["class_id"]),
                confidence=float(item["confidence"]),
                polygon=tuple((float(x), float(y)) for x, y in item["polygon"]),
            )
            for item in image["predictions"]
        ]
        targets = load_yolo_ground_truth(label_dir / f"{image['image_id']}.txt", int(image["width"]), int(image["height"]))
        records, gt_class, gt_scale, gt_class_scale = match_image(predictions, targets, args.match_iou)
        all_records.extend(records)
        for key, value in gt_class.items():
            total_gt_by_class[key] += value
        for key, value in gt_scale.items():
            total_gt_by_scale[key] += value
        for key, value in gt_class_scale.items():
            total_gt_by_class_scale[key] += value

    total_gt = sum(total_gt_by_class.values())
    scoring_records = [
        MatchRecord(record.confidence, record.class_id, record.true_positive) for record in all_records
    ]
    if args.fixed_conf is None:
        selected = best_confidence(scoring_records, total_gt)
        confidence = selected.confidence
    else:
        confidence = args.fixed_conf
        selected = score_records(scoring_records, total_gt, confidence)

    names = {int(key): value for key, value in payload.get("class_names", {}).items()}
    overall = {"confidence": confidence, **score_bucket(all_records, total_gt, confidence)}
    class_rows = []
    for class_id in sorted(set(total_gt_by_class) | {record.class_id for record in all_records}):
        bucket = [record for record in all_records if record.class_id == class_id]
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                **score_bucket(bucket, total_gt_by_class.get(class_id, 0), confidence),
            }
        )
    scale_rows = []
    for bin_name in [item[2] for item in SCALE_BINS]:
        bucket = [
            record
            for record in all_records
            if (record.true_positive and record.gt_scale_bin == bin_name)
            or ((not record.true_positive) and record.pred_scale_bin == bin_name)
        ]
        scale_rows.append({"scale_bin": bin_name, **score_bucket(bucket, total_gt_by_scale.get(bin_name, 0), confidence)})
    class_scale_rows = []
    for class_id, bin_name in sorted(total_gt_by_class_scale):
        bucket = [
            record
            for record in all_records
            if record.class_id == class_id
            and (
                (record.true_positive and record.gt_scale_bin == bin_name)
                or ((not record.true_positive) and record.pred_scale_bin == bin_name)
            )
        ]
        class_scale_rows.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "scale_bin": bin_name,
                **score_bucket(bucket, total_gt_by_class_scale.get((class_id, bin_name), 0), confidence),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metric": f"F1@polygon-IoU{args.match_iou:g}",
        "split": args.split,
        "threshold_mode": "fixed" if args.fixed_conf is not None else "optimized_on_this_split",
        "overall": overall,
        "weights": payload.get("weights"),
        "imgsz": payload.get("imgsz"),
        "nms_iou": payload.get("nms_iou"),
        "max_det": payload.get("max_det"),
        "image_count": payload.get("image_count"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(
        args.output_dir / "by_class.csv",
        class_rows,
        ["class_id", "class_name", "gt", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    write_csv(
        args.output_dir / "by_scale.csv",
        scale_rows,
        ["scale_bin", "gt", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    write_csv(
        args.output_dir / "by_class_scale.csv",
        class_scale_rows,
        ["class_id", "class_name", "scale_bin", "gt", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
