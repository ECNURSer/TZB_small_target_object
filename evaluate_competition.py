#!/usr/bin/env python3
"""Generate low-confidence OBB predictions and evaluate competition F1 at polygon IoU 0.3."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_CONFIG_ROOT = PROJECT_ROOT / ".ultralytics"
YOLO_CONFIG_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics_src"))

from ultralytics import YOLO  # noqa: E402

from competition_scoring import (  # noqa: E402
    ObjectAnnotation,
    best_class_confidences,
    best_confidence,
    class_scores,
    load_yolo_ground_truth,
    merge_matches,
    score_records,
    score_to_dict,
)


def resolve_split(data_file: Path, split: str) -> tuple[list[Path], Path]:
    """Resolve an Ultralytics data YAML split to image paths and its label directory."""
    data = yaml.safe_load(data_file.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_file.parent))
    if not root.is_absolute():
        root = (data_file.parent / root).resolve()
    split_value = data.get(split)
    if not isinstance(split_value, str):
        raise ValueError(f"Data YAML does not define a single directory for split={split}: {data_file}")
    image_dir = Path(split_value)
    if not image_dir.is_absolute():
        image_dir = (root / image_dir).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    label_dir = image_dir.parent / "labels"
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    stems = [path.stem for path in image_paths]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Image stems must be unique for cached scoring: {image_dir}")
    return image_paths, label_dir


def generate_cache(args: argparse.Namespace, image_paths: list[Path], cache_path: Path) -> dict:
    """Run inference once at low confidence and write a reusable prediction cache."""
    model = YOLO(str(args.weights))
    started = time.perf_counter()
    images = []
    speed_totals = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    for start in range(0, len(image_paths), args.chunk_size):
        chunk = image_paths[start : start + args.chunk_size]
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=args.imgsz,
            conf=args.min_conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            batch=args.batch,
            device=args.device,
            stream=True,
            verbose=False,
        )
        for source_path, result in zip(chunk, results, strict=True):
            height, width = result.orig_shape
            detections = []
            if result.obb is not None:
                polygons = result.obb.xyxyxyxy.cpu().numpy()
                confidences = result.obb.conf.cpu().numpy()
                classes = result.obb.cls.cpu().numpy()
                for polygon, confidence, class_id in zip(polygons, confidences, classes):
                    detections.append(
                        {
                            "class_id": int(class_id),
                            "confidence": float(confidence),
                            "polygon": [[float(x), float(y)] for x, y in polygon],
                        }
                    )
            images.append(
                {
                    # ListSource rewrites result.path to image0/image1; preserve the converted dataset stem instead.
                    "image_id": source_path.stem,
                    "width": int(width),
                    "height": int(height),
                    "predictions": detections,
                }
            )
            for key in speed_totals:
                speed_totals[key] += float(result.speed.get(key, 0.0))
        del results
        gc.collect()
        torch.cuda.empty_cache()
        print(f"device {args.device}: cached {min(start + len(chunk), len(image_paths))}/{len(image_paths)} images")
    wall_seconds = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "matching": "same-class confidence-greedy one-to-one polygon IoU",
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "min_conf": args.min_conf,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "device": str(args.device),
        "image_count": len(images),
        "wall_seconds": wall_seconds,
        "speed_ms_per_image": {
            key: value / len(images) if images else 0.0 for key, value in speed_totals.items()
        },
        "class_names": model.names,
        "images": images,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def score_cache(
    payload: dict,
    label_dir: Path,
    iou_threshold: float,
    score_max_det: int | None = None,
    fixed_conf: float | None = None,
) -> dict:
    """Match one prediction cache and score a fixed or optimized confidence threshold."""
    images = []
    for image in payload["images"]:
        cached_predictions = image["predictions"][:score_max_det] if score_max_det else image["predictions"]
        predictions = [
            ObjectAnnotation(
                class_id=int(item["class_id"]),
                confidence=float(item["confidence"]),
                polygon=tuple((float(x), float(y)) for x, y in item["polygon"]),
            )
            for item in cached_predictions
        ]
        targets = load_yolo_ground_truth(
            label_dir / f"{image['image_id']}.txt", int(image["width"]), int(image["height"])
        )
        images.append((predictions, targets))
    records, total_gt_by_class = merge_matches(images, iou_threshold=iou_threshold)
    total_gt = sum(total_gt_by_class.values())
    names = {int(key): value for key, value in payload.get("class_names", {}).items()}
    if fixed_conf is None:
        selected = best_confidence(records, total_gt)
        class_thresholds, best_by_class = best_class_confidences(records, total_gt_by_class)
        optimization = {
            "threshold_mode": "optimized_on_evaluated_split",
            "best": score_to_dict(selected),
            "best_per_class": {
                "score": score_to_dict(best_by_class),
                "thresholds": {
                    names.get(class_id, str(class_id)): confidence
                    for class_id, confidence in class_thresholds.items()
                },
            },
        }
    else:
        selected = score_records(records, total_gt, fixed_conf)
        optimization = {
            "threshold_mode": "fixed",
            "fixed_conf": fixed_conf,
            "score": score_to_dict(selected),
        }
    per_class = class_scores(records, total_gt_by_class, selected.confidence)
    return {
        "metric": f"class-aware F1@polygon-IoU{iou_threshold:g}",
        "matching": "predictions sorted by confidence; same class; one GT matched at most once",
        **optimization,
        "per_class": {
            names.get(class_id, str(class_id)): score_to_dict(score) for class_id, score in per_class.items()
        },
        "total_ground_truths": sum(total_gt_by_class.values()),
        "prediction_cache": {
            key: payload[key]
            for key in ("weights", "imgsz", "min_conf", "nms_iou", "max_det", "image_count", "wall_seconds")
        },
        "score_max_det": score_max_det or payload["max_det"],
        "speed_ms_per_image": payload.get("speed_ms_per_image", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 OBB F1@IoU0.3 评估与置信度搜索")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="7", help="Single inference GPU")
    parser.add_argument("--min-conf", type=float, default=0.05, help="Prediction cache floor; 0.05 avoids dense-OBB NMS OOM")
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument(
        "--score-max-det",
        type=int,
        help="Score only the top N cached detections per image; requires a cache generated with max_det >= N",
    )
    parser.add_argument("--chunk-size", type=int, default=8, help="Images processed before releasing CUDA cache")
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--fixed-conf", type=float, help="Score this confidence without optimizing on the selected split")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N images for smoke testing")
    args = parser.parse_args()

    args.weights = args.weights.expanduser().resolve()
    args.data = (
        args.data.expanduser().resolve()
        if args.data
        else PROJECT_ROOT / "dataset_yolo" / f"fold_{args.fold}" / "data.yaml"
    )
    args.cache = args.cache.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if "," in str(args.device):
        raise ValueError("Competition inference uses one GPU; pass a single --device such as 7")
    if args.score_max_det is not None and args.score_max_det <= 0:
        raise ValueError("--score-max-det must be positive")
    if args.fixed_conf is not None and not 0.0 <= args.fixed_conf <= 1.0:
        raise ValueError("--fixed-conf must be between 0 and 1")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    image_paths, label_dir = resolve_split(args.data, args.split)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        image_paths = image_paths[: args.limit]
    if args.reuse_cache:
        payload = json.loads(args.cache.read_text(encoding="utf-8"))
    else:
        payload = generate_cache(args, image_paths, args.cache)
    if payload.get("image_count") != len(image_paths):
        raise ValueError("Prediction cache image count does not match the selected data split")
    expected_ids = [path.stem for path in image_paths]
    cached_ids = [image["image_id"] for image in payload.get("images", [])]
    if cached_ids != expected_ids:
        raise ValueError("Prediction cache image IDs do not match the selected data split")
    cached_weights = Path(payload.get("weights", "")).expanduser().resolve()
    if cached_weights != args.weights:
        raise ValueError(f"Prediction cache weights do not match --weights: {cached_weights}")
    if args.score_max_det is not None and args.score_max_det > int(payload["max_det"]):
        raise ValueError("--score-max-det cannot exceed the max_det used to generate the prediction cache")
    metrics = score_cache(payload, label_dir, args.match_iou, args.score_max_det, args.fixed_conf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
