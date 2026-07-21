#!/usr/bin/env python3
"""Shared experiment result recording for training and evaluation scripts."""

from __future__ import annotations

import csv
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULT_FIELDS = [
    "timestamp_utc",
    "stage",
    "run_name",
    "model",
    "fold",
    "split",
    "epochs",
    "imgsz",
    "batch",
    "weights",
    "precision",
    "recall",
    "f1",
    "map50",
    "map50_95",
    "fitness",
    "competition_precision",
    "competition_recall",
    "competition_f1_03",
    "competition_conf",
    "inference_ms",
    "params_m",
    "results_dir",
]


def metric_values(metrics: Any) -> dict[str, float | str]:
    """Normalize the metrics object returned by Ultralytics."""
    values = getattr(metrics, "results_dict", {}) or {}

    def pick(*keys: str) -> float | str:
        for key in keys:
            if key in values:
                return float(values[key])
        return ""

    precision = pick("metrics/precision(B)", "metrics/precision(M)")
    recall = pick("metrics/recall(B)", "metrics/recall(M)")
    f1 = 2 * precision * recall / (precision + recall) if precision != "" and recall != "" and precision + recall else 0.0
    speed = getattr(metrics, "speed", {}) or {}
    competition = getattr(metrics, "competition_score", None)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": pick("metrics/mAP50(B)", "metrics/mAP50(M)"),
        "map50_95": pick("metrics/mAP50-95(B)", "metrics/mAP50-95(M)"),
        "fitness": pick("fitness"),
        "competition_precision": float(competition.precision) if competition is not None else "",
        "competition_recall": float(competition.recall) if competition is not None else "",
        "competition_f1_03": float(competition.f1) if competition is not None else "",
        "competition_conf": float(competition.confidence) if competition is not None else "",
        "inference_ms": float(speed["inference"]) if "inference" in speed else "",
    }


def write_class_metrics(metrics: Any, csv_path: Path) -> None:
    """Write per-class OBB metrics returned by Ultralytics."""
    rows = getattr(metrics, "summary", lambda: [])()
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value.item() if hasattr(value, "item") else value for key, value in row.items()})


def append_result(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one row using a process lock so concurrent folds do not corrupt the table."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {field: row.get(field, "") for field in RESULT_FIELDS}
    normalized["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with csv_path.open("a+", newline="", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        reader = csv.DictReader(stream)
        existing_rows = list(reader)
        existing_fields = reader.fieldnames or []
        if existing_fields and existing_fields != RESULT_FIELDS:
            stream.seek(0)
            stream.truncate()
            migrated = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
            migrated.writeheader()
            for existing in existing_rows:
                migrated.writerow({field: existing.get(field, "") for field in RESULT_FIELDS})
        stream.seek(0, 2)
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(normalized)
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
