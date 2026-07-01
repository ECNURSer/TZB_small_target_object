#!/usr/bin/env python3
"""Run standalone YOLO26 OBB inference on images, directories, or videos."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_CONFIG_ROOT = PROJECT_ROOT / ".ultralytics"
YOLO_CONFIG_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics_src"))

from ultralytics import YOLO  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26 OBB 独立推理")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="predict")
    args = parser.parse_args()

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在: {weights}")
    YOLO(str(weights)).predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(PROJECT_ROOT / "runs" / "predict"),
        name=args.name,
        save=True,
        save_txt=True,
        save_conf=True,
    )


if __name__ == "__main__":
    main()
