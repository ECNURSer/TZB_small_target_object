#!/usr/bin/env python3
"""Run LSKNet-T OBB inference with the project default 1300-epoch checkpoint."""

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


DEFAULT_WEIGHTS = PROJECT_ROOT / "runs/lsknet_t_obb_full_fold0_img1280/weights/best.pt"
DEFAULT_CONF = 0.30566


def main() -> None:
    parser = argparse.ArgumentParser(description="LSKNet-T OBB 独立推理")
    parser.add_argument("--source", required=True, help="图像、目录、视频或 glob")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=0.7, help="OBB NMS IoU；LSKNet 使用常规 NMS，此参数有效")
    parser.add_argument("--max-det", type=int, default=600)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="lsknet_t_obb_full_fold0_img1280_predict")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "predict")
    parser.add_argument("--nosave", action="store_true", help="不保存可视化图片/视频")
    parser.add_argument("--no-txt", action="store_true", help="不保存 YOLO OBB txt 结果")
    args = parser.parse_args()

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在: {weights}")

    YOLO(str(weights)).predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        project=str(args.project.expanduser().resolve()),
        name=args.name,
        save=not args.nosave,
        save_txt=not args.no_txt,
        save_conf=True,
    )


if __name__ == "__main__":
    main()
