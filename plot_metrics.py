#!/usr/bin/env python3
"""Plot key Ultralytics training metrics from one results.csv file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制单次训练曲线")
    parser.add_argument("results_csv", type=Path, help="例如 runs/yolo26n_obb_fold0/results.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.results_csv)
    data.columns = [column.strip() for column in data.columns]
    columns = [
        name
        for name in ("train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
        if name in data.columns
    ]
    if not columns:
        raise ValueError("results.csv 中没有可识别的训练指标")
    figure, axes = plt.subplots((len(columns) + 1) // 2, 2, figsize=(12, 3.5 * ((len(columns) + 1) // 2)))
    axes = axes.reshape(-1)
    for axis, column in zip(axes, columns):
        axis.plot(data["epoch"], data[column])
        axis.set(title=column, xlabel="epoch")
        axis.grid(alpha=0.25)
    for axis in axes[len(columns) :]:
        axis.set_visible(False)
    figure.tight_layout()
    output = args.output or args.results_csv.with_name("metrics_summary.png")
    figure.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()

