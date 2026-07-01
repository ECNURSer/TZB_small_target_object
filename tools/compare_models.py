#!/usr/bin/env python3
"""Compare YOLO26 n/s/m experiments from the unified result table."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return statistics.fmean(values) if values else float("nan")


def std(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="按 mAP50-95 汇总并比较 n/s/m")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "results" / "experiments.csv")
    parser.add_argument("--stage", default="test", choices=("test", "train_val"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "MODEL_COMPARISON.md")
    args = parser.parse_args()

    with args.input.expanduser().resolve().open(encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["stage"] == args.stage]
    if not rows:
        raise ValueError(f"结果表中没有 stage={args.stage} 的记录")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)
    ranked = sorted(groups.items(), key=lambda item: (mean(item[1], "map50_95"), mean(item[1], "map50")), reverse=True)

    lines = [
        "# YOLO26 OBB 模型对比",
        "",
        f"数据来源：`{args.input}`，阶段：`{args.stage}`。按平均 mAP50-95 降序排名。",
        "",
        "| 排名 | 模型 | 记录数 | mAP50-95 均值 | mAP50-95 标准差 | mAP50 均值 | F1 | 推理 ms/图 | 参数量 M |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (model, model_rows) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {model} | {len(model_rows)} | {mean(model_rows, 'map50_95'):.4f} "
            f"| {std(model_rows, 'map50_95'):.4f} | {mean(model_rows, 'map50'):.4f} "
            f"| {mean(model_rows, 'f1'):.4f} | {mean(model_rows, 'inference_ms'):.2f} "
            f"| {mean(model_rows, 'params_m'):.2f} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
