#!/usr/bin/env python3
"""Render the unified experiment CSV as a Markdown table."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fmt(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return value or "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="将统一实验 CSV 生成为 Markdown 表")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "results" / "experiments.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "TRAINING_DIAGNOSTICS.md")
    args = parser.parse_args()
    csv_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"结果表不存在: {csv_path}")
    with csv_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    lines = [
        "# Ultralytics 训练诊断指标",
        "",
        "该文件由 `python tools/summarize_results.py` 生成，不代替比赛 F1 结果。",
        "",
        "| 时间(UTC) | 阶段 | 模型 | Fold | Split | mAP50 | mAP50-95 | Precision | Recall | Run |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['timestamp_utc']} | {row['stage']} | {row['model']} | {row['fold']} | {row['split']} "
            f"| {fmt(row['map50'])} | {fmt(row['map50_95'])} | {fmt(row['precision'])} "
            f"| {fmt(row['recall'])} | {row['run_name']} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
