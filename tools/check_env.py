#!/usr/bin/env python3
"""Check that the environment can use CUDA and imports the pinned local source."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torchvision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
config_root = PROJECT_ROOT / ".ultralytics"
config_root.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics_src"))

import ultralytics  # noqa: E402


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"ultralytics={ultralytics.__version__} ({ultralytics.__file__})")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"gpu_count={torch.cuda.device_count()}")
    if ultralytics.__version__ != "8.4.80":
        raise RuntimeError("未导入项目固定的 Ultralytics 8.4.80")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 PyTorch CUDA runtime 与 NVIDIA 驱动兼容性")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"gpu{index}={props.name}, {props.total_memory / 1024**3:.1f} GiB")


if __name__ == "__main__":
    main()
