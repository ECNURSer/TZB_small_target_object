# 服务器环境记录

采集时间：2026-06-30，时区 Asia/Shanghai。

## 操作系统与计算资源

| 项目 | 实测值 |
|---|---|
| 操作系统 | Ubuntu 22.04.4 LTS |
| CPU | Intel Xeon Processor (Skylake, IBRS) |
| CPU 拓扑 | 2 socket × 20 core × 2 thread，共 80 逻辑 CPU |
| 内存 | 629 GiB，总可用约 377 GiB（采集时） |
| 系统盘 | 492 GiB，总剩余约 426 GiB（采集时） |
| GPU | 8 × NVIDIA A100-PCIE-40GB |
| 单卡显存 | 40960 MiB |
| NVIDIA 驱动 | 570.133.20 |
| 驱动支持的最高 CUDA | 12.8（`nvidia-smi` 显示值） |
| CUDA Toolkit / nvcc | 未安装，不影响 PyTorch wheel 使用自带 CUDA runtime |
| Conda | Miniconda，conda 26.3.2 |
| Conda 环境 | `yolo26-obb`，Python 3.11.15 |
| PyTorch | 2.11.0+cu126 |
| TorchVision | 0.26.0+cu126 |
| PyTorch CUDA runtime | 12.6 |
| Ultralytics | 8.4.80，本项目 `ultralytics_src/` |

## 运行说明

新的 m 全量单模型配置使用 8 张 GPU：`device=0,1,2,3,4,5,6,7`、`imgsz=1280`、全局 `batch=96`（每卡 12）。运行前必须先检查 `nvidia-smi`，确认八张卡全部空闲。历史基线的全局 batch 为 n=64、s=64、m=72，使用 `imgsz=1024`。

PyTorch wheel 使用 CUDA 12.6 runtime，当前驱动满足要求，不依赖系统 `nvcc`。环境可通过 `python tools/check_env.py` 和 `python -m pip check` 重新验证。
