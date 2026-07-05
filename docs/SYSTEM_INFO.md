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

## 当前限制

2026-06-30 复核时，0-3 号卡正在被其他任务使用，4-7 号卡空闲。因此默认训练配置暂用 `device=4,5,6,7`，三种模型统一使用全局 `batch=32`。GPU 状态会变化，每次训练前必须先运行 `nvidia-smi`；若可用卡号变化，通过 `--device` 覆盖配置。

当前 PyTorch wheel 使用 CUDA 12.6 runtime，驱动版本满足要求；不依赖系统 `nvcc`。`python tools/check_env.py` 与 `pip check` 均已通过。外部正式数据已经接入。YOLO26s 已使用全局 batch 64、imgsz 1024 完成 500 epoch，峰值显存约 30.8GB；下一轮 YOLO26n 同样使用 batch 64，并启用训练器 NaN/Inf 与 EMA 回滚保护。
