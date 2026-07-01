#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${CONDA_ENV:-yolo26-obb}"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.11 pip
fi

PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_WHEEL="https://pypi.tuna.tsinghua.edu.cn/packages/ae/30/a3a2120621bf9c17779b169fc17e3dc29b230c29d0f8222f499f5e159aa8/torch-2.10.0-cp311-cp311-manylinux_2_28_x86_64.whl"
TORCHVISION_WHEEL="https://pypi.tuna.tsinghua.edu.cn/packages/43/ae/ad5d6165797de234c9658752acb4fce65b78a6a18d82efdf8367c940d8da/torchvision-0.25.0-cp311-cp311-manylinux_2_28_x86_64.whl"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install --index-url "$PIP_MIRROR" --upgrade pip
if conda run -n "$ENV_NAME" python -c 'import torch, torchvision; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "检测到可用的 CUDA PyTorch，保留当前版本。"
else
    conda run --no-capture-output -n "$ENV_NAME" python -m pip install --timeout 300 --retries 20 --index-url "$PIP_MIRROR" "$TORCH_WHEEL" "$TORCHVISION_WHEEL"
fi
conda run --no-capture-output -n "$ENV_NAME" python -m pip install --timeout 300 --retries 20 --index-url "$PIP_MIRROR" -e "$PROJECT/ultralytics_src" tensorboard pytest pandas

echo "环境创建完成: $ENV_NAME"
conda run --no-capture-output -n "$ENV_NAME" python "$PROJECT/tools/check_env.py"
