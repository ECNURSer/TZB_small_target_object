#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${CONDA_ENV:-yolo26-obb}"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.11 pip
fi

PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install --index-url "$PIP_MIRROR" --upgrade pip
if conda run -n "$ENV_NAME" python -c 'import torch, torchvision; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "检测到可用的 CUDA PyTorch，保留当前版本。"
else
    conda run --no-capture-output -n "$ENV_NAME" python -m pip install --timeout 300 --retries 20 --index-url "$PIP_MIRROR" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
fi
conda run --no-capture-output -n "$ENV_NAME" python -m pip install --timeout 300 --retries 20 --index-url "$PIP_MIRROR" -e "$PROJECT/ultralytics_src" tensorboard pytest pandas

echo "环境创建完成: $ENV_NAME"
conda run --no-capture-output -n "$ENV_NAME" python "$PROJECT/tools/check_env.py"
