#!/usr/bin/env bash
# 一键创建 snake_rl 环境并锁定 ROCm-6.4 nightly 版 PyTorch
set -e

ENV_NAME=snake_rl
YAML_FILE=environment.yml          # conda 部分
REQ_FILE=requirements_rest.txt     # 剩余 PyPI 包
ROCM_INDEX=https://download.pytorch.org/whl/nightly/rocm6.4

echo ">>> 1. 创建 conda 环境（不含 PyTorch）"
conda env create -f "$YAML_FILE" -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ">>> 2. 安装 torch/torchvision（仅 ROCm-6.4 索引）"
pip install --pre torch torchvision \
        --index-url "$ROCM_INDEX" \
        --no-index --no-deps

echo ">>> 3. 安装其余 PyPI 依赖"
pip install -r "$REQ_FILE"

echo ">>> 4. 验证"
python - <<'PY'
import torch, torchvision, numpy, gymnasium, cv2
print("PyTorch:", torch.__version__)
print("CUDA/ROCm available:", torch.cuda.is_available())
PY

echo ">>> 完成！激活环境：conda activate $ENV_NAME"