#!/bin/bash

# 强化学习训练启动脚本
# 使用方法: ./run_training.sh [模型类型] [回合数]

set -e  # 遇到错误时退出

# 默认参数
MODEL_TYPE=${1:-"dqn"}  # 默认使用 DQN
EPISODES=${2:-1000}     # 默认训练 1000 回合

echo "======================================"
echo "强化学习 Atari 游戏训练"
echo "======================================"
echo "模型类型: $MODEL_TYPE"
echo "训练回合: $EPISODES"
echo "游戏环境: ALE/Assault-v5"
echo "======================================"

# 检查 conda 环境
if ! conda info --envs | grep -q "rl_atari"; then
    echo "错误: 未找到 rl_atari conda 环境"
    echo "请先运行: conda env create -f environment.yml"
    exit 1
fi

# 激活环境
echo "激活 conda 环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rl_atari

# 创建必要的目录
echo "创建输出目录..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p videos

# 开始训练
echo "开始训练..."
python src/train.py \
    --model $MODEL_TYPE \
    --episodes $EPISODES \
    --save_dir checkpoints \
    --log_dir logs \
    --save_interval 100 \
    --eval_interval 50

echo "======================================"
echo "训练完成！"
echo "模型保存在: checkpoints/"
echo "日志保存在: logs/"
echo "======================================"
echo "使用以下命令评估模型:"
echo "python src/evaluate.py --model_path checkpoints/best_model_${MODEL_TYPE}.pth"
echo "======================================"