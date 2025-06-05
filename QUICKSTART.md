# 快速开始指南

本指南将帮助您快速设置和运行强化学习 Atari 游戏项目。

## 1. 环境设置

### 创建 Conda 环境

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate rl_atari
```

### 验证安装

```bash
# 运行测试脚本验证环境
python test_setup.py
```

如果所有测试通过，说明环境配置正确！

## 2. 快速训练

### 方法一：使用启动脚本（推荐）

```bash
# 使用默认参数训练 DQN
./run_training.sh

# 训练 Dueling DQN，1500 回合
./run_training.sh dueling 1500
```

### 方法二：直接运行 Python 脚本

```bash
# 基础训练
python src/train.py

# 使用优先经验回放训练
python src/train.py --prioritized_replay --episodes 2000
```

## 3. 监控训练

### 使用 TensorBoard

```bash
# 在新终端中启动 TensorBoard
tensorboard --logdir=logs
```

然后在浏览器中打开 `http://localhost:6006` 查看训练曲线。

### 查看训练日志

训练过程中会在终端显示实时信息：
- 当前回合奖励
- 平均奖励（最近100回合）
- 训练损失
- 探索率变化

## 4. 评估模型

### 基础评估

```bash
# 评估最佳模型
python src/evaluate.py --model_path checkpoints/best_model_dqn.pth
```

### 录制游戏视频

```bash
# 评估并录制视频
python src/evaluate.py --model_path checkpoints/best_model_dqn.pth --record_video
```

### 实时观看

```bash
# 实时观看智能体游戏（需要图形界面）
python src/evaluate.py --model_path checkpoints/best_model_dqn.pth --render
```

## 5. 交互式实验

### 启动 Jupyter Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

在 Notebook 中可以：
- 查看环境和数据预处理
- 理解模型架构
- 运行小规模训练演示
- 可视化训练结果

## 6. 常见问题

### Q: 训练很慢怎么办？
A: 
- 确保使用 GPU：检查 `torch.cuda.is_available()` 返回 `True`
- 减少 `buffer_size` 或 `batch_size`
- 使用更少的训练回合进行测试

### Q: 内存不足怎么办？
A:
- 减少 `buffer_size`（如设为 50000）
- 减少 `batch_size`（如设为 16）
- 关闭其他占用内存的程序

### Q: 如何调整超参数？
A:
- 学习率：通常在 1e-5 到 1e-3 之间
- 探索率衰减：根据训练回合数调整
- 目标网络更新频率：通常在 1000-10000 之间

### Q: 训练效果不好怎么办？
A:
- 增加训练回合数
- 尝试不同的模型架构（DQN vs Dueling DQN）
- 使用优先经验回放
- 调整奖励函数或环境设置

## 7. 项目结构说明

```
.
├── README.md              # 详细项目说明
├── QUICKSTART.md          # 本快速开始指南
├── environment.yml        # Conda 环境配置
├── test_setup.py          # 环境测试脚本
├── run_training.sh        # 训练启动脚本
├── src/                   # 源代码
│   ├── model.py           # 神经网络模型
│   ├── agent.py           # DQN 智能体
│   ├── utils.py           # 工具函数
│   ├── train.py           # 训练脚本
│   └── evaluate.py        # 评估脚本
├── notebooks/             # Jupyter 笔记本
│   └── demo.ipynb         # 演示笔记本
├── checkpoints/           # 模型保存目录（训练后生成）
├── logs/                  # 训练日志目录（训练后生成）
└── videos/                # 视频保存目录（录制后生成）
```

## 8. 下一步

- 尝试不同的 Atari 游戏环境
- 实现更高级的算法（如 Rainbow DQN）
- 调优超参数以获得更好的性能
- 添加自定义的奖励函数
- 实现多智能体训练

祝您训练愉快！🎮🤖