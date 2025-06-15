# 强化学习 Atari 游戏 Demo

本项目使用 OpenAI Gym 和 PyTorch 实现了一个强化学习 Demo，用于玩 Atari 游戏 Assault-v5。

## 项目结构

```
.
├── README.md                 # 项目说明文档
├── environment.yml          # Conda 环境配置文件
├── src/                     # 源代码目录
│   ├── agent.py             # 强化学习智能体
│   ├── model.py             # 神经网络模型
│   ├── train.py             # 训练脚本
│   ├── evaluate.py          # 评估脚本
│   └── utils.py             # 工具函数
└── notebooks/               # Jupyter notebooks 用于实验和可视化
    └── demo.ipynb           # 演示笔记本
```

## 实现计划

1. [x] 创建项目结构和 README
2. [x] 创建 Conda 环境配置文件
3. [x] 实现神经网络模型 (DQN)
4. [x] 实现强化学习智能体
5. [x] 实现训练脚本
6. [x] 实现评估脚本
7. [x] 创建演示笔记本
8. [x] 测试和优化

## 环境配置

使用以下命令创建并激活 Conda 环境：

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate rl_atari
```

## 算法选择

本项目将使用深度 Q 网络 (DQN) 算法，这是一种结合了 Q-learning 和深度神经网络的强化学习算法，特别适合处理具有高维状态空间的问题，如 Atari 游戏。

## 使用方法

### 训练智能体

```bash
# 基础训练（使用默认参数）
python src/train.py

# 自定义训练参数
python src/train.py --episodes 2000 --lr 5e-4 --batch_size 64 --prioritized_replay

# 使用 Dueling DQN
python src/train.py --model dueling --episodes 1500
```

### 评估智能体

```bash
# 评估训练好的模型
python src/evaluate.py --model_path checkpoints/best_model_dqn.pth

# 评估并录制视频
python src/evaluate.py --model_path checkpoints/best_model_dqn.pth --record_video

# 实时观看智能体游戏
python src/evaluate.py --model_path checkpoints/best_model_dqn.pth --render
```

### 使用 Jupyter Notebook

```bash
# 启动 Jupyter
jupyter notebook notebooks/demo.ipynb
```

## 已实现功能

- [x] 基础 DQN 算法
- [x] 经验回放 (Experience Replay)
- [x] 目标网络 (Target Network)
- [x] 双 DQN (Double DQN)
- [x] 优先经验回放 (Prioritized Experience Replay)
- [x] Dueling DQN 架构
- [x] 图像预处理和帧堆叠
- [x] TensorBoard 日志记录
- [x] 模型保存和加载
- [x] 视频录制功能
- [x] 交互式演示笔记本
- [x] 多步学习 (N-step Learning)
- [x] Rainbow DQN (完整实现)
- [x] 噪声网络 (Noisy Networks)
- [x] 分布式Q学习 (Distributional DQN/C51)
- [x] 超参数优化支持

## 待办事项

- [ ] 添加分布式训练支持
- [ ] 实现模型压缩和加速
- [ ] 添加更多游戏环境支持

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 1000 | 训练回合数 |
| `--buffer_size` | 100000 | 经验回放缓冲区大小 |
| `--batch_size` | 32 | 训练批量大小 |
| `--gamma` | 0.99 | 折扣因子 |
| `--lr` | 1e-4 | 学习率 |
| `--epsilon_start` | 1.0 | 初始探索率 |
| `--epsilon_final` | 0.01 | 最终探索率 |
| `--epsilon_decay` | 100000 | 探索率衰减帧数 |
| `--target_update` | 1000 | 目标网络更新频率 |
| `--prioritized_replay` | False | 是否使用优先经验回放 |

## 性能优化建议

1. **GPU 加速**: 确保安装了 CUDA 版本的 PyTorch
2. **内存管理**: 根据可用内存调整 `buffer_size`
3. **并行处理**: 可以同时训练多个智能体
4. **超参数调优**: 使用网格搜索或贝叶斯优化
5. **早停策略**: 监控验证性能避免过拟合