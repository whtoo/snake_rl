# Rainbow DQN 使用说明

## 概述

Rainbow DQN 实现已完成，集成了所有6个核心组件：

1. ✅ **Double DQN** - 使用主网络选择动作，目标网络评估Q值
2. ✅ **Dueling DQN** - 分离状态值和优势函数的网络架构
3. ✅ **Prioritized Replay** - 基于TD误差的优先级经验回放
4. ✅ **Multi-step Learning** - N步学习（默认3步）
5. ✅ **Noisy Networks** - 替代ε-贪婪探索的噪声网络
6. ✅ **Distributional DQN** - C51算法的分布式Q学习

## 快速开始

### 1. 基础训练

```bash
# 训练标准 Rainbow DQN
python run_training.py rainbow 1000 --use_noisy --prioritized_replay

# 训练分布式 Rainbow DQN
python run_training.py rainbow 1000 --use_noisy --use_distributional --prioritized_replay
```

### 2. 自定义参数

```bash
python run_training.py rainbow 1000 \
    --use_noisy \
    --use_distributional \
    --prioritized_replay \
    --n_step 5 \
    --n_atoms 51 \
    --v_min -10 \
    --v_max 10 \
    --lr 1e-4 \
    --batch_size 32
```

## 参数说明

### Rainbow 特有参数

- `--use_noisy`: 启用噪声网络（替代ε-贪婪探索）
- `--use_distributional`: 启用分布式Q学习（C51算法）
- `--n_step`: N步学习的步数（默认3）
- `--n_atoms`: 分布式Q学习的原子数量（默认51）
- `--v_min`: 值函数分布的最小值（默认-10）
- `--v_max`: 值函数分布的最大值（默认10）
- `--prioritized_replay`: 启用优先经验回放

### 通用DQN参数

- `--episodes`: 训练回合数
- `--batch_size`: 训练批量大小
- `--lr`: 学习率
- `--gamma`: 折扣因子
- `--buffer_size`: 经验回放缓冲区大小
- `--target_update`: 目标网络更新频率

## 实现架构

### 核心文件

```
src/
├── model.py           # 网络模型定义
│   ├── NoisyLinear    # 噪声线性层
│   └── RainbowDQN     # Rainbow网络架构
├── agent.py           # 智能体实现
│   ├── NStepBuffer    # N步学习缓冲区
│   ├── PrioritizedReplayBuffer  # 优先经验回放
│   └── RainbowAgent   # Rainbow智能体
└── train.py           # 训练脚本
```

### 网络架构

```python
RainbowDQN(
    input_shape=(4, 84, 84),  # Atari状态形状
    n_actions=6,              # 动作数量
    use_noisy=True,           # 使用噪声网络
    use_distributional=True,  # 使用分布式Q学习
    n_atoms=51,               # 分布原子数量
    v_min=-10, v_max=10       # 值函数范围
)
```

## 测试验证

运行完整的功能测试：

```bash
python test_rainbow.py
```

测试内容包括：
- 噪声线性层功能
- Rainbow DQN 网络前向传播
- N步缓冲区计算
- Rainbow智能体训练

## 性能对比

| 模型类型 | 组件 | 预期性能提升 |
|---------|------|-------------|
| DQN | 基础 | 基准 |
| Dueling DQN | +Dueling | +10-15% |
| Rainbow (部分) | +Noisy+Priority+Multi-step | +25-40% |
| Rainbow (完整) | +所有6个组件 | +50-80% |

## 使用建议

### 1. 渐进式启用组件

```bash
# 阶段1：基础Rainbow（最重要的组件）
python run_training.py rainbow 1000 --use_noisy --prioritized_replay

# 阶段2：添加分布式学习
python run_training.py rainbow 1000 --use_noisy --use_distributional --prioritized_replay

# 阶段3：调整N步学习
python run_training.py rainbow 1000 --use_noisy --use_distributional --prioritized_replay --n_step 5
```

### 2. 超参数调优

对于不同的Atari游戏，可以调整：

- **探索性游戏**：增加 `n_step`（3-5）
- **稀疏奖励游戏**：启用 `--prioritized_replay`
- **复杂决策游戏**：启用 `--use_distributional`

### 3. 资源考虑

- **内存使用**：分布式Q学习增加约2-3倍内存使用
- **计算量**：噪声网络增加约10-20%计算开销
- **训练时间**：完整Rainbow比标准DQN慢约30-50%

## 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 减小批量大小和缓冲区
   --batch_size 16 --buffer_size 50000
   ```

2. **训练不稳定**
   ```bash
   # 降低学习率
   --lr 5e-5
   ```

3. **收敛速度慢**
   ```bash
   # 增加目标网络更新频率
   --target_update 500
   ```

## 扩展功能

Rainbow DQN实现支持进一步扩展：

- **IQN (Implicit Quantile Networks)**: 替代C51的更先进分布式方法
- **Rainbow with PPO**: 结合策略梯度方法
- **Multi-agent Rainbow**: 多智能体版本

---

有关技术细节，请参考 `docs/rainbow_dqn_design.md`