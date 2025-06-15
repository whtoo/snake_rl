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

### 验证 Rainbow DQN 功能

```bash
# 测试 Rainbow DQN 各组件功能
python test_rainbow.py
```

这将验证：
- 分布式 Q 学习网络
- 噪声网络层
- 多步学习缓冲区
- 优先经验回放
- 所有组件的集成

## 2. 快速训练

### 方法一：使用启动脚本（推荐）

```bash
# 使用默认参数训练 DQN
./run_training.sh

# 训练 Dueling DQN，1500 回合
./run_training.sh dueling 1500

# 训练 Rainbow DQN（推荐）
./run_training.sh rainbow 2000
```

### 方法二：直接运行 Python 脚本

```bash
# 基础 DQN 训练
python src/train.py

# 使用优先经验回放训练
python src/train.py --prioritized_replay --episodes 2000

# 训练 Rainbow DQN（推荐配置）
python src/train.py --model rainbow --episodes 2000 --use_noisy --prioritized_replay

# 训练完整 Rainbow DQN（所有高级功能）
python src/train.py --model rainbow --episodes 2000 --use_noisy --use_distributional --prioritized_replay --n_step 3

# 训练带多步学习的 DQN
python src/train.py --n_step 3 --episodes 1500

# 训练带噪声网络的 DQN
python src/train.py --model noisy --episodes 1500
```

### Rainbow DQN 参数说明
Rainbow DQN 支持以下可选参数：
- `--use_noisy`: 使用噪声网络（默认：True）
- `--use_distributional`: 使用分布式DQN（默认：False）
- `--prioritized_replay`: 使用优先经验回放（默认：True）
- `--n_step`: 多步学习步数（默认：3）
- `--n_atoms`: 原子数（分布式DQN，默认：51）
- `--v_min`, `--v_max`: 价值分布范围（分布式DQN，默认：-10,10）

完整参数列表见 `docs/rainbow_usage.md`

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
# 评估 DQN 模型
python src/evaluate.py --model_path checkpoints/best_model_dqn.pth

# 评估 Rainbow DQN 模型
python src/evaluate.py --model_path checkpoints/best_model_rainbow.pth

# 评估 Dueling DQN 模型
python src/evaluate.py --model_path checkpoints/best_model_dueling.pth
```

### 录制游戏视频

```bash
# 评估并录制视频
python src/evaluate.py --model_path checkpoints/best_model_rainbow.pth --record_video

# 录制多个回合的视频
python src/evaluate.py --model_path checkpoints/best_model_rainbow.pth --record_video --episodes 5
```

### 实时观看

```bash
# 实时观看智能体游戏（需要图形界面）
python src/evaluate.py --model_path checkpoints/best_model_rainbow.pth --render

# 连续观看多个回合
python src/evaluate.py --model_path checkpoints/best_model_rainbow.pth --render --episodes 3
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
- 尝试不同的模型架构（DQN vs Dueling DQN vs Rainbow DQN）
- 使用优先经验回放
- 启用多步学习（--n_step 3）
- 使用 Rainbow DQN（集成所有高级技术）
- 调整奖励函数或环境设置

### Q: 如何选择最适合的模型？
A:
- **初学者**：从基础 DQN 开始
- **中级用户**：使用 Dueling DQN 或带优先经验回放的 DQN
- **高级用户**：使用 Rainbow DQN（性能最佳，但训练时间较长）
- **资源有限**：使用 DQN + 优先经验回放

### Q: Rainbow DQN 和普通 DQN 有什么区别？
A:
- Rainbow DQN 集成了 6 种先进技术：双 DQN、优先经验回放、Dueling 网络、多步学习、分布式 Q 学习、噪声网络
- 训练效果更好，但计算资源需求更高
- 推荐用于最终的高性能模型训练

## 7. 项目结构说明

```
.
├── README.md              # 详细项目说明
├── QUICKSTART.md          # 本快速开始指南
├── environment.yml        # Conda 环境配置
├── test_setup.py          # 环境测试脚本
├── test_rainbow.py        # Rainbow DQN 功能测试
├── run_training.sh        # 训练启动脚本（Linux/Mac）
├── run_training.bat       # 训练启动脚本（Windows）
├── run_training.py        # Python 训练启动器
├── src/                   # 源代码
│   ├── model.py           # 神经网络模型（DQN, Dueling DQN, Rainbow DQN）
│   ├── agent.py           # 强化学习智能体（多种算法实现）
│   ├── utils.py           # 工具函数
│   ├── train.py           # 训练脚本
│   └── evaluate.py        # 评估脚本
├── notebooks/             # Jupyter 笔记本
│   └── demo.ipynb         # 演示笔记本
├── tests/                 # 测试文件
│   └── test_rainbow_components.py  # Rainbow 组件测试
├── docs/                  # 技术文档
│   ├── rainbow_dqn_design.md      # Rainbow DQN 设计文档
│   └── rainbow_usage.md           # Rainbow DQN 使用指南
├── checkpoints/           # 模型保存目录（训练后生成）
├── logs/                  # 训练日志目录（训练后生成）
└── videos/                # 视频保存目录（录制后生成）
```

## 8. 高级功能和下一步

### 🚀 已实现的高级功能
- **Rainbow DQN**：集成了所有先进技术的最强算法
- **多步学习**：使用 `--n_step` 参数提升学习效率
- **噪声网络**：替代 ε-贪婪策略的探索方法
- **分布式 Q 学习**：学习价值分布而非期望值
- **优先经验回放**：重要经验优先学习

### 📚 学习建议
1. **初学者路径**：DQN → Dueling DQN → Rainbow DQN
2. **阅读技术文档**：查看 `docs/` 目录了解算法细节
3. **运行测试**：使用 `python test_rainbow.py` 验证功能
4. **实验对比**：训练不同模型并比较性能

### 🎯 进阶探索
- 尝试不同的 Atari 游戏环境
- 调优超参数以获得更好的性能
- 添加自定义的奖励函数
- 实现分布式训练支持
- 探索模型压缩和加速技术

### 📖 推荐阅读
- `docs/rainbow_dqn_design.md` - Rainbow DQN 算法设计
- `docs/rainbow_usage.md` - 详细使用指南
- `notebooks/demo.ipynb` - 交互式演示

祝您训练愉快！🎮🤖✨