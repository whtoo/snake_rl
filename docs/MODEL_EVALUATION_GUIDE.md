# 🎮 训练模型评估使用指南

## 📋 概述

本指南详细介绍如何使用训练好的DQN/Rainbow模型进行游戏评估，包括基础评估、可视化演示和高级分析功能。

## 🚀 快速开始

### 基础评估命令

```bash
# 评估DQN模型（10回合）
python src/evaluate.py --model_path models/best_dqn_model.pth --n_episodes 10

# 评估Rainbow模型
python src/evaluate.py --model_path models/best_rainbow_model.pth --model rainbow --n_episodes 10

# 带可视化的评估
python src/evaluate.py --model_path models/best_dqn_model.pth --render --n_episodes 5

# 录制游戏视频
python src/evaluate.py --model_path models/best_dqn_model.pth --record_video --video_path videos/
```

## 📊 详细参数说明

### 必需参数
- `--model_path`: 训练好的模型文件路径（.pth格式）

### 可选参数
- `--env`: 游戏环境名称（默认："ALE/Assault-v5"）
- `--model`: 模型类型，可选 "dqn", "dueling", "rainbow"（默认："dqn"）
- `--n_episodes`: 评估回合数（默认：10）
- `--max_steps`: 每回合最大步数（默认：10000）
- `--render`: 是否显示游戏画面
- `--record_video`: 是否录制游戏视频
- `--video_path`: 视频保存目录（默认："videos"）
- `--seed`: 随机种子（默认：42）

## 🎯 使用场景示例

### 1. 模型性能评估

```bash
# 全面评估模型性能（100回合）
python src/evaluate.py \
    --model_path models/best_rainbow_model.pth \
    --model rainbow \
    --n_episodes 100 \
    --seed 42
```

**输出示例：**
```
==================================================
EVALUATION RESULTS
==================================================
Environment: ALE/Assault-v5
Model: rainbow
Episodes: 100
Mean Reward: 1247.35 ± 234.67
Mean Episode Length: 2341.23 ± 456.78
Best Episode Reward: 1856.00
Worst Episode Reward: 678.00
==================================================
```

### 2. 可视化演示

```bash
# 实时观看智能体游戏（适合演示）
python src/evaluate.py \
    --model_path models/best_dqn_model.pth \
    --render \
    --n_episodes 3 \
    --max_steps 5000
```

### 3. 视频录制

```bash
# 录制高质量游戏视频
python src/evaluate.py \
    --model_path models/best_rainbow_model.pth \
    --model rainbow \
    --record_video \
    --video_path videos/rainbow_demo \
    --n_episodes 5
```

### 4. 不同环境测试

```bash
# 在不同游戏环境中测试模型泛化能力
python src/evaluate.py \
    --model_path models/best_dqn_model.pth \
    --env "ALE/Breakout-v5" \
    --n_episodes 20
```

## 🔧 高级评估功能

### 创建批量评估脚本

```python
# batch_evaluate.py
import subprocess
import os
import json
from datetime import datetime

def batch_evaluate(model_configs, output_dir="evaluation_results"):
    """
    批量评估多个模型
    
    参数:
        model_configs: 模型配置列表
        output_dir: 结果保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for config in model_configs:
        print(f"评估模型: {config['name']}")
        
        # 构建评估命令
        cmd = [
            "python", "src/evaluate.py",
            "--model_path", config["model_path"],
            "--model", config["model_type"],
            "--n_episodes", str(config["episodes"]),
            "--seed", str(config["seed"])
        ]
        
        # 执行评估
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 解析结果
        if result.returncode == 0:
            # 这里可以解析输出获取具体指标
            results.append({
                "model_name": config["name"],
                "model_path": config["model_path"],
                "evaluation_time": datetime.now().isoformat(),
                "stdout": result.stdout,
                "success": True
            })
        else:
            results.append({
                "model_name": config["name"],
                "error": result.stderr,
                "success": False
            })
    
    # 保存结果
    with open(os.path.join(output_dir, "batch_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

# 使用示例
if __name__ == "__main__":
    configs = [
        {
            "name": "DQN_v1",
            "model_path": "models/dqn_episode_1000.pth",
            "model_type": "dqn",
            "episodes": 50,
            "seed": 42
        },
        {
            "name": "Rainbow_v1",
            "model_path": "models/rainbow_episode_1000.pth",
            "model_type": "rainbow",
            "episodes": 50,
            "seed": 42
        }
    ]
    
    results = batch_evaluate(configs)
    print(f"批量评估完成，共评估 {len(configs)} 个模型")
```

### 创建性能对比分析脚本

```python
# performance_analysis.py
import matplotlib.pyplot as plt
import numpy as np
import json
from typing import List, Dict

def analyze_model_performance(results_file: str):
    """
    分析模型性能并生成对比图表
    
    参数:
        results_file: 批量评估结果文件
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # 提取性能指标（需要根据实际输出格式调整）
    model_names = []
    mean_rewards = []
    std_rewards = []
    
    for result in results:
        if result['success']:
            # 解析stdout获取指标
            stdout = result['stdout']
            # 这里需要实现具体的解析逻辑
            # 示例：从输出中提取Mean Reward
            lines = stdout.split('\n')
            for line in lines:
                if 'Mean Reward:' in line:
                    # 解析 "Mean Reward: 1247.35 ± 234.67"
                    parts = line.split(':')
                    if len(parts) > 1:
                        reward_part = parts[1].strip()
                        mean_val = float(reward_part.split('±')[0].strip())
                        std_val = float(reward_part.split('±')[1].strip())
                        
                        model_names.append(result['model_name'])
                        mean_rewards.append(mean_val)
                        std_rewards.append(std_val)
                        break
    
    # 创建对比图表
    plt.figure(figsize=(12, 8))
    
    # 子图1：平均奖励对比
    plt.subplot(2, 2, 1)
    bars = plt.bar(model_names, mean_rewards, yerr=std_rewards, capsize=5)
    plt.title('模型平均奖励对比')
    plt.ylabel('平均奖励')
    plt.xticks(rotation=45)
    
    # 为每个柱子添加数值标签
    for bar, mean_val in zip(bars, mean_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{mean_val:.1f}', ha='center', va='bottom')
    
    # 子图2：稳定性对比（标准差）
    plt.subplot(2, 2, 2)
    plt.bar(model_names, std_rewards, color='orange')
    plt.title('模型稳定性对比（标准差越小越稳定）')
    plt.ylabel('奖励标准差')
    plt.xticks(rotation=45)
    
    # 子图3：效率指标（奖励/标准差比率）
    plt.subplot(2, 2, 3)
    efficiency = [m/s if s > 0 else 0 for m, s in zip(mean_rewards, std_rewards)]
    plt.bar(model_names, efficiency, color='green')
    plt.title('模型效率指标（奖励/标准差）')
    plt.ylabel('效率比率')
    plt.xticks(rotation=45)
    
    # 子图4：综合评分
    plt.subplot(2, 2, 4)
    # 归一化分数：(奖励 - 最小奖励) / (最大奖励 - 最小奖励)
    if len(mean_rewards) > 1:
        min_reward = min(mean_rewards)
        max_reward = max(mean_rewards)
        normalized_scores = [(r - min_reward) / (max_reward - min_reward) 
                           for r in mean_rewards]
    else:
        normalized_scores = [1.0] * len(mean_rewards)
    
    colors = plt.cm.viridis(normalized_scores)
    bars = plt.bar(model_names, normalized_scores, color=colors)
    plt.title('模型综合评分（归一化）')
    plt.ylabel('归一化分数')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 生成性能报告
    generate_performance_report(model_names, mean_rewards, std_rewards)

def generate_performance_report(names, means, stds):
    """
    生成详细的性能报告
    """
    report = "# 模型性能评估报告\n\n"
    report += f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # 排序模型（按平均奖励）
    sorted_data = sorted(zip(names, means, stds), key=lambda x: x[1], reverse=True)
    
    report += "## 模型排名\n\n"
    for i, (name, mean, std) in enumerate(sorted_data, 1):
        report += f"{i}. **{name}**: {mean:.2f} ± {std:.2f}\n"
    
    report += "\n## 详细分析\n\n"
    
    best_model = sorted_data[0]
    report += f"### 最佳模型: {best_model[0]}\n"
    report += f"- 平均奖励: {best_model[1]:.2f}\n"
    report += f"- 标准差: {best_model[2]:.2f}\n"
    report += f"- 稳定性: {'高' if best_model[2] < np.mean(stds) else '中等'}\n\n"
    
    if len(sorted_data) > 1:
        improvement = best_model[1] - sorted_data[1][1]
        report += f"### 性能提升\n"
        report += f"最佳模型比第二名高出: {improvement:.2f} ({improvement/sorted_data[1][1]*100:.1f}%)\n\n"
    
    # 保存报告
    with open('performance_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("性能报告已保存到 performance_report.md")
```

## 🛠️ 故障排除

### 常见问题及解决方案

1. **模型文件不存在**
   ```
   Error: Model file models/best_model.pth not found!
   ```
   - 检查模型文件路径是否正确
   - 确认训练已完成并保存了模型

2. **模型类型不匹配**
   ```
   RuntimeError: Error(s) in loading state_dict
   ```
   - 确保 `--model` 参数与训练时使用的模型类型一致
   - DQN模型使用 `--model dqn`
   - Rainbow模型使用 `--model rainbow`

3. **环境不兼容**
   ```
   gym.error.UnregisteredEnv
   ```
   - 检查环境名称是否正确
   - 确保安装了相应的游戏环境

4. **显存不足**
   ```
   CUDA out of memory
   ```
   - 减少批处理大小
   - 使用CPU评估：设置环境变量 `CUDA_VISIBLE_DEVICES=""`

## 📈 性能指标解读

### 关键指标说明

- **Mean Reward**: 平均奖励，越高越好
- **Standard Deviation**: 标准差，越小表示性能越稳定
- **Best/Worst Episode**: 最好/最差单回合表现
- **Episode Length**: 回合长度，通常越长表示智能体存活时间越久

### 评估建议

1. **充足的评估回合数**: 建议至少50-100回合以获得可靠统计
2. **多种子测试**: 使用不同随机种子评估模型鲁棒性
3. **长期稳定性**: 观察长时间评估中的性能变化
4. **环境泛化**: 在相似但不同的环境中测试模型

## 🎯 最佳实践

1. **定期评估**: 在训练过程中定期评估模型性能
2. **保存最佳模型**: 基于评估结果保存表现最好的模型
3. **性能监控**: 建立性能监控系统跟踪模型表现
4. **对比分析**: 对比不同算法和超参数的效果
5. **可视化分析**: 使用图表直观展示模型性能

通过本指南，您可以全面评估训练好的强化学习模型，获得详细的性能分析，并为进一步的模型优化提供数据支持。