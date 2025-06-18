#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量模型评估和对比脚本

这个脚本可以批量评估多个模型并生成详细的对比报告。
使用方法:
    python batch_evaluate.py --models_dir models/
    python batch_evaluate.py --config evaluation_config.json
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))


def find_model_files(models_dir: str) -> List[Dict[str, str]]:
    """
    在指定目录中查找模型文件
    
    参数:
        models_dir: 模型目录路径
    
    返回:
        模型信息列表
    """
    model_files = []
    
    for file in os.listdir(models_dir):
        if file.endswith('.pth'):
            file_path = os.path.join(models_dir, file)
            
            # 自动检测模型类型
            if 'rainbow' in file.lower():
                model_type = 'rainbow'
            elif 'dueling' in file.lower():
                model_type = 'dueling'
            else:
                model_type = 'dqn'
            
            model_files.append({
                'name': file.replace('.pth', ''),
                'path': file_path,
                'type': model_type
            })
    
    return model_files


def run_evaluation(model_config: Dict[str, str], episodes: int = 50) -> Dict[str, Any]:
    """
    运行单个模型的评估
    
    参数:
        model_config: 模型配置
        episodes: 评估回合数
    
    返回:
        评估结果
    """
    print(f"🔄 评估模型: {model_config['name']}")
    
    # 构建评估命令
    cmd = [
        sys.executable, "src/evaluate.py",
        "--model_path", model_config['path'],
        "--model", model_config['type'],
        "--n_episodes", str(episodes),
        "--seed", "42"
    ]
    
    try:
        # 执行评估
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        evaluation_time = time.time() - start_time
        
        if result.returncode == 0:
            # 解析输出
            stdout = result.stdout
            metrics = parse_evaluation_output(stdout)
            
            return {
                'name': model_config['name'],
                'type': model_config['type'],
                'path': model_config['path'],
                'success': True,
                'evaluation_time': evaluation_time,
                'episodes': episodes,
                **metrics
            }
        else:
            return {
                'name': model_config['name'],
                'type': model_config['type'],
                'path': model_config['path'],
                'success': False,
                'error': result.stderr,
                'evaluation_time': evaluation_time
            }
    
    except subprocess.TimeoutExpired:
        return {
            'name': model_config['name'],
            'type': model_config['type'],
            'path': model_config['path'],
            'success': False,
            'error': 'Evaluation timeout (>10 minutes)',
            'evaluation_time': 600
        }
    except Exception as e:
        return {
            'name': model_config['name'],
            'type': model_config['type'],
            'path': model_config['path'],
            'success': False,
            'error': str(e),
            'evaluation_time': 0
        }


def parse_evaluation_output(stdout: str) -> Dict[str, float]:
    """
    解析评估输出获取指标
    
    参数:
        stdout: 评估脚本的标准输出
    
    返回:
        解析出的指标字典
    """
    metrics = {}
    
    lines = stdout.split('\n')
    for line in lines:
        line = line.strip()
        
        if 'Mean Reward:' in line:
            # 解析 "Mean Reward: 1247.35 ± 234.67"
            parts = line.split(':')
            if len(parts) > 1:
                reward_part = parts[1].strip()
                if '±' in reward_part:
                    mean_val = float(reward_part.split('±')[0].strip())
                    std_val = float(reward_part.split('±')[1].strip())
                    metrics['mean_reward'] = mean_val
                    metrics['std_reward'] = std_val
        
        elif 'Mean Episode Length:' in line:
            # 解析 "Mean Episode Length: 2341.23 ± 456.78"
            parts = line.split(':')
            if len(parts) > 1:
                length_part = parts[1].strip()
                if '±' in length_part:
                    mean_val = float(length_part.split('±')[0].strip())
                    std_val = float(length_part.split('±')[1].strip())
                    metrics['mean_length'] = mean_val
                    metrics['std_length'] = std_val
        
        elif 'Best Episode Reward:' in line:
            parts = line.split(':')
            if len(parts) > 1:
                metrics['best_reward'] = float(parts[1].strip())
        
        elif 'Worst Episode Reward:' in line:
            parts = line.split(':')
            if len(parts) > 1:
                metrics['worst_reward'] = float(parts[1].strip())
    
    return metrics


def generate_comparison_report(results: List[Dict[str, Any]], output_dir: str):
    """
    生成对比报告
    
    参数:
        results: 评估结果列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 过滤成功的结果
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ 没有成功的评估结果，无法生成报告")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(successful_results)
    
    # 生成图表
    create_comparison_plots(df, output_dir)
    
    # 生成文本报告
    create_text_report(df, output_dir)
    
    # 保存详细数据
    df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    with open(os.path.join(output_dir, 'raw_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📊 报告已生成到目录: {output_dir}")


def create_comparison_plots(df: pd.DataFrame, output_dir: str):
    """
    创建对比图表
    
    参数:
        df: 结果数据框
        output_dir: 输出目录
    """
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('模型性能对比分析', fontsize=16, fontweight='bold')
    
    # 1. 平均奖励对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['name'], df['mean_reward'], yerr=df['std_reward'], 
                   capsize=5, alpha=0.8, color='skyblue')
    ax1.set_title('平均奖励对比')
    ax1.set_ylabel('平均奖励')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars1, df['mean_reward']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 2. 稳定性对比（变异系数）
    ax2 = axes[0, 1]
    cv = df['std_reward'] / df['mean_reward']  # 变异系数
    bars2 = ax2.bar(df['name'], cv, alpha=0.8, color='lightcoral')
    ax2.set_title('稳定性对比（变异系数，越小越稳定）')
    ax2.set_ylabel('变异系数')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars2, cv):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. 最佳表现对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(df['name'], df['best_reward'], alpha=0.8, color='lightgreen')
    ax3.set_title('最佳单回合表现')
    ax3.set_ylabel('最佳奖励')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars3, df['best_reward']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 4. 综合评分（归一化）
    ax4 = axes[1, 1]
    # 计算综合评分：平均奖励权重0.6 + 稳定性权重0.4
    normalized_reward = (df['mean_reward'] - df['mean_reward'].min()) / (df['mean_reward'].max() - df['mean_reward'].min())
    normalized_stability = 1 - ((cv - cv.min()) / (cv.max() - cv.min()))  # 稳定性越高越好
    composite_score = 0.6 * normalized_reward + 0.4 * normalized_stability
    
    colors = plt.cm.viridis(composite_score)
    bars4 = ax4.bar(df['name'], composite_score, color=colors, alpha=0.8)
    ax4.set_title('综合评分（奖励60% + 稳定性40%）')
    ax4.set_ylabel('综合评分')
    ax4.set_ylim(0, 1)
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars4, composite_score):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建详细的性能分布图
    create_performance_distribution_plot(df, output_dir)


def create_performance_distribution_plot(df: pd.DataFrame, output_dir: str):
    """
    创建性能分布图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 奖励分布箱线图
    model_names = df['name'].tolist()
    reward_data = []
    
    # 这里我们使用正态分布模拟每个模型的奖励分布
    for _, row in df.iterrows():
        # 基于均值和标准差生成模拟数据
        simulated_rewards = np.random.normal(row['mean_reward'], row['std_reward'], 100)
        reward_data.append(simulated_rewards)
    
    ax1.boxplot(reward_data, labels=model_names)
    ax1.set_title('奖励分布对比')
    ax1.set_ylabel('奖励值')
    ax1.tick_params(axis='x', rotation=45)
    
    # 性能雷达图（如果模型数量适中）
    if len(df) <= 6:
        create_radar_chart(df, ax2)
    else:
        # 如果模型太多，显示排名图
        df_sorted = df.sort_values('mean_reward', ascending=True)
        ax2.barh(range(len(df_sorted)), df_sorted['mean_reward'])
        ax2.set_yticks(range(len(df_sorted)))
        ax2.set_yticklabels(df_sorted['name'])
        ax2.set_title('模型排名（按平均奖励）')
        ax2.set_xlabel('平均奖励')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_radar_chart(df: pd.DataFrame, ax):
    """
    创建雷达图
    """
    # 准备数据
    categories = ['平均奖励', '稳定性', '最佳表现', '评估速度']
    
    # 归一化数据
    normalized_data = []
    for _, row in df.iterrows():
        reward_norm = (row['mean_reward'] - df['mean_reward'].min()) / (df['mean_reward'].max() - df['mean_reward'].min())
        stability_norm = 1 - ((row['std_reward'] / row['mean_reward']) - (df['std_reward'] / df['mean_reward']).min()) / ((df['std_reward'] / df['mean_reward']).max() - (df['std_reward'] / df['mean_reward']).min())
        best_norm = (row['best_reward'] - df['best_reward'].min()) / (df['best_reward'].max() - df['best_reward'].min())
        speed_norm = 1 - ((row['evaluation_time'] - df['evaluation_time'].min()) / (df['evaluation_time'].max() - df['evaluation_time'].min()))
        
        normalized_data.append([reward_norm, stability_norm, best_norm, speed_norm])
    
    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
    
    for i, (_, row) in enumerate(df.iterrows()):
        values = normalized_data[i] + normalized_data[i][:1]  # 闭合数据
        ax.plot(angles, values, 'o-', linewidth=2, label=row['name'], color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_ylim(0, 1)
    ax.set_title('综合性能雷达图')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))


def create_text_report(df: pd.DataFrame, output_dir: str):
    """
    创建文本报告
    """
    report = []
    report.append("# 模型性能评估报告\n")
    report.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"评估模型数量: {len(df)}\n\n")
    
    # 排序模型
    df_sorted = df.sort_values('mean_reward', ascending=False)
    
    report.append("## 📊 模型排名（按平均奖励）\n\n")
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        report.append(f"{i}. **{row['name']}** ({row['type']})\n")
        report.append(f"   - 平均奖励: {row['mean_reward']:.2f} ± {row['std_reward']:.2f}\n")
        report.append(f"   - 最佳表现: {row['best_reward']:.2f}\n")
        report.append(f"   - 最差表现: {row['worst_reward']:.2f}\n")
        report.append(f"   - 变异系数: {row['std_reward']/row['mean_reward']:.3f}\n")
        report.append(f"   - 评估时间: {row['evaluation_time']:.1f}秒\n\n")
    
    # 统计分析
    report.append("## 📈 统计分析\n\n")
    
    best_model = df_sorted.iloc[0]
    report.append(f"### 最佳模型: {best_model['name']}\n")
    report.append(f"- 模型类型: {best_model['type']}\n")
    report.append(f"- 平均奖励: {best_model['mean_reward']:.2f}\n")
    report.append(f"- 稳定性评级: {'优秀' if best_model['std_reward']/best_model['mean_reward'] < 0.3 else '良好' if best_model['std_reward']/best_model['mean_reward'] < 0.5 else '一般'}\n\n")
    
    if len(df_sorted) > 1:
        second_best = df_sorted.iloc[1]
        improvement = best_model['mean_reward'] - second_best['mean_reward']
        improvement_pct = improvement / second_best['mean_reward'] * 100
        report.append(f"### 性能提升\n")
        report.append(f"最佳模型比第二名高出: {improvement:.2f} ({improvement_pct:.1f}%)\n\n")
    
    # 模型类型分析
    type_analysis = df.groupby('type').agg({
        'mean_reward': ['mean', 'std', 'count'],
        'std_reward': 'mean'
    }).round(2)
    
    report.append("### 模型类型分析\n\n")
    for model_type in type_analysis.index:
        count = type_analysis.loc[model_type, ('mean_reward', 'count')]
        avg_reward = type_analysis.loc[model_type, ('mean_reward', 'mean')]
        std_reward = type_analysis.loc[model_type, ('mean_reward', 'std')]
        report.append(f"- **{model_type.upper()}** ({count}个模型):\n")
        report.append(f"  - 平均奖励: {avg_reward:.2f} ± {std_reward:.2f}\n")
    
    report.append("\n## 🎯 建议\n\n")
    
    # 基于结果给出建议
    cv_threshold = 0.4
    high_variance_models = df[df['std_reward'] / df['mean_reward'] > cv_threshold]
    
    if len(high_variance_models) > 0:
        report.append("### 稳定性改进建议\n")
        report.append("以下模型表现不够稳定，建议：\n")
        for _, model in high_variance_models.iterrows():
            report.append(f"- **{model['name']}**: 考虑调整超参数或增加训练时间\n")
        report.append("\n")
    
    # 保存报告
    with open(os.path.join(output_dir, 'evaluation_report.md'), 'w', encoding='utf-8') as f:
        f.writelines(report)


def main():
    parser = argparse.ArgumentParser(description="批量模型评估工具")
    parser.add_argument("--models_dir", type=str, help="模型目录路径")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--episodes", type=int, default=50, help="每个模型的评估回合数")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                       help="结果输出目录")
    parser.add_argument("--parallel", type=int, default=1, 
                       help="并行评估进程数（暂未实现）")
    
    args = parser.parse_args()
    
    # 获取模型列表
    if args.config:
        # 从配置文件加载
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        model_configs = config['models']
        episodes = config.get('episodes', args.episodes)
    elif args.models_dir:
        # 从目录自动发现
        model_configs = find_model_files(args.models_dir)
        episodes = args.episodes
    else:
        print("❌ 请指定 --models_dir 或 --config 参数")
        return
    
    if not model_configs:
        print("❌ 未找到任何模型文件")
        return
    
    print(f"🔍 发现 {len(model_configs)} 个模型")
    print(f"📊 每个模型将评估 {episodes} 回合")
    print("="*50)
    
    # 执行批量评估
    results = []
    for i, model_config in enumerate(model_configs, 1):
        print(f"\n[{i}/{len(model_configs)}] ", end="")
        result = run_evaluation(model_config, episodes)
        results.append(result)
        
        if result['success']:
            print(f"✅ 完成 - 平均奖励: {result['mean_reward']:.2f}")
        else:
            print(f"❌ 失败 - {result['error']}")
    
    # 生成报告
    print("\n" + "="*50)
    print("📊 生成评估报告...")
    generate_comparison_report(results, args.output_dir)
    
    # 显示简要总结
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_model = max(successful_results, key=lambda x: x['mean_reward'])
        print(f"\n🏆 最佳模型: {best_model['name']} (平均奖励: {best_model['mean_reward']:.2f})")
    
    print(f"\n📁 详细报告已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()