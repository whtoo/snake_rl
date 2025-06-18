#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速模型评估脚本

这个脚本提供了一个简单的接口来快速评估训练好的模型。
使用方法:
    python quick_evaluate.py models/best_model.pth
    python quick_evaluate.py models/rainbow_model.pth --model rainbow
"""

import sys
import os
import argparse
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.evaluate import evaluate_agent, parse_args


def quick_evaluate(model_path, model_type="dqn", episodes=20, render=False):
    """
    快速评估模型
    
    参数:
        model_path: 模型文件路径
        model_type: 模型类型 ("dqn", "dueling", "rainbow")
        episodes: 评估回合数
        render: 是否显示游戏画面
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件 {model_path} 不存在!")
        return None
    
    print(f"🎮 开始评估模型: {model_path}")
    print(f"📊 模型类型: {model_type}")
    print(f"🔢 评估回合数: {episodes}")
    print("="*50)
    
    # 构造参数
    class Args:
        def __init__(self):
            self.env = "ALE/Assault-v5"
            self.model = model_type
            self.model_path = model_path
            self.n_episodes = episodes
            self.max_steps = 10000
            self.render = render
            self.record_video = False
            self.video_path = "videos"
            self.seed = 42
    
    args = Args()
    
    try:
        # 执行评估
        episode_rewards, episode_lengths = evaluate_agent(args)
        
        # 计算额外统计信息
        import numpy as np
        
        success_rate = len([r for r in episode_rewards if r > 0]) / len(episode_rewards)
        median_reward = np.median(episode_rewards)
        
        print("\n🎯 额外统计信息:")
        print(f"成功率 (奖励>0): {success_rate:.1%}")
        print(f"中位数奖励: {median_reward:.2f}")
        print(f"奖励范围: {min(episode_rewards):.2f} ~ {max(episode_rewards):.2f}")
        
        # 性能评级
        mean_reward = np.mean(episode_rewards)
        if mean_reward > 1000:
            grade = "🏆 优秀"
        elif mean_reward > 500:
            grade = "🥈 良好"
        elif mean_reward > 100:
            grade = "🥉 一般"
        else:
            grade = "📈 需要改进"
        
        print(f"\n📈 性能评级: {grade}")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': mean_reward,
            'success_rate': success_rate,
            'grade': grade
        }
        
    except Exception as e:
        print(f"❌ 评估过程中发生错误: {e}")
        return None


def auto_detect_model_type(model_path):
    """
    根据文件名自动检测模型类型
    
    参数:
        model_path: 模型文件路径
    
    返回:
        模型类型字符串
    """
    filename = os.path.basename(model_path).lower()
    
    if 'rainbow' in filename:
        return 'rainbow'
    elif 'dueling' in filename:
        return 'dueling'
    else:
        return 'dqn'


def main():
    parser = argparse.ArgumentParser(description="快速模型评估工具")
    parser.add_argument("model_path", help="模型文件路径")
    parser.add_argument("--model", type=str, 
                       choices=["dqn", "dueling", "rainbow"],
                       help="模型类型 (如果不指定，将自动检测)")
    parser.add_argument("--episodes", type=int, default=20, 
                       help="评估回合数 (默认: 20)")
    parser.add_argument("--render", action="store_true", 
                       help="显示游戏画面")
    parser.add_argument("--compare", nargs="+", 
                       help="对比多个模型 (提供多个模型路径)")
    
    args = parser.parse_args()
    
    if args.compare:
        # 对比模式
        print("🔄 对比模式: 评估多个模型")
        results = []
        
        all_models = [args.model_path] + args.compare
        for model_path in all_models:
            model_type = args.model or auto_detect_model_type(model_path)
            result = quick_evaluate(model_path, model_type, args.episodes, args.render)
            if result:
                result['model_path'] = model_path
                result['model_type'] = model_type
                results.append(result)
        
        # 显示对比结果
        if results:
            print("\n" + "="*60)
            print("📊 模型对比结果")
            print("="*60)
            
            # 按平均奖励排序
            results.sort(key=lambda x: x['mean_reward'], reverse=True)
            
            for i, result in enumerate(results, 1):
                model_name = os.path.basename(result['model_path'])
                print(f"{i}. {model_name} ({result['model_type']})")
                print(f"   平均奖励: {result['mean_reward']:.2f}")
                print(f"   成功率: {result['success_rate']:.1%}")
                print(f"   评级: {result['grade']}")
                print()
    
    else:
        # 单模型评估
        model_type = args.model or auto_detect_model_type(args.model_path)
        quick_evaluate(args.model_path, model_type, args.episodes, args.render)


if __name__ == "__main__":
    main()