#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速优化测试脚本
用于验证Rainbow DQN优化配置的效果
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import torch
import numpy as np
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

from src.model import RainbowDQN
from src.agent import RainbowAgent
from src.utils import make_env
from optimized_config import OPTIMIZED_CONFIG

def quick_test():
    """
    快速测试优化配置效果
    """
    print("=== Rainbow DQN 优化配置快速测试 ===")
    
    # 使用优化配置
    config = OPTIMIZED_CONFIG.copy()
    config['episodes'] = 3  # 只测试3个回合
    config['memory_size'] = 1000  # 减小内存大小
    config['learning_starts'] = 100  # 更早开始学习
    config['use_per'] = config.get('prioritized_replay', True)
    config['augmentation_config'] = {'add_noise': {'scale': config.get('aug_noise_scale', 3.0)}} if config.get('use_state_augmentation', True) else None
    
    print(f"学习率: {config['lr']}")
    print(f"批量大小: {config['batch_size']}")
    print(f"梯度裁剪: {config['grad_clip_norm']}")
    print(f"Huber Delta: {config['huber_delta']}")
    print("="*50)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env = make_env("ALE/Assault-v5")
    
    # 获取环境信息
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    print(f"状态形状: {state_shape}")
    print(f"动作数量: {n_actions}")
    
    # 创建模型
    model = RainbowDQN(
        input_shape=state_shape,
        n_actions=n_actions,
        use_noisy=config['use_noisy'],
        use_distributional=config['use_distributional'],
        n_atoms=config['n_atoms'],
        v_min=config['v_min'],
        v_max=config['v_max']
    ).to(device)
    
    # 创建目标模型
    target_model = RainbowDQN(
        input_shape=state_shape,
        n_actions=n_actions,
        use_noisy=config['use_noisy'],
        use_distributional=config['use_distributional'],
        n_atoms=config['n_atoms'],
        v_min=config['v_min'],
        v_max=config['v_max']
    ).to(device)
    
    # 创建智能体
    agent = RainbowAgent(
        model=model,
        target_model=target_model,
        env=env,
        device=device,
        base_n_step=config['base_n_step'],
        max_n_step=config['max_n_step'],
        adapt_n_step_freq=config['adapt_n_step_freq'],
        td_error_threshold_low=config['td_error_threshold_low'],
        td_error_threshold_high=config['td_error_threshold_high'],
        augmentation_config=config['augmentation_config'],
        use_noisy=config['use_noisy'],
        use_distributional=config['use_distributional'],
        n_atoms=config['n_atoms'],
        v_min=config['v_min'],
        v_max=config['v_max'],
        huber_delta=config['huber_delta'],
        grad_clip_norm=config['grad_clip_norm'],
        buffer_size=config['memory_size'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        lr=config['lr'],
        epsilon_start=config['epsilon_start'],
        epsilon_final=config['epsilon_final'],
        epsilon_decay=config['epsilon_decay'],
        target_update=config['target_update'],
        prioritized_replay=config['use_per']
    )
    
    print("开始快速测试...")
    
    # 训练循环
    episode_rewards = []
    episode_losses = []
    
    for episode in range(config['episodes']):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        steps = 0
        start_time = time.time()
        
        while True:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            
            # 存储经验
            agent.store_experience(state, action, reward, next_state, done or truncated)
            
            # 学习
            if len(agent.memory) >= config['learning_starts']:
                loss = agent.update_model()
                if loss is not None:
                    episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done or truncated or steps >= 200:  # 限制最大步数
                break
        
        # 计算平均损失
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        duration = time.time() - start_time
        
        episode_rewards.append(episode_reward)
        episode_losses.append(avg_loss)
        
        print(f"QUICK TEST (Ep: {episode+1}/{config['episodes']}) "
              f"Reward: {episode_reward:.2f}, Avg Loss: {avg_loss:.4f}, "
              f"Duration: {duration:.2f}s, Steps: {steps}")
    
    # 输出测试结果
    print("\n=== 测试结果 ===")
    print(f"平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"平均损失: {np.mean(episode_losses):.4f}")
    print(f"损失标准差: {np.std(episode_losses):.4f}")
    
    # 检查损失趋势
    if len(episode_losses) >= 2:
        loss_trend = episode_losses[-1] - episode_losses[0]
        if loss_trend < 0:
            print(f"✓ 损失下降趋势: {loss_trend:.4f}")
        else:
            print(f"⚠ 损失上升趋势: {loss_trend:.4f}")
    
    print("\n=== 优化配置验证完成 ===")
    
    env.close()
    return episode_rewards, episode_losses

if __name__ == "__main__":
    quick_test()