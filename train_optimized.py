#!/usr/bin/env python3
"""
优化版训练脚本 - Rainbow DQN 损失优化

使用优化后的超参数和配置进行训练，旨在降低损失值并提高性能。
"""

import os
import sys
import torch
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import signal
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import DQN, DuelingDQN, RainbowDQN
from src.agent import DQNAgent, RainbowAgent
from src.utils import make_env, plot_rewards
from src.input_shield import input_shield_context
from optimized_config import OPTIMIZED_CONFIG
import gc # For explicit garbage collection

# 全局变量用于信号处理
stop_training = False

def signal_handler(signum, frame):
    global stop_training
    print("\n收到中断信号，正在安全停止训练...")
    stop_training = True

def parse_args():
    parser = argparse.ArgumentParser(description="优化版 Rainbow DQN 训练")
    
    # 基础参数
    parser.add_argument("--env", type=str, default="ALE/Assault-v5", help="环境名称")
    parser.add_argument("--model", type=str, default="rainbow", 
                       choices=["dqn", "dueling", "rainbow"], help="模型类型")
    parser.add_argument("--use_optimized_config", action="store_true", default=True,
                       help="使用优化配置")
    
    # 可覆盖的优化参数
    parser.add_argument("--episodes", type=int, help="训练回合数")
    parser.add_argument("--lr", type=float, help="学习率")
    parser.add_argument("--batch_size", type=int, help="批量大小")
    parser.add_argument("--target_update", type=int, help="目标网络更新频率")
    parser.add_argument("--grad_clip_norm", type=float, help="梯度裁剪范数")
    parser.add_argument("--huber_delta", type=float, help="Huber Loss delta参数")
    
    # 目录参数
    parser.add_argument("--save_dir", type=str, default="checkpoints_optimized", 
                       help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default="logs_optimized", 
                       help="日志保存目录")
    
    return parser.parse_args()

def get_config(args):
    """获取最终配置，命令行参数可覆盖优化配置"""
    if args.use_optimized_config:
        config = OPTIMIZED_CONFIG.copy()
    else:
        # 使用默认配置
        config = {
            'lr': 1e-4,
            'batch_size': 32,
            'episodes': 100,
            'target_update': 1000,
            'grad_clip_norm': 10,
            'huber_delta': 1.0,
        }
    
    # 命令行参数覆盖配置
    for key in ['episodes', 'lr', 'batch_size', 'target_update', 'grad_clip_norm', 'huber_delta']:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)
    
    return config

def create_model_and_agent(model_type, config, env, device):
    """创建模型和智能体"""
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    if model_type == "rainbow":
        model = RainbowDQN(
            input_shape=input_shape,
            n_actions=n_actions,
            use_noisy=config.get('use_noisy', True),
            use_distributional=config.get('use_distributional', True),
            n_atoms=config.get('n_atoms', 51),
            v_min=config.get('v_min', -10),
            v_max=config.get('v_max', 10)
        )
        target_model = RainbowDQN(
            input_shape=input_shape,
            n_actions=n_actions,
            use_noisy=config.get('use_noisy', True),
            use_distributional=config.get('use_distributional', True),
            n_atoms=config.get('n_atoms', 51),
            v_min=config.get('v_min', -10),
            v_max=config.get('v_max', 10)
        )
        
        agent = RainbowAgent(
            model=model,
            target_model=target_model,
            env=env,
            device=device,
            buffer_size=config.get('buffer_size', 100000),
            batch_size=config['batch_size'],
            gamma=config.get('gamma', 0.99),
            lr=config['lr'],
            target_update=config['target_update'],
            prioritized_replay=config.get('prioritized_replay', True),
            use_noisy=config.get('use_noisy', True),
            use_distributional=config.get('use_distributional', True),
            n_atoms=config.get('n_atoms', 51),
            v_min=config.get('v_min', -10),
            v_max=config.get('v_max', 10),
            base_n_step=config.get('base_n_step', 3),
            max_n_step=config.get('max_n_step', 8),
            adapt_n_step_freq=config.get('adapt_n_step_freq', 1500),
            td_error_threshold_low=config.get('td_error_threshold_low', 0.08),
            td_error_threshold_high=config.get('td_error_threshold_high', 0.4),
            huber_delta=config['huber_delta'],
            grad_clip_norm=config['grad_clip_norm']
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return agent

def main():
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 解析参数
    args = parse_args()
    config = get_config(args)
    
    print("=== 优化版 Rainbow DQN 训练 ===")
    print(f"使用优化配置: {args.use_optimized_config}")
    print(f"学习率: {config['lr']}")
    print(f"批量大小: {config['batch_size']}")
    print(f"目标网络更新频率: {config['target_update']}")
    print(f"梯度裁剪范数: {config['grad_clip_norm']}")
    print(f"Huber Delta: {config['huber_delta']}")
    print("="*50)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env = make_env(args.env)
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建模型和智能体
    agent = create_model_and_agent(args.model, config, env, device)
    
    # 创建TensorBoard日志记录器
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 训练统计
    rewards = []
    losses = []
    best_avg_reward = -float("inf")
    
    # 训练循环
    total_steps = 0
    start_time = time.time()
    
    print("开始优化训练。按 Ctrl+C 终止训练。")
    
    with input_shield_context():
        for episode in range(1, config['episodes'] + 1):
            if stop_training:
                print(f"在第 {episode} 回合开始前终止训练。")
                break
            
            state, _ = env.reset()
            episode_reward = 0
            episode_loss = 0
            episode_steps = 0
            episode_start_time = time.time()
            
            done = False
            truncated = False
            
            while not (done or truncated):
                if stop_training:
                    break
                
                # 选择动作
                action = agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, truncated, _ = env.step(action)
                
                # 存储经验
                agent.store_experience(state, action, reward, next_state, done)
                
                # 更新模型
                if total_steps > agent.batch_size: # Ensure buffer has enough samples for a batch
                    # print(f"[MEM_TRACE] train_optimized.py: Before agent.update_model(), total_steps={total_steps}")
                    loss = agent.update_model()
                    # print(f"[MEM_TRACE] train_optimized.py: After agent.update_model(), loss={loss}, total_steps={total_steps}")
                    if loss is not None:
                        episode_loss += loss
                        episode_steps += 1
                else:
                    loss = None # Or some other indicator that model wasn't updated
                
                # 更新目标网络
                if total_steps > 0 and total_steps % config['target_update'] == 0:
                    # print(f"[MEM_TRACE] train_optimized.py: Before agent.update_target_model(), total_steps={total_steps}")
                    agent.update_target_model()
                    # print(f"[MEM_TRACE] train_optimized.py: After agent.update_target_model(), total_steps={total_steps}")
                
                state = next_state
                episode_reward += reward
                total_steps += 1
            
            if stop_training:
                break
            
            # 记录统计
            rewards.append(episode_reward)
            avg_reward = np.mean(rewards[-100:])
            
            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0.0
            losses.append(avg_loss)
            
            episode_duration = time.time() - episode_start_time
            
            # 打印训练日志
            log_message = (f"OPTIMIZED RAINBOW (Ep: {episode}/{config['episodes']}) "
                          f"Reward: {episode_reward:.2f}, Avg Loss: {avg_loss:.4f}, "
                          f"Duration: {episode_duration:.2f}s, Steps: {episode_steps}")
            print(log_message)
            sys.stdout.flush()
            
            # 记录到TensorBoard
            writer.add_scalar('Reward/Episode', episode_reward, episode)
            writer.add_scalar('Reward/Average', avg_reward, episode)
            writer.add_scalar('Loss/Average', avg_loss, episode)
            writer.add_scalar('Training/Duration', episode_duration, episode)
            writer.add_scalar('Training/Steps', episode_steps, episode)
            
            # 评估和保存
            if episode % config.get('eval_interval', 10) == 0:
                eval_reward = evaluate_agent(agent, env, num_episodes=5)
                print(f"评估奖励 (第{episode}回合): {eval_reward:.2f}")
                sys.stdout.flush()
                
                writer.add_scalar('Evaluation/Reward', eval_reward, episode)
                
                # 保存最佳模型
                if eval_reward > best_avg_reward:
                    best_avg_reward = eval_reward
                    best_model_path = os.path.join(args.save_dir, f"best_model_{args.model}_optimized.pth")
                    agent.save_model(best_model_path)
                    print(f"保存最佳模型: {best_model_path}")
                    sys.stdout.flush()
            
            # 定期保存
            if episode % config.get('save_interval', 50) == 0:
                model_path = os.path.join(args.save_dir, f"model_{args.model}_ep{episode}_optimized.pth")
                agent.save_model(model_path)
                print(f"保存模型检查点: {model_path}")
                sys.stdout.flush()

            # Explicit garbage collection at the end of each episode
            if episode % 10 == 0:
                # print(f"[GC_TRACE] Collecting garbage at end of Episode {episode}, Total Steps {total_steps}")
                gc.collect()
                # print(f"[GC_TRACE] Garbage collection complete for Episode {episode}")
                sys.stdout.flush() # Ensure GC messages (if any uncommented) are printed
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, f"final_model_{args.model}_optimized.pth")
    agent.save_model(final_model_path)
    
    total_time = time.time() - start_time
    print(f"\n训练完成！")
    print(f"总训练时间: {total_time:.2f}秒")
    print(f"最佳平均奖励: {best_avg_reward:.2f}")
    print(f"最终平均损失: {np.mean(losses[-10:]):.4f}")
    print(f"模型已保存到: {final_model_path}")
    
    writer.close()

def evaluate_agent(agent, env, num_episodes=5):
    """评估智能体性能"""
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / num_episodes

if __name__ == "__main__":
    main()