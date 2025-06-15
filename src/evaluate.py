import os
import torch
import numpy as np
import argparse
import time
from tqdm import tqdm

from model import DQN, DuelingDQN
from agent import DQNAgent
from utils import make_env, record_video, display_video

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="DQN评估脚本")
    parser.add_argument("--env", type=str, default="ALE/Assault-v5", help="Gym环境名称")
    parser.add_argument("--model", type=str, default="dqn", choices=["dqn", "dueling"], help="模型类型")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--n_episodes", type=int, default=10, help="评估回合数")
    parser.add_argument("--max_steps", type=int, default=10000, help="每回合最大步数")
    parser.add_argument("--render", action="store_true", help="是否渲染游戏画面")
    parser.add_argument("--record_video", action="store_true", help="是否录制视频")
    parser.add_argument("--video_path", type=str, default="videos", help="视频保存目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    return parser.parse_args()

def evaluate_agent(args):
    """
    评估训练好的DQN智能体
    
    参数:
        args: 命令行参数
    """
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建环境
    env = make_env(args.env)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    if args.model == "dqn":
        model = DQN(input_shape, n_actions)
        target_model = DQN(input_shape, n_actions)
    else:  # dueling
        model = DuelingDQN(input_shape, n_actions)
        target_model = DuelingDQN(input_shape, n_actions)
    
    # 创建智能体
    agent = DQNAgent(
        model=model,
        target_model=target_model,
        env=env,
        device=device
    )
    
    # 加载模型
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        return
    
    print(f"Loading model from {args.model_path}")
    agent.load_model(args.model_path)
    
    # 评估统计
    episode_rewards = []
    episode_lengths = []
    
    # 创建视频保存目录
    if args.record_video:
        os.makedirs(args.video_path, exist_ok=True)
    
    print(f"Evaluating agent for {args.n_episodes} episodes...")
    
    # 评估循环
    for episode in tqdm(range(1, args.n_episodes + 1)):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        frames = []
        
        done = False
        truncated = False
        
        # 单回合循环
        for step in range(args.max_steps):
            # 记录帧（如果需要录制视频）
            if args.record_video:
                frame = env.render()
                frames.append(frame)
            
            # 选择动作（评估模式，不使用探索）
            action = agent.select_action(state, evaluate=True)
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            
            # 更新状态和统计
            state = next_state
            episode_reward += float(reward)
            episode_length += 1
            
            # 如果需要渲染，显示游戏画面
            if args.render:
                env.render()
                time.sleep(0.01)  # 稍微延迟以便观看
            
            if done or truncated:
                break
        
        # 记录回合统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 保存视频（如果需要）
        if args.record_video and len(frames) > 0:
            video_filename = os.path.join(args.video_path, f"episode_{episode}.mp4")
            save_video(frames, video_filename)
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # 计算统计信息
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Environment: {args.env}")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f} ± {std_length:.2f}")
    print(f"Best Episode Reward: {max(episode_rewards):.2f}")
    print(f"Worst Episode Reward: {min(episode_rewards):.2f}")
    print("="*50)
    
    return episode_rewards, episode_lengths

def save_video(frames, filename, fps=30):
    """
    保存视频文件
    
    参数:
        frames: 帧列表
        filename: 输出文件名
        fps: 每秒帧数
    """
    try:
        import cv2
        
        # 获取帧尺寸
        height, width, layers = frames[0].shape
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        # 写入帧
        for frame in frames:
            # OpenCV使用BGR格式，需要转换
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        # 释放资源
        video_writer.release()
        print(f"Video saved to {filename}")
        
    except ImportError:
        print("Warning: OpenCV not available, cannot save video")
    except Exception as e:
        print(f"Error saving video: {e}")

def demo_agent(args):
    """
    演示智能体玩游戏
    
    参数:
        args: 命令行参数
    """
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建环境（用于演示的环境需要支持渲染）
    env = make_env(args.env)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    if args.model == "dqn":
        model = DQN(input_shape, n_actions)
        target_model = DQN(input_shape, n_actions)
    else:  # dueling
        model = DuelingDQN(input_shape, n_actions)
        target_model = DuelingDQN(input_shape, n_actions)
    
    # 创建智能体
    agent = DQNAgent(
        model=model,
        target_model=target_model,
        env=env,
        device=device
    )
    
    # 加载模型
    print(f"Loading model from {args.model_path}")
    agent.load_model(args.model_path)
    
    # 录制视频
    print("Recording gameplay...")
    frames, total_reward = record_video(env, agent, device, max_steps=args.max_steps)
    
    print(f"Total reward: {total_reward}")
    print(f"Recorded {len(frames)} frames")
    
    # 显示视频
    if frames:
        print("Displaying gameplay...")
        display_video(frames)
    
    return frames, total_reward

if __name__ == "__main__":
    args = parse_args()
    
    if args.render or args.record_video:
        # 如果需要渲染或录制视频，运行演示模式
        demo_agent(args)
    else:
        # 否则运行标准评估
        evaluate_agent(args)