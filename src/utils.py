import numpy as np
import torch
# 先导入ale_py以注册ALE命名空间，然后导入gymnasium
try:
    import ale_py
except ImportError:
    pass
import gymnasium as gym
import cv2
from collections import deque
import matplotlib.pyplot as plt
import time
# 移除不必要的FrameStack导入，项目使用自定义的StackFrames类

class SkipFrame(gym.Wrapper):
    """
    跳帧包装器
    每n帧执行一次动作，并返回最后一帧
    """
    def __init__(self, env, skip=4):
        """
        初始化跳帧包装器
        
        参数:
            env: 游戏环境
            skip: 跳过的帧数
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """
        执行动作并跳过指定帧数
        """
        total_reward = 0.0
        done = False
        info = {}
        
        # 重复执行动作skip次，累积奖励
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
                
        return obs, total_reward, done, truncated, info


class PreprocessFrame(gym.ObservationWrapper):
    """
    图像预处理包装器
    将图像调整大小、灰度化并归一化
    """
    def __init__(self, env, width=84, height=84):
        """
        初始化图像预处理包装器
        
        参数:
            env: 游戏环境
            width: 输出图像宽度
            height: 输出图像高度
        """
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, self.height, self.width),
            dtype=np.uint8
        )

    def observation(self, observation):
        """
        预处理观测
        """
        # 转换为灰度图
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # 调整大小
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # 转换为PyTorch格式 (C, H, W)
        frame = np.expand_dims(frame, axis=0)
        
        return frame


class StackFrames(gym.ObservationWrapper):
    """
    帧堆叠包装器
    将连续的n帧堆叠在一起作为状态
    """
    def __init__(self, env, num_stack=4):
        """
        初始化帧堆叠包装器
        
        参数:
            env: 游戏环境
            num_stack: 堆叠的帧数
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)
        
        # 更新观测空间
        low = np.repeat(self.observation_space.low, num_stack, axis=0)
        high = np.repeat(self.observation_space.high, num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs):
        """
        重置环境并初始化帧堆栈
        """
        observation, info = self.env.reset(**kwargs)
        
        # 初始化帧堆栈
        for _ in range(self.num_stack):
            self.frames.append(observation)
            
        return self.observation(None), info

    def observation(self, observation):
        """
        返回堆叠的帧
        """
        # 如果不是初始化，则添加新的观测到堆栈
        if observation is not None:
            self.frames.append(observation)
            
        # 堆叠帧
        stacked_frames = np.concatenate(list(self.frames), axis=0)
        
        return stacked_frames


def make_env(env_name):
    """
    创建并包装游戏环境
    
    参数:
        env_name: 环境名称
        
    返回:
        包装后的环境
    """
    env = gym.make(env_name, render_mode="rgb_array")
    env = SkipFrame(env)
    env = PreprocessFrame(env)
    env = StackFrames(env)
    
    return env


def make_env_with_render(env_name, render_mode="human"):
    """
    创建并包装游戏环境（支持自定义渲染模式）
    
    参数:
        env_name: 环境名称
        render_mode: 渲染模式 ("human", "rgb_array", "ansi"等)
        
    返回:
        包装后的环境
    """
    env = gym.make(env_name, render_mode=render_mode)
    env = SkipFrame(env)
    env = PreprocessFrame(env)
    env = StackFrames(env)
    
    return env


def plot_rewards(rewards, avg_rewards, title="Training Rewards", save_path=None):
    """
    绘制奖励曲线
    
    参数:
        rewards: 每个回合的奖励列表
        avg_rewards: 平均奖励列表
        title: 图表标题
        save_path: 保存路径 (如果不为None则保存图表)
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward")
    plt.plot(avg_rewards, label="Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def display_video(frames, fps=30):
    """
    显示游戏视频
    
    参数:
        frames: 帧列表
        fps: 每秒帧数
    """
    plt.figure(figsize=(frames[0].shape[1]/72, frames[0].shape[0]/72), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
    
    for i in range(len(frames)):
        animate(i)
        plt.pause(1/fps)
        
    plt.close()


def record_video(env, agent, device, max_steps=1000):
    """
    记录智能体玩游戏的视频
    
    参数:
        env: 游戏环境
        agent: 智能体
        device: 计算设备
        max_steps: 最大步数
        
    返回:
        frames: 帧列表
        total_reward: 总奖励
    """
    frames = []
    state, _ = env.reset()
    total_reward = 0
    
    for _ in range(max_steps):
        # 记录帧
        frame = env.render()
        frames.append(frame)
        
        # 选择动作
        action = agent.select_action(state, evaluate=True)
        
        # 执行动作
        next_state, reward, done, truncated, _ = env.step(action)
        
        total_reward += reward
        state = next_state
        
        if done or truncated:
            break
    
    return frames, total_reward