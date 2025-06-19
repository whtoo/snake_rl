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

class ExperienceAugmenter:
    """
    Applies specified data augmentations to experiences (state and next_state).
    This can help improve generalization and robustness of the RL agent.
    Currently supports adding Gaussian noise to image-based states.
    """
    def __init__(self, augmentation_config=None):
        """
        Initializes the ExperienceAugmenter.

        Args:
            augmentation_config (dict, optional): Configuration specifying which augmentations
                to apply and their parameters.
                Example: {'add_noise': {'scale': 5.0}} to add Gaussian noise with std_dev 5.0.
                Defaults to None, meaning no augmentations are applied.
        """
        self.augmentation_config = augmentation_config if augmentation_config else {}

    def _add_gaussian_noise(self, state_image_stack, scale):
        """
        Adds Gaussian noise N(0, scale) to the input state image stack.
        The input is assumed to be a NumPy array of dtype uint8 (pixel values 0-255).
        Noise is added, and the result is clipped to the valid [0, 255] range and
        returned as a uint8 NumPy array.

        Args:
            state_image_stack (np.ndarray): The state (or next_state) image stack,
                typically with shape (num_stack, height, width) or (height, width).
            scale (float): The standard deviation of the Gaussian noise to add.
                           This scale should be relative to the pixel value range (0-255).

        Returns:
            np.ndarray: The augmented state image stack with noise, clipped and dtype uint8.
        """
        if not isinstance(state_image_stack, np.ndarray):
            # Ensure input is a NumPy array
            state_image_stack = np.array(state_image_stack, dtype=np.uint8)

        # Convert to float32 for noise addition to prevent overflow/underflow issues with uint8
        noisy_state = state_image_stack.astype(np.float32)

        # Generate Gaussian noise with the same shape as the state
        noise = np.random.normal(0, scale, noisy_state.shape).astype(np.float32)

        noisy_state += noise

        # Clip the values to the valid uint8 range [0, 255] and convert back to uint8
        noisy_state = np.clip(noisy_state, 0, 255)
        return noisy_state.astype(np.uint8)

    def augment(self, state, action, reward, next_state, done):
        """
        Applies configured augmentations to the state and next_state components of an experience tuple.
        Action, reward, and done fields are returned unmodified.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting next state.
            done (bool): Whether the episode terminated.

        Returns:
            tuple: The (potentially) augmented experience (aug_state, action, reward, aug_next_state, done).
        """
        aug_state = state
        aug_next_state = next_state

        if 'add_noise' in self.augmentation_config:
            noise_params = self.augmentation_config['add_noise']
            scale = noise_params.get('scale', 0.0) # Default scale if not provided
            if scale > 0:
                if state is not None:
                    aug_state = self._add_gaussian_noise(state, scale)
                if next_state is not None: # next_state can be None if episode terminates early in some setups
                    aug_next_state = self._add_gaussian_noise(next_state, scale)

        # Add other augmentations here based on self.augmentation_config
        # Example:
        # if 'random_flip' in self.augmentation_config:
        #     if state is not None:
        #         aug_state = self._random_horizontal_flip(aug_state) # Apply to already (potentially) noisy state
        #     if next_state is not None:
        #         aug_next_state = self._random_horizontal_flip(aug_next_state)

        return aug_state, action, reward, aug_next_state, done


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
    
    # Apply SkipFrame universally if desired, or also make it conditional
    env = SkipFrame(env, skip=4) # Assuming skip=4 is a good default

    # Conditionally apply image-specific wrappers
    if len(env.observation_space.shape) > 1 : # Indicates image-like data (e.g. H, W, C or H, W)
        env = PreprocessFrame(env)
        env = StackFrames(env, num_stack=4) # Assuming num_stack=4 is a good default
    # Note: If StackFrames is used for non-image data, it might need adjustments
    # or a different stacking mechanism if PreprocessFrame is skipped.
    # For CartPole, if PreprocessFrame and StackFrames are skipped,
    # the observation will be a 1D vector. DQNAgent expects (C,H,W) or similar.
    # This will require model adjustments for CartPole if these wrappers are skipped.
    # For now, this change addresses the cv2 error. Further compatibility for 1D envs
    # would require more changes in model input handling or a different set of wrappers.
    # Let's assume for Rainbow, image-based envs are the primary target,
    # so if these wrappers are skipped, the user must ensure the model matches the raw env output.
    # The current RainbowDQN model expects image-like input.
    # A better check might be specific to Atari-like envs.
    # For the purpose of this integration test with CartPole,
    # we must ensure the model can handle the (4,) shape if PreprocessFrame is skipped.
    # The provided model.py DQN/RainbowDQN expects (C,H,W) like input.
    # This means CartPole won't work with the current models if PreprocessFrame is skipped,
    # UNLESS the model itself is made flexible or CartPole is not the test env.
    #
    # Given the error, the immediate fix is to make PreprocessFrame conditional.
    # If CartPole is used for testing, and it bypasses PreprocessFrame,
    # then StackFrames will also likely fail or produce unexpected shapes if it expects a channel dim.
    # And the DQN model input layer will mismatch.
    #
    # Re-thinking: The original code *always* applied these.
    # This means CartPole was *always* being converted to an image by PreprocessFrame.
    # The error `scn is 1` means `cvtColor` received a single channel image.
    # `CartPole-v1` obs is `(4,)`. `make_env` calls `gym.make(env_name, render_mode="rgb_array")`.
    # For CartPole, even with `render_mode="rgb_array"`, `env.step()` returns the 4-float vector.
    # `env.render()` returns the image.
    # `PreprocessFrame` wrapper takes `observation` from `env.step()`.
    # So, `cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)` where observation is `(4,)` is the error.
    #
    # The fix is to ensure PreprocessFrame is only applied to actual image observations.
    # A more robust check for image data:
    is_image_env = len(env.observation_space.shape) == 3 and env.observation_space.shape[-1] == 3 # HWC format
    # Or, if some envs return (H,W) grayscale, or (C,H,W)
    # For now, let's assume typical Atari RGB output from `env.step()` if it's an image env.
    # However, `env.step()` for CartPole returns state vector, not image.
    # The `render_mode="rgb_array"` in `gym.make` affects `env.render()`.
    #
    # The issue is that PreprocessFrame *assumes* `observation` is an RGB image.
    # This assumption is wrong for CartPole.
    #
    # Correct approach: PreprocessFrame should only be added if the environment
    # is known to produce image frames as observations directly from step().
    # This is typically true for ALE (Atari Learning Environment) games.

    # A simple heuristic: if env_name starts with "ALE/" or contains "Pong", "Breakout", etc.
    # This is not perfectly robust. A better way is to check observation space properties.
    # If observation space is Box and shape is (H, W, C)
    obs_shape = env.observation_space.shape
    if isinstance(env.observation_space, gym.spaces.Box) and len(obs_shape) == 3 and obs_shape[0] > 10 and obs_shape[1] > 10: # Heuristic for HWC image
        env = PreprocessFrame(env)
        env = StackFrames(env)
    # If it's already preprocessed to (1, H, W) or (C, H, W) by some other wrapper
    elif isinstance(env.observation_space, gym.spaces.Box) and len(obs_shape) == 3 and (obs_shape[0] == 1 or obs_shape[0] == 3 or obs_shape[0] == 4):
        env = StackFrames(env) # Only stack if channels are already first dim

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
    env = SkipFrame(env, skip=4)

    obs_shape = env.observation_space.shape
    if isinstance(env.observation_space, gym.spaces.Box) and len(obs_shape) == 3 and obs_shape[0] > 10 and obs_shape[1] > 10:
        env = PreprocessFrame(env)
        env = StackFrames(env)
    elif isinstance(env.observation_space, gym.spaces.Box) and len(obs_shape) == 3 and (obs_shape[0] == 1 or obs_shape[0] == 3 or obs_shape[0] == 4):
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