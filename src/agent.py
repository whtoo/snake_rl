import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

# 定义经验元组的结构
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """
    经验回放缓冲区
    存储和采样智能体与环境交互的经验
    """
    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        
        参数:
            capacity: 缓冲区容量 (最大存储经验数量)
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        将经验存入缓冲区
        """
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        从缓冲区随机采样一批经验
        
        参数:
            batch_size: 批量大小
            
        返回:
            states, actions, rewards, next_states, dones 的批量数据
        """
        experiences = random.sample(self.buffer, batch_size)
        
        # 将经验批量转换为张量形式
        states = torch.cat([torch.FloatTensor(exp.state).unsqueeze(0) for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float).unsqueeze(1)
        next_states = torch.cat([torch.FloatTensor(exp.next_state).unsqueeze(0) for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        返回缓冲区当前大小
        """
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    根据TD误差对经验进行优先级采样
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        初始化优先经验回放缓冲区
        
        参数:
            capacity: 缓冲区容量
            alpha: 优先级指数 (0 - 均匀采样, 1 - 完全按优先级采样)
            beta_start: 初始重要性采样指数
            beta_frames: beta从beta_start到1的帧数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def beta_by_frame(self, frame_idx):
        """
        根据当前帧计算beta值
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        """
        将经验存入缓冲区
        """
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = Experience(state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        根据优先级采样经验
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        # 计算采样概率
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # 计算重要性权重
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        # 计算权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
        
        # 将经验批量转换为张量形式
        states = torch.cat([torch.FloatTensor(exp.state).unsqueeze(0) for exp in samples])
        actions = torch.tensor([exp.action for exp in samples], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([exp.reward for exp in samples], dtype=torch.float).unsqueeze(1)
        next_states = torch.cat([torch.FloatTensor(exp.next_state).unsqueeze(0) for exp in samples])
        dones = torch.tensor([exp.done for exp in samples], dtype=torch.float).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        更新经验的优先级
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        """
        返回缓冲区当前大小
        """
        return len(self.buffer)


class DQNAgent:
    """
    DQN智能体
    """
    def __init__(self, model, target_model, env, device, 
                 buffer_size=100000, batch_size=32, gamma=0.99,
                 lr=1e-4, epsilon_start=1.0, epsilon_final=0.01,
                 epsilon_decay=10000, target_update=1000,
                 prioritized_replay=False):
        """
        初始化DQN智能体
        
        参数:
            model: 主Q网络
            target_model: 目标Q网络
            env: 游戏环境
            device: 计算设备 (CPU/GPU)
            buffer_size: 经验回放缓冲区大小
            batch_size: 训练批量大小
            gamma: 折扣因子
            lr: 学习率
            epsilon_start: 初始探索率
            epsilon_final: 最终探索率
            epsilon_decay: 探索率衰减帧数
            target_update: 目标网络更新频率
            prioritized_replay: 是否使用优先经验回放
        """
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # 目标网络不进行梯度更新
        
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        
        # 创建经验回放缓冲区
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size)
            self.prioritized_replay = True
        else:
            self.memory = ReplayBuffer(buffer_size)
            self.prioritized_replay = False
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 探索参数
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        
        # 训练步数计数器
        self.steps_done = 0
    
    def select_action(self, state, evaluate=False):
        """
        根据当前状态选择动作
        
        参数:
            state: 当前状态
            evaluate: 是否处于评估模式 (不使用探索)
            
        返回:
            选择的动作
        """
        # 计算当前探索率
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                 np.exp(-1. * self.steps_done / self.epsilon_decay)
        
        self.steps_done += 1
        
        # 评估模式或随机数大于探索率时，使用贪婪策略
        if evaluate or random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.model(state_tensor)
                return q_values.max(1)[1].item()
        else:
            # 随机探索
            return random.randrange(self.env.action_space.n)
    
    def update_model(self):
        """
        更新模型参数
        """
        # 如果经验不足，不进行更新
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 从经验回放缓冲区采样
        if self.prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones_like(rewards).to(self.device)
        
        # 将数据转移到设备
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算当前Q值
        q_values = self.model(states).gather(1, actions)
        
        # 计算目标Q值 (使用目标网络)
        with torch.no_grad():
            # 双DQN: 使用主网络选择动作，使用目标网络评估动作
            next_q_values = self.model(next_states)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q_values_target = self.target_model(next_states).gather(1, next_actions)
            
            # 计算目标值 (r + gamma * max Q(s', a'))
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values_target
        
        # 计算损失
        td_errors = torch.abs(q_values - target_q_values)
        loss = (td_errors * weights).mean()
        
        # 更新优先级 (如果使用优先经验回放)
        if self.prioritized_replay:
            priorities = td_errors.detach().cpu().numpy() + 1e-6  # 添加小值防止优先级为0
            self.memory.update_priorities(indices, priorities)
        
        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_model(self):
        """
        更新目标网络
        """
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save_model(self, path):
        """
        保存模型
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
    
    def load_model(self, path):
        """
        加载模型
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']