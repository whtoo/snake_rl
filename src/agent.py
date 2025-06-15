import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if isinstance(self.memory, PrioritizedReplayBuffer):
            # 优先经验回放返回7个值
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            # 普通经验回放返回5个值
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            indices = None
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
        if isinstance(self.memory, PrioritizedReplayBuffer) and indices is not None:
            priorities = td_errors.detach().cpu().numpy().flatten() + 1e-6  # 添加小值防止优先级为0
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


class NStepBuffer:
    """
    N步学习缓冲区
    存储n步序列并计算n步回报
    """
    def __init__(self, n_step=3, gamma=0.99):
        """
        初始化N步缓冲区
        
        参数:
            n_step: n步学习的步数
            gamma: 折扣因子
        """
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
    
    def add(self, state, action, reward, next_state, done):
        """
        添加一步经验到缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
            
        返回:
            如果缓冲区满了，返回n步经验；否则返回None
        """
        # Store the experience as a tuple: (state, action, reward, next_state, done)
        current_experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(current_experience)
        
        if len(self.buffer) < self.n_step:
            return None
        
        # Calculate n-step reward and identify the n-step next_state and done
        n_step_reward = 0.0
        n_step_next_state = None
        n_step_done = False
        
        for i in range(self.n_step):
            exp = self.buffer[i]
            n_step_reward += (self.gamma ** i) * exp.reward

            if exp.done:
                n_step_next_state = exp.next_state
                n_step_done = True
                # This experience is the one that caused termination within the n-step window.
                # The n-step transition will be from buffer[0] up to this experience.
                # The remaining items in the buffer after this point are not part of this specific n-step transition's reward sum.
                # The effective number of steps for this transition is i + 1.
                # The 'next_state' for the n-step transition is the 'next_state' of this terminating experience.
                break  # Stop accumulating reward

            # If no early termination, the n-step_next_state and n_step_done come from the last element in the window
            if i == self.n_step - 1:
                n_step_next_state = exp.next_state
                n_step_done = exp.done

        # The state and action are from the first element of the n-step sequence (which is now self.buffer[0] due to deque's nature)
        s0, a0, _, _, _ = self.buffer[0]
        
        # The returned transition is (state_0, action_0, n_step_reward, n_step_next_state_n, n_step_done_n)
        # The deque will automatically discard self.buffer[0] on the next append if maxlen is reached.
        # The returned transition is based on the oldest `n_step` experiences currently in the buffer.
        return Experience(s0, a0, n_step_reward, n_step_next_state, n_step_done)
    
    def get_last_n_step(self):
        """
        当游戏结束时，处理缓冲区中剩余的经验。
        为缓冲区中剩余的每个状态，构造一个n-step（或更短）的转换。
        """
        # This method is called when an episode ends (done=True for the last experience added).
        # The main buffer (e.g. ReplayBuffer) will receive these transitions.
        # The NStepBuffer itself will be reset after this.

        # Example: n_step = 3, buffer has [T0, T1] when episode ends.
        # T0 = (s0, a0, r0, s1, d0=false)
        # T1 = (s1, a1, r1, s2, d1=true)
        #
        # We need to form transitions for T0.
        # For T0:
        #   Reward: r0 + gamma*r1
        #   Next State: s2 (from T1.next_state)
        #   Done: True (from T1.done)
        # Result: (s0, a0, r0 + gamma*r1, s2, True)

        # If the buffer is not empty after an episode ends, these are partial n-step transitions.
        # The `add` method already processed and returned full n-step transitions.
        # This method processes what's left, which will be less than n_step items usually.
        # Or, if an episode ends exactly when buffer becomes full, `add` returns the full one,
        # and this method would be called with a full buffer too, but `RainbowAgent` calls `reset`
        # after `get_last_n_step`, so we need to make sure this doesn't double-add.
        # The `RainbowAgent` calls `add`, then if `done`, calls `get_last_n_step`, then `reset`.
        # So, `get_last_n_step` should process all items currently in `self.buffer`.

        experiences_to_return = []

        # Iterate through the experiences currently in the buffer.
        # Each experience self.buffer[j] will be the *start* of an n-step transition.
        for j in range(len(self.buffer)):
            n_step_reward = 0.0
            n_step_next_state = None
            n_step_done = False
            
            # Calculate reward and find next_state/done for the transition starting at self.buffer[j]
            for i in range(self.n_step):
                current_idx_in_buffer = j + i
                if current_idx_in_buffer >= len(self.buffer):
                    # We've run out of experiences in the buffer for this n-step calculation
                    # This means the n_step_next_state and n_step_done are from the last actual experience in the buffer
                    if i > 0: # Make sure we have at least one step
                        n_step_next_state = self.buffer[-1].next_state
                        n_step_done = self.buffer[-1].done
                    else: # Should not happen if len(self.buffer) > 0
                        pass
                    break

                exp = self.buffer[current_idx_in_buffer]
                n_step_reward += (self.gamma ** i) * exp.reward

                if exp.done:
                    n_step_next_state = exp.next_state
                    n_step_done = True
                    break # Stop accumulating reward, this is the end of the episode

                # If loop finishes without early break, next_state/done are from the last considered experience
                if i == self.n_step - 1 or current_idx_in_buffer == len(self.buffer) - 1:
                    n_step_next_state = exp.next_state
                    n_step_done = exp.done

            start_exp = self.buffer[j]
            experiences_to_return.append(Experience(start_exp.state, start_exp.action, n_step_reward, n_step_next_state, n_step_done))
            
        return experiences_to_return
    
    def reset(self):
        """
        重置缓冲区
        """
        self.buffer.clear()


class RainbowAgent(DQNAgent):
    """
    Rainbow DQN智能体
    集成了所有Rainbow组件：Double DQN, Dueling DQN, Prioritized Replay,
    Multi-step Learning, Noisy Networks, Distributional DQN
    """
    def __init__(self, model, target_model, env, device,
                 n_step=3, use_noisy=True, use_distributional=False,
                 n_atoms=51, v_min=-10, v_max=10, **kwargs):
        """
        初始化Rainbow智能体
        
        参数:
            model: 主Q网络 (RainbowDQN)
            target_model: 目标Q网络
            env: 游戏环境
            device: 计算设备
            n_step: n步学习的步数
            use_noisy: 是否使用噪声网络
            use_distributional: 是否使用分布式Q学习
            n_atoms: 分布式Q学习的原子数量
            v_min: 值函数的最小值
            v_max: 值函数的最大值
            **kwargs: 其他DQN参数
        """
        # 如果使用噪声网络，禁用epsilon探索
        if use_noisy:
            kwargs['epsilon_start'] = 0.0
            kwargs['epsilon_final'] = 0.0
            kwargs['epsilon_decay'] = 1
        
        # 默认使用优先经验回放
        if 'prioritized_replay' not in kwargs:
            kwargs['prioritized_replay'] = True
            
        super().__init__(model, target_model, env, device, **kwargs)
        
        self.n_step = n_step
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # 初始化n步缓冲区
        self.n_step_buffer = NStepBuffer(n_step, self.gamma)
        
        # 分布式Q学习相关
        if use_distributional:
            self.delta_z = (v_max - v_min) / (n_atoms - 1)
            self.support = torch.linspace(v_min, v_max, n_atoms).to(device)
    
    def select_action(self, state, evaluate=False):
        """
        选择动作
        """
        # 如果使用噪声网络，直接使用贪婪策略
        if self.use_noisy:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                if self.use_distributional:
                    # 分布式Q学习：计算期望Q值
                    # self.model(state_tensor) already returns softmaxed probabilities for RainbowDQN
                    dist = self.model(state_tensor)  # [batch, actions, atoms]
                    q_values = (dist * self.support).sum(2)  # [batch, actions]
                else:
                    q_values = self.model(state_tensor)
                
                return q_values.max(1)[1].item()
        else:
            # 使用父类的epsilon-greedy策略
            return super().select_action(state, evaluate)
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        存储经验到n步缓冲区和回放缓冲区
        """
        # 添加到n步缓冲区
        n_step_exp = self.n_step_buffer.add(state, action, reward, next_state, done)
        
        # 如果n步缓冲区满了，将n步经验存入回放缓冲区
        if n_step_exp is not None:
            n_state, n_action, n_reward, n_next_state, n_done = n_step_exp
            self.memory.push(n_state, n_action, n_reward, n_next_state, n_done)
        
        # 如果游戏结束，处理缓冲区中剩余的经验
        if done:
            remaining_exps = self.n_step_buffer.get_last_n_step()
            for exp in remaining_exps:
                exp_state, exp_action, exp_reward, exp_next_state, exp_done = exp
                self.memory.push(exp_state, exp_action, exp_reward, exp_next_state, exp_done)
            self.n_step_buffer.reset()
    
    def update_model(self):
        """
        更新模型参数
        """
        # 如果经验不足，不进行更新
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 重置噪声网络的噪声
        if self.use_noisy:
            # Assuming RainbowDQN model has sample_noise method as per model.py
            if hasattr(self.model, 'sample_noise'):
                self.model.sample_noise()
            if hasattr(self.target_model, 'sample_noise'):
                self.target_model.sample_noise()
        
        # 从经验回放缓冲区采样
        if isinstance(self.memory, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            indices = None
            weights = torch.ones_like(rewards).to(self.device)
        
        # 将数据转移到设备
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        if self.use_distributional:
            loss = self._compute_distributional_loss(states, actions, rewards, next_states, dones, weights)
        else:
            loss = self._compute_standard_loss(states, actions, rewards, next_states, dones, weights)
        
        # 更新优先级
        if isinstance(self.memory, PrioritizedReplayBuffer) and indices is not None:
            # 为了计算TD误差，我们需要重新计算一下
            with torch.no_grad():
                if self.use_distributional:
                    # self.model(states) already returns softmaxed probabilities
                    current_dist_probs = self.model(states)
                    current_q = (current_dist_probs * self.support).sum(2)
                    current_q_selected = current_q.gather(1, actions)
                    
                    # self.target_model(next_states) already returns softmaxed probabilities
                    next_dist_probs_target = self.target_model(next_states)
                    # For PER TD error, we usually use the main network for action selection (Double DQN style)
                    # but for consistency with how loss target is calculated, let's use target Q for TD error target.
                    # Actually, the loss calculation for distributional uses model for next_actions and target_model for next_dist.
                    # To be consistent with the actual loss's target, we should replicate that logic.

                    # Replicating Double DQN logic for next actions from _compute_distributional_loss
                    next_dist_model = self.model(next_states) # Probs from model
                    next_q_model = (next_dist_model * self.support).sum(2)
                    next_actions_for_td = next_q_model.max(1)[1].unsqueeze(1) # Use model for action selection
                    
                    # Get next distribution from target network using these actions
                    next_dist_target_net_for_td = self.target_model(next_states) # Probs from target_model
                    # Gather the distributions for the best actions selected by the main model
                    next_best_dist_target_net = next_dist_target_net_for_td.gather(1, next_actions_for_td.unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)

                    next_q_max = (next_best_dist_target_net * self.support.unsqueeze(0)).sum(1).unsqueeze(1)

                    target_q = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_max
                    td_errors = torch.abs(current_q_selected - target_q)
                else:
                    current_q = self.model(states).gather(1, actions)
                    next_q_values = self.model(next_states)
                    next_actions = next_q_values.max(1)[1].unsqueeze(1)
                    next_q_target = self.target_model(next_states).gather(1, next_actions)
                    target_q = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_target
                    td_errors = torch.abs(current_q - target_q)
                
                priorities = td_errors.detach().cpu().numpy().flatten() + 1e-6
                self.memory.update_priorities(indices, priorities)
        
        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def _compute_standard_loss(self, states, actions, rewards, next_states, dones, weights):
        """
        计算标准DQN损失
        """
        # 计算当前Q值
        q_values = self.model(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            # 双DQN
            next_q_values = self.model(next_states)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q_values_target = self.target_model(next_states).gather(1, next_actions)
            
            # 使用n步折扣
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values_target
        
        # 计算损失
        td_errors = torch.abs(q_values - target_q_values)
        loss = (td_errors * weights).mean()
        
        return loss
    
    def _compute_distributional_loss(self, states, actions, rewards, next_states, dones, weights):
        """
        计算分布式Q学习损失 (C51算法)
        """
        batch_size = states.size(0)
        
        # 计算当前分布
        # self.model(states) returns probabilities from RainbowDQN's forward pass
        current_dist_probs = self.model(states)
        # Gather the probabilities for the taken actions
        current_probs_selected = current_dist_probs.gather(1, actions.unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)
        # Convert to log probabilities for KL divergence calculation
        current_log_probs_selected = current_probs_selected.log() # log(softmax(logits))
        
        # 计算目标分布
        with torch.no_grad():
            # 使用主网络选择动作 (Double DQN)
            # self.model(next_states) returns probabilities
            next_dist_probs_model = self.model(next_states)
            next_q_values_model = (next_dist_probs_model * self.support.unsqueeze(0)).sum(2)
            next_actions = next_q_values_model.max(1)[1]
            
            # 使用目标网络评估动作 for chosen next_actions
            # self.target_model(next_states) returns probabilities
            next_dist_probs_target_net = self.target_model(next_states)
            next_dist_target = next_dist_probs_target_net.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)
            
            # 计算目标支撑
            rewards = rewards.expand(-1, self.n_atoms)
            dones = dones.expand(-1, self.n_atoms)
            support = self.support.expand(batch_size, -1)
            
            target_support = rewards + (1 - dones) * (self.gamma ** self.n_step) * support
            target_support = target_support.clamp(self.v_min, self.v_max)
            
            # 分布投影
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Correct projection for cases where l == u (i.e., b is an integer)
            # This ensures probability mass is not lost when Tz falls exactly on an atom.
            # Apply corrections based on Kaixhin/Rainbow's approach:
            # If l == u, it means Tz landed exactly on an atom.
            # To ensure the distribution logic works with l and u,
            # we can shift l down by 1 or u up by 1,
            # provided they don't go out of bounds [0, n_atoms-1].
            
            # Only adjust l if it's equal to u AND u is not already 0 (so l can be moved to l-1)
            l_eq_u_and_u_gt_0 = (l == u) & (u > 0)
            l[l_eq_u_and_u_gt_0] -= 1

            # Only adjust u if it's equal to l AND l is not already n_atoms-1 (so u can be moved to u+1)
            # Note: the original l (before adjustment above) should be used for u's condition.
            # However, if l was adjusted, l!=u now. So, this needs careful thought.
            # Let's use the common formulation:
            # l_orig = b.floor().long() -> used for u adjustment condition
            # u_orig = b.ceil().long() -> used for l adjustment condition
            # This is simpler:
            # For elements where l_j = u_j (because b_j is an integer):
            # If b_j = V_MIN (so l_j=u_j=0), the prob goes to atom 0. (u_j - b_j)=0, (b_j - l_j)=0. Error.
            #   We need m[0] += p_j. This means (b_j-l_j) should be 1 for u_j, or (u_j-b_j) for l_j.
            #   If l_j=0, u_j=0. If we make u_j=1, then (u_j-b_j)=1 for l_j=0. (b_j-l_j)=0 for u_j=1. Correct.
            # If b_j = V_MAX (so l_j=u_j=N-1), the prob goes to atom N-1.
            #   If l_j=N-1, u_j=N-1. If we make l_j=N-2, then (u_j-b_j)=0 for l_j=N-2. (b_j-l_j)=1 for u_j=N-1. Correct.

            # Simpler fix for l=u (from reference):
            # if l=u, then prob * (u-b) = 0 and prob * (b-l) = 0.
            # The mass is assigned to l if u-b=1 and b-l=0, or to u if u-b=0, b-l=1.
            # When l=u=b (integer), we want all mass at index b.
            # One way: if l==u, m[l] += prob. This can be done by setting (b-l) to 1 and (u-b) to 0 for index u,
            # or vice-versa for index l.
            # The logic `l[(u > 0) & (l == u)] -= 1` and `u[(l < (self.n_atoms - 1)) & (l == u)] += 1` (using original l for u's condition) handles this.

            # Let's use a mask for l==u elements.
            eq_mask = (l == u)
            ne_mask = ~eq_mask

            target_dist = torch.zeros_like(next_dist_target, device=self.device) # Shape (batch_size, n_atoms)

            # Calculate contributions for non-equal l and u
            m_l_contrib = next_dist_target * (u.float() - b) # Shape (batch_size, n_atoms)
            m_u_contrib = next_dist_target * (b - l.float()) # Shape (batch_size, n_atoms)

            # Zero out contributions where l == u (i.e., where ne_mask is False)
            m_l_contrib_ne = torch.where(ne_mask, m_l_contrib, torch.zeros_like(m_l_contrib))
            m_u_contrib_ne = torch.where(ne_mask, m_u_contrib, torch.zeros_like(m_u_contrib))

            target_dist.scatter_add_(1, l, m_l_contrib_ne)
            target_dist.scatter_add_(1, u, m_u_contrib_ne)

            # Handle equal l and u (b is integer): add full probability to that atom
            if eq_mask.any():
                # Create a source tensor that only has values from next_dist_target where eq_mask is true
                src_eq = torch.where(eq_mask, next_dist_target, torch.zeros_like(next_dist_target))
                # l (or u) where eq_mask is true gives the atom index.
                # scatter_add_ will sum if multiple source elements map to the same target index,
                # but here, for each batch item, l[eq_mask] should point to a unique atom for that batch item's sum.
                # This is fine as we are adding the specific probability next_dist_target[eq_mask_indices] to target_dist[eq_mask_indices, l[eq_mask_indices]].
                # The previous scatter_adds for m_l_contrib_ne and m_u_contrib_ne would have added 0 at these eq_mask locations.
                target_dist.scatter_add_(1, l, src_eq) # Use full l, src_eq has zeros where eq_mask is false.

        # 计算KL散度损失
        # current_dist here should be log probabilities of selected actions
        loss = -(target_dist * current_log_probs_selected).sum(1)
        loss = (loss * weights.squeeze()).mean()
        
        return loss