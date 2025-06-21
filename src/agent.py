"""
Defines reinforcement learning agents and the core Experience namedtuple.

This module provides:
- Experience: A namedtuple for storing (state, action, reward, next_state, done) transitions.
- DQNAgent: A base Deep Q-Network agent implementing core DQN logic,
  including epsilon-greedy exploration, model updates, and target network updates.
  It can use either a standard or a prioritized replay buffer.
- RainbowAgent: An advanced DQN agent that inherits from DQNAgent and integrates
  several improvements:
    - Prioritized Experience Replay (via DQNAgent)
    - N-step Learning (using AdaptiveNStepBuffer)
    - Noisy Networks for exploration
    - Distributional Q-learning
    - (Potentially) Experience Augmentation
"""
import numpy as np
import torch
import torch.optim as optim
import time # Ensure time is imported
import random
from collections import namedtuple # deque removed
from .utils import ExperienceAugmenter # Import ExperienceAugmenter

# Import moved buffer classes
from .buffers.replay_buffers import SumTree, ReplayBuffer, PrioritizedReplayBuffer
from .buffers.n_step_buffers import NStepBuffer, AdaptiveNStepBuffer

# Experience namedtuple moved to src.utils

# SumTree class removed, now imported.

# ReplayBuffer class removed, now imported.

# PrioritizedReplayBuffer class removed, now imported.

class DQNAgent:
    """
    DQN智能体
    """
    def __init__(self, model, target_model, env, device,
                 buffer_size=100000, batch_size=32, gamma=0.99,
                 lr=1e-4, epsilon_start=1.0, epsilon_final=0.01,
                 epsilon_decay=10000, target_update=1000,
                 prioritized_replay=False, huber_delta=1.0, grad_clip_norm=1.0):
        
        # 存储优化参数
        self.huber_delta = huber_delta
        self.grad_clip_norm = grad_clip_norm
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update

        # DIAGNOSTIC: Force standard ReplayBuffer to disable PER
        self.memory = ReplayBuffer(buffer_size)
        self.prioritized_replay = False
        # if prioritized_replay:
        #     # For PrioritizedReplayBuffer, alpha, beta_start, beta_frames are class variables or handled internally
        #     self.memory = PrioritizedReplayBuffer(buffer_size)
        #     self.prioritized_replay = True
        # else:
        #     self.memory = ReplayBuffer(buffer_size)
        #     self.prioritized_replay = False

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0 # Tracks total environment interactions via select_action

    def _prepare_batch_for_update(self):
        """Samples a batch from memory and prepares it for model update.

        Handles PER sampling (calculates weights) and standard buffer sampling.
        Moves all tensors to the agent's device.

        Returns:
            tuple: (states, actions, rewards, next_states, dones, weights, update_indices)
                   or None if memory has insufficient samples.
        """
        if len(self.memory) < self.batch_size:
            # print(f"[MEM_TRACE] DQNAgent._prepare_batch_for_update: Not enough samples in memory ({len(self.memory)} < {self.batch_size})")
            return None # Indicates not enough samples

        print(f"[MEM_TRACE] DQNAgent._prepare_batch_for_update: Before self.memory.sample()")
        if isinstance(self.memory, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, tree_indices, weights_tensor = self.memory.sample(self.batch_size)
            print(f"[MEM_TRACE] DQNAgent._prepare_batch_for_update: After PER self.memory.sample()")
            weights = weights_tensor.to(self.device)
            update_indices = tree_indices
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            print(f"[MEM_TRACE] DQNAgent._prepare_batch_for_update: After standard self.memory.sample()")
            update_indices = None
            weights = torch.ones_like(rewards).to(self.device) # Ensure weights is defined and on device


        print(f"[MEM_TRACE] DQNAgent._prepare_batch_for_update: Before moving batch to device: {self.device}")
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        print(f"[MEM_TRACE] DQNAgent._prepare_batch_for_update: After moving batch to device")

        return states, actions, rewards, next_states, dones, weights, update_indices

    def select_action(self, state, evaluate=False):
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                 np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if evaluate or random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.model(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.env.action_space.n)

    def update_model(self): # Signature reverted
        print(f"[MEM_TRACE] DQNAgent.update_model: Start")
        prepared_batch = self._prepare_batch_for_update()
        if prepared_batch is None:
            print(f"[MEM_TRACE] DQNAgent.update_model: prepared_batch is None, returning 0.0")
            return 0.0
        states, actions, rewards, next_states, dones, weights, update_indices = prepared_batch

        q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_q_values_model = self.model(next_states)
            next_actions = next_q_values_model.max(1)[1].unsqueeze(1)
            next_q_values_target = self.target_model(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values_target

        td_errors = torch.abs(q_values - target_q_values)
        loss = (td_errors * weights).mean() # td_errors needs to be [batch_size, 1] like weights. Weights are now ones.

        # DIAGNOSTIC: PER is disabled, so no priority updates
        # if isinstance(self.memory, PrioritizedReplayBuffer) and update_indices is not None:
        #     # Pass raw TD errors (detached from graph, on CPU)
        #     td_errors_numpy = td_errors.detach().cpu().numpy().flatten()
        #     self.memory.update_priorities(update_indices, td_errors_numpy)

        # print(f"[MEM_TRACE] DQNAgent.update_model: Loss calculated: {loss.item()}, before optimizer step")

        self.optimizer.zero_grad()
        loss.backward()
        # 使用更严格的梯度裁剪以提高训练稳定性
        grad_clip_norm = getattr(self, 'grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
        self.optimizer.step()

        # print(f"[MEM_TRACE] DQNAgent.update_model: End, loss={loss.item() if hasattr(loss, 'item') else loss}")
        return loss.item()

    def update_target_model(self):
        # print(f"[MEM_TRACE] DQNAgent.update_target_model: Updating target model")
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, weights_only=False)
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(f"⚠️ 模型结构不完全匹配，使用兼容模式加载: {e}")
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                raise e
        try:
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(f"⚠️ 目标模型结构不完全匹配，使用兼容模式加载")
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'], strict=False)
            else:
                raise e
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"⚠️ 优化器状态加载失败，将使用默认状态: {e}")
        if 'steps_done' in checkpoint:
            self.steps_done = checkpoint['steps_done']
        else:
            print("⚠️ 检查点中未找到steps_done，使用默认值0")
            self.steps_done = 0

# NStepBuffer class removed, now imported.

# AdaptiveNStepBuffer class removed, now imported.

class RainbowAgent(DQNAgent):
    """
    Rainbow DQN智能体
    集成了所有Rainbow组件：Double DQN, Dueling DQN, Prioritized Replay,
    Multi-step Learning, Noisy Networks, Distributional DQN
    """
    def __init__(self, model, target_model, env, device,
                 base_n_step=3, max_n_step=10,       # Renamed n_step to base_n_step, added max_n_step
                 adapt_n_step_freq=1000,      # Frequency to call adapt_n_step
                 td_error_threshold_low=0.1,  # Thresholds for N-step adaptation
                 td_error_threshold_high=0.5,
                 augmentation_config=None,    # For Experience Augmentation
                 use_noisy=True, use_distributional=False,
                 n_atoms=51, v_min=-10, v_max=10, 
                 huber_delta=1.0, grad_clip_norm=1.0, **kwargs): # 添加优化参数
        if use_noisy:
            kwargs['epsilon_start'] = 0.0
            kwargs['epsilon_final'] = 0.0
            kwargs['epsilon_decay'] = 1
        if 'prioritized_replay' not in kwargs:
            kwargs['prioritized_replay'] = True

        # Remove Rainbow-specific arguments from kwargs if they were passed,
        # especially the legacy 'n_step' which causes TypeError in DQNAgent.
        # Named parameters in RainbowAgent's signature (like base_n_step, use_noisy)
        # are handled by Python and won't be in kwargs if passed correctly.
        # This primarily handles unexpected/legacy params passed via **kwargs.
        kwargs.pop('n_step', None)
        kwargs.pop('base_n_step', None) # Also remove if passed via kwargs instead of named param
        kwargs.pop('max_n_step', None)  # Also remove if passed via kwargs
        kwargs.pop('adapt_n_step_freq', None)
        kwargs.pop('td_error_threshold_low', None)
        kwargs.pop('td_error_threshold_high', None)
        kwargs.pop('augmentation_config', None)
        # use_noisy, use_distributional, n_atoms, v_min, v_max are named params,
        # but if they were also in kwargs, good to remove.
        # However, use_noisy modifies other kwargs, so care is needed.
        # The main offender is 'n_step'. For others, relying on them being named params.
        # If use_noisy was in kwargs, it would be an issue for the logic below.
        # For now, just ensuring 'n_step' is gone is the primary fix for the TypeError.


        super().__init__(model, target_model, env, device, **kwargs)

        # 存储优化参数
        self.huber_delta = huber_delta
        self.grad_clip_norm = grad_clip_norm
        
        # self.n_step (from parent DQNAgent or original kwargs) is not used directly by AdaptiveNStepBuffer
        # base_n_step is now the explicit parameter for the initial/minimum N.
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.n_step_buffer = AdaptiveNStepBuffer(
            base_n_step=base_n_step,
            max_n_step=max_n_step,
            gamma=self.gamma, # DQNAgent sets self.gamma
            td_error_threshold_low=td_error_threshold_low,
            td_error_threshold_high=td_error_threshold_high
            # n_step_increment and n_step_decrement use defaults in AdaptiveNStepBuffer
        )
        self.adapt_n_step_freq = adapt_n_step_freq
        self._rainbow_training_updates_count = 0 # Retained for adapt_n_step_freq logic

        if augmentation_config:
            self.augmenter = ExperienceAugmenter(augmentation_config)
            print("Experience Augmentation enabled.")
        else:
            self.augmenter = None

        if use_distributional:
            self.delta_z = (v_max - v_min) / (n_atoms - 1)
            self.support = torch.linspace(v_min, v_max, n_atoms).to(device)

    def select_action(self, state, evaluate=False):
        if self.use_noisy:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                if self.use_distributional:
                    dist = self.model(state_tensor)
                    q_values = (dist * self.support).sum(2)
                else:
                    q_values = self.model(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return super().select_action(state, evaluate)

    def store_experience(self, state, action, reward, next_state, done):
        # Augment experience if augmenter is enabled
        current_state, current_action, current_reward, current_next_state, current_done = state, action, reward, next_state, done
        if self.augmenter:
            # Assuming augment returns the full tuple, but we only use augmented states for now
            # And we are careful not to modify original action, reward, done for this specific stored experience
            aug_s, _, _, aug_ns, _ = self.augmenter.augment(state, action, reward, next_state, done)
            current_state, current_next_state = aug_s, aug_ns
            # print("DEBUG: Augmented state stored.") # For debugging

        # Ensure states are NumPy arrays on CPU before storing in buffers
        if isinstance(current_state, torch.Tensor):
            current_state = current_state.cpu().numpy()
        elif not isinstance(current_state, np.ndarray): # Ensure it's an ndarray if not a tensor
            current_state = np.asarray(current_state)

        if current_next_state is not None: # next_state can be None
            if isinstance(current_next_state, torch.Tensor):
                current_next_state = current_next_state.cpu().numpy()
            elif not isinstance(current_next_state, np.ndarray): # Ensure it's an ndarray if not a tensor
                current_next_state = np.asarray(current_next_state)

        # Diagnostic print for state dtypes and ensure uint8
        # print(f"Storing state dtype: {current_state.dtype}, next_state dtype: {current_next_state.dtype if current_next_state is not None else 'None'}")
        if current_state.dtype != np.uint8:
            # print(f"Converting state from {current_state.dtype} to uint8")
            current_state = current_state.astype(np.uint8)
        if current_next_state is not None and current_next_state.dtype != np.uint8:
            # print(f"Converting next_state from {current_next_state.dtype} to uint8")
            current_next_state = current_next_state.astype(np.uint8)

        n_step_exp = self.n_step_buffer.add(current_state, current_action, current_reward, current_next_state, current_done)
        if n_step_exp is not None:
            self.memory.push(n_step_exp.state, n_step_exp.action, n_step_exp.reward, n_step_exp.next_state, n_step_exp.done)
        if done:
            remaining_exps = self.n_step_buffer.get_last_n_step()
            for exp in remaining_exps:
                self.memory.push(exp.state, exp.action, exp.reward, exp.next_state, exp.done)
            self.n_step_buffer.reset()

    def update_model(self): # Signature reverted
        prepared_batch = super()._prepare_batch_for_update()
        if prepared_batch is None:
            return 0.0
        states, actions, rewards, next_states, dones, weights, update_indices = prepared_batch

        if self.use_noisy:
            if hasattr(self.model, 'sample_noise'): self.model.sample_noise()
            if hasattr(self.target_model, 'sample_noise'): self.target_model.sample_noise()

        # Loss calculation uses self.n_step_buffer.current_n_step for discounting Q(s',a')
        if self.use_distributional:
            loss = self._compute_distributional_loss(states, actions, rewards, next_states, dones, weights)
        else:
            loss = self._compute_standard_loss(states, actions, rewards, next_states, dones, weights)

        # Calculate TD errors for PER update and N-step adaptation
        # print(f"[MEM_TRACE] RainbowAgent.update_model: Before _calculate_n_step_td_errors")
        td_errors_numpy = self._calculate_n_step_td_errors(states, actions, rewards, next_states, dones)
        # print(f"[MEM_TRACE] RainbowAgent.update_model: After _calculate_n_step_td_errors, td_errors_numpy shape: {td_errors_numpy.shape if td_errors_numpy is not None else 'None'}")

        # Update PER priorities if applicable
        if isinstance(self.memory, PrioritizedReplayBuffer) and update_indices is not None and td_errors_numpy is not None:
            self.memory.update_priorities(update_indices, td_errors_numpy)

        # Record TD errors for N-step adaptation
        if td_errors_numpy is not None: # Ensure td_errors were computed
            # print(f"[MEM_TRACE] RainbowAgent.update_model: Before n_step_buffer.record_td_error")
            self.n_step_buffer.record_td_error(np.mean(td_errors_numpy))
            # print(f"[MEM_TRACE] RainbowAgent.update_model: After n_step_buffer.record_td_error")

        # print(f"[MEM_TRACE] RainbowAgent.update_model: Loss calculated: {loss.item()}, before optimizer step (Rainbow override)") # Matches DQNAgent log point
        self.optimizer.zero_grad()
        loss.backward()
        # 使用更严格的梯度裁剪以提高训练稳定性
        grad_clip_norm = getattr(self, 'grad_clip_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
        self.optimizer.step()

        self._rainbow_training_updates_count += 1 # Increment Rainbow specific counter

        if self._rainbow_training_updates_count > 0 and self._rainbow_training_updates_count % self.adapt_n_step_freq == 0: # Use the same counter for n-step adaptation frequency
            # print(f"[MEM_TRACE] RainbowAgent.update_model: Before n_step_buffer.adapt_n_step(), updates_count: {self._rainbow_training_updates_count}")
            self.n_step_buffer.adapt_n_step()
            # print(f"[MEM_TRACE] RainbowAgent.update_model: After n_step_buffer.adapt_n_step()")

        # print(f"[MEM_TRACE] RainbowAgent.update_model: End, loss={loss.item()}") # Matches DQNAgent log point
        return loss.item()

    def _calculate_n_step_td_errors(self, states, actions, rewards, next_states, dones):
        """Calculates N-step TD errors for experiences after a model update.

        This is used for updating PrioritizedReplayBuffer priorities and for
        AdaptiveNStepBuffer to adjust N. The rewards here are N-step rewards,
        and gamma is applied for self.n_step_buffer.current_n_step.

        Args:
            states: Batch of current states.
            actions: Batch of actions taken.
            rewards: Batch of N-step rewards received.
            next_states: Batch of N-step next states.
            dones: Batch of N-step done flags.

        Returns:
            np.ndarray: Flattened array of absolute TD errors.
        """
        with torch.no_grad(): # All calculations for TD error should not affect gradients
            if self.use_distributional:
                # For distributional, TD error is often |E[Q(s,a)] - (r + gamma^N * E[Q(s',a')])|
                # Re-calculate current Qs expected value for selected actions
                current_dist_probs_model = self.model(states)
                current_q_selected_dist_probs = current_dist_probs_model.gather(1, actions.unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)
                current_q_selected_expected = (current_q_selected_dist_probs * self.support.unsqueeze(0)).sum(1, keepdim=True)

                # Re-calculate target Qs expected value (Double DQN style for action selection)
                next_dist_probs_model = self.model(next_states)
                next_q_values_model = (next_dist_probs_model * self.support.unsqueeze(0)).sum(2)
                next_actions = next_q_values_model.max(1)[1].unsqueeze(1)

                next_dist_probs_target_net = self.target_model(next_states)
                next_best_dist_target_net = next_dist_probs_target_net.gather(1, next_actions.unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)
                next_q_max_expected = (next_best_dist_target_net * self.support.unsqueeze(0)).sum(1, keepdim=True)

                target_q_for_td_error = rewards + (1 - dones) * (self.gamma ** self.n_step_buffer.current_n_step) * next_q_max_expected
                td_errors_tensor = torch.abs(current_q_selected_expected - target_q_for_td_error)
            else: # Standard DQN
                current_q_model_vals = self.model(states).gather(1, actions)

                next_q_values_model = self.model(next_states)
                next_actions = next_q_values_model.max(1)[1].unsqueeze(1)
                next_q_target_net = self.target_model(next_states).gather(1, next_actions)

                target_q_for_td_error = rewards + (1 - dones) * (self.gamma ** self.n_step_buffer.current_n_step) * next_q_target_net
                td_errors_tensor = torch.abs(current_q_model_vals - target_q_for_td_error)

            return td_errors_tensor.detach().cpu().numpy().flatten()

    def _compute_standard_loss(self, states, actions, rewards, next_states, dones, weights):
        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values_model = self.model(next_states)
            next_actions = next_q_values_model.max(1)[1].unsqueeze(1)
            next_q_values_target = self.target_model(next_states).gather(1, next_actions)
            # Use current_n_step from adaptive buffer for discounting future rewards
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step_buffer.current_n_step) * next_q_values_target
        
        td_errors = q_values - target_q_values
        
        # 使用 Huber Loss 替代 MAE，对异常值更鲁棒
        huber_delta = getattr(self, 'huber_delta', 1.0)
        abs_td_errors = torch.abs(td_errors)
        
        # Huber Loss: 当误差小于delta时使用平方损失，大于delta时使用线性损失
        huber_loss = torch.where(
            abs_td_errors <= huber_delta,
            0.5 * td_errors.pow(2),
            huber_delta * (abs_td_errors - 0.5 * huber_delta)
        )
        
        loss = (huber_loss * weights).mean()
        return loss

    def _compute_distributional_loss(self, states, actions, rewards, next_states, dones, weights):
        batch_size = states.size(0)
        current_dist_probs = self.model(states)
        current_probs_selected = current_dist_probs.gather(1, actions.unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)
        current_log_probs_selected = current_probs_selected.log()

        with torch.no_grad():
            next_dist_probs_model = self.model(next_states)
            next_q_values_model = (next_dist_probs_model * self.support.unsqueeze(0)).sum(2)
            next_actions = next_q_values_model.max(1)[1]

            next_dist_probs_target_net = self.target_model(next_states)
            next_dist_target = next_dist_probs_target_net.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)

            rewards_exp = rewards.expand(-1, self.n_atoms)
            dones_exp = dones.expand(-1, self.n_atoms)
            support_exp = self.support.expand(batch_size, -1)

            # Use current_n_step from adaptive buffer for discounting future rewards
            target_support = rewards_exp + (1 - dones_exp) * (self.gamma ** self.n_step_buffer.current_n_step) * support_exp
            target_support = target_support.clamp(self.v_min, self.v_max)

            b = (target_support - self.v_min) / self.delta_z
            lower_indices = b.floor().long()
            upper_indices = b.ceil().long()

            eq_mask = (lower_indices == upper_indices)
            ne_mask = ~eq_mask

            target_dist = torch.zeros_like(next_dist_target, device=self.device)

            m_l_contrib = next_dist_target * (upper_indices.float() - b)
            m_u_contrib = next_dist_target * (b - lower_indices.float())

            m_l_contrib_ne = torch.where(ne_mask, m_l_contrib, torch.zeros_like(m_l_contrib))
            m_u_contrib_ne = torch.where(ne_mask, m_u_contrib, torch.zeros_like(m_u_contrib))

            target_dist.scatter_add_(1, lower_indices.clamp(0, self.n_atoms -1) , m_l_contrib_ne)
            target_dist.scatter_add_(1, upper_indices.clamp(0, self.n_atoms -1), m_u_contrib_ne)

            if eq_mask.any():
                src_eq = torch.where(eq_mask, next_dist_target, torch.zeros_like(next_dist_target))
                target_dist.scatter_add_(1, lower_indices.clamp(0, self.n_atoms -1), src_eq)

        loss = -(target_dist * current_log_probs_selected).sum(1)
        loss = (loss * weights.squeeze()).mean()
        return loss