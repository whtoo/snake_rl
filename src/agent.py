import numpy as np
import torch
import torch.optim as optim
import random
from collections import deque, namedtuple
from .utils import ExperienceAugmenter # Import ExperienceAugmenter

# 定义经验元组的结构
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class SumTree:
    """
    SumTree data structure for efficient prioritized experience replay.
    Leaf nodes store experience priorities, and internal nodes store the sum of their children's priorities.

    Args:
        capacity (int): The number of leaf nodes, which is the capacity of the replay buffer.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32) # Use float32 for priorities
        self.data_indices = np.zeros(capacity, dtype=int) # Stores the actual index in the main experience buffer for each leaf
        self._max_priority = 1.0 # Tracks the maximum priority for new experiences
        self.data_pointer = 0 # Points to the current leaf index to update (cyclical)

    def _propagate(self, tree_idx, change):
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, tree_idx, s):
        """
        Recursively finds the leaf node corresponding to a cumulative priority value `s`.

        Args:
            tree_idx (int): Current node's index in the tree.
            s (float): Cumulative priority value.

        Returns:
            int: Index of the found leaf node in the tree.
        """
        left = 2 * tree_idx + 1
        right = left + 1
        if left >= len(self.tree): # Reached a leaf node
            return tree_idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self):
        """Returns the total sum of all priorities (value of the root node)."""
        return self.tree[0]

    def add(self, priority, data_buffer_idx):
        """
        Adds a new priority to a leaf node, corresponding to an experience.
        If the buffer is full, it overwrites the oldest entry in terms of SumTree leaf updates.

        Args:
            priority (float): Priority of the new experience.
            data_buffer_idx (int): Index of the actual experience in the main replay buffer.
        """
        tree_idx = self.data_pointer + self.capacity - 1 # Calculate tree index for the current leaf
        self.data_indices[self.data_pointer] = data_buffer_idx # Store mapping from this leaf to the experience's actual buffer index
        
        self.update(tree_idx, priority) # Update the tree with the new priority

        self.data_pointer = (self.data_pointer + 1) % self.capacity # Move pointer to the next leaf to update
        if priority > self._max_priority: # Update overall max priority if this new priority is higher
             self._max_priority = priority

    def update(self, tree_idx, priority):
        """
        Updates the priority of a leaf node and propagates the change up the tree.

        Args:
            tree_idx (int): Tree index of the leaf node to update.
            priority (float): New priority value.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change) # Propagate change to parent nodes

        # Update max_priority if a leaf's priority (and not an internal node's sum) changes to be the new max
        if priority > self._max_priority and tree_idx >= (self.capacity - 1): # tree_idx for leaves start at capacity - 1
             self._max_priority = priority


    def get_leaf(self, s):
        """
        Samples a leaf node based on a cumulative priority value `s`.

        Args:
            s (float): A random value between 0 and `total_priority()`.

        Returns:
            tuple: (leaf_tree_idx, priority_value, data_buffer_idx)
                   - leaf_tree_idx: The tree index of the sampled leaf.
                   - priority_value: The priority of the sampled leaf.
                   - data_buffer_idx: The index of the corresponding experience in the main replay buffer.
        """
        leaf_idx = self._retrieve(0, s)
        data_idx_in_sumtree_leaf_array = leaf_idx - self.capacity + 1 # Convert tree leaf index to index in self.data_indices
        return leaf_idx, self.tree[leaf_idx], self.data_indices[data_idx_in_sumtree_leaf_array]

    @property
    def max_priority(self):
        """Returns the current maximum priority seen in the tree (or 1.0 if tree is empty/all zeros)."""
        return self._max_priority if self._max_priority > 0 else 1.0


class ReplayBuffer:
    """
    经验回放缓冲区
    存储和采样智能体与环境交互的经验
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.cat([torch.FloatTensor(exp.state).unsqueeze(0) for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float).unsqueeze(1)
        next_states = torch.cat([torch.FloatTensor(exp.next_state).unsqueeze(0) for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float).unsqueeze(1)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    epsilon = 1e-5  # Small constant added to priorities to ensure no zero priority.
    alpha = 0.6  # Exponent for converting TD errors to priorities. Controls shape of distribution.
    beta_start = 0.4  # Initial value for beta (importance-sampling exponent). Annealed towards 1.0.
    beta_frames = 100000 # Number of frames over which beta is annealed to 1.0.

    def __init__(self, capacity):
        """
        Initializes the PrioritizedReplayBuffer.

        Args:
            capacity (int): Maximum number of experiences to store.
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.buffer = [None] * capacity # Stores the actual Experience tuples
        self.write_pos = 0 # Points to the next available slot in self.buffer
        self.num_experiences = 0 # Current number of experiences in the buffer
        self.frame = 1 # Counter for frames, used for beta annealing

    def _get_priority(self, td_error):
        """
        Converts a TD error to a priority value.
        Priority = (|TD_error| + epsilon) ^ alpha.

        Args:
            td_error (float or np.ndarray): The TD error(s).

        Returns:
            float or np.ndarray: The calculated priority(ies).
        """
        return (np.abs(td_error) + self.epsilon) ** self.alpha

    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer[self.write_pos] = experience
        
        # New experiences are added with max priority
        current_max_priority = self.tree.max_priority
        self.tree.add(current_max_priority, self.write_pos) # data_buffer_idx is self.write_pos
        
        self.write_pos = (self.write_pos + 1) % self.capacity
        if self.num_experiences < self.capacity:
            self.num_experiences += 1
    
    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer using stratified sampling.
        Calculates and returns importance sampling (IS) weights for each sampled experience.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing:
                - states (torch.Tensor): Batch of states.
                - actions (torch.Tensor): Batch of actions.
                - rewards (torch.Tensor): Batch of rewards.
                - next_states (torch.Tensor): Batch of next states.
                - dones (torch.Tensor): Batch of done flags.
                - tree_indices (np.ndarray): Array of tree indices for the sampled experiences (for priority updates).
                - weights_tensor (torch.Tensor): Importance sampling weights for the batch.
        """
        batch_experiences = []
        tree_indices = np.empty((batch_size,), dtype=np.int32) # To store sumtree leaf indices
        priorities_for_weights = np.empty((batch_size,), dtype=np.float32)
        
        total_p = self.tree.total_priority()
        segment_len = total_p / batch_size # For stratified sampling; ensures samples are from different parts of the distribution
        
        # Anneal beta
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        for i in range(batch_size):
            s = random.uniform(segment_len * i, segment_len * (i + 1))
            tree_idx, priority, data_buffer_idx = self.tree.get_leaf(s)

            batch_experiences.append(self.buffer[data_buffer_idx])
            tree_indices[i] = tree_idx
            priorities_for_weights[i] = priority

        probs = priorities_for_weights / total_p
        
        # N for IS weight calculation: number of experiences in the buffer.
        # Using self.num_experiences is accurate for both partially and fully filled buffers.
        N = self.num_experiences
        weights = (N * probs) ** (-beta)
        
        # Normalize weights by dividing by the maximum weight for stability
        if weights.max() > 0:
            weights /= weights.max()
        else: # Handle case where all priorities (and thus probs and weights) might be zero
            weights = np.ones_like(weights)

        weights_tensor = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

        states = torch.cat([torch.FloatTensor(exp.state).unsqueeze(0) for exp in batch_experiences])
        actions = torch.tensor([exp.action for exp in batch_experiences], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([exp.reward for exp in batch_experiences], dtype=torch.float).unsqueeze(1)
        next_states = torch.cat([torch.FloatTensor(exp.next_state).unsqueeze(0) for exp in batch_experiences])
        dones = torch.tensor([exp.done for exp in batch_experiences], dtype=torch.float).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones, tree_indices, weights_tensor
    
    def update_priorities(self, tree_indices, td_errors):
        """
        Updates the priorities of sampled experiences using their TD errors.

        Args:
            tree_indices (np.ndarray): Array of tree indices for the experiences whose priorities need updating.
            td_errors (np.ndarray): Array of TD errors corresponding to the sampled experiences.
        """
        priorities = self._get_priority(td_errors)
        for tree_idx, priority_val in zip(tree_indices, priorities):
            self.tree.update(tree_idx, priority_val)
    
    def __len__(self):
        return self.num_experiences


class DQNAgent:
    """
    DQN智能体
    """
    def __init__(self, model, target_model, env, device, 
                 buffer_size=100000, batch_size=32, gamma=0.99,
                 lr=1e-4, epsilon_start=1.0, epsilon_final=0.01,
                 epsilon_decay=10000, target_update=1000,
                 prioritized_replay=False):
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        
        if prioritized_replay:
            # For PrioritizedReplayBuffer, alpha, beta_start, beta_frames are class variables or handled internally
            self.memory = PrioritizedReplayBuffer(buffer_size)
            self.prioritized_replay = True
        else:
            self.memory = ReplayBuffer(buffer_size)
            self.prioritized_replay = False
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
    
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
    
    def update_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        if isinstance(self.memory, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, tree_indices, weights = self.memory.sample(self.batch_size)
            weights = weights.to(self.device)
            # Store tree_indices for priority update
            # In the old version, 'indices' were buffer indices, now they are tree_indices
            update_indices = tree_indices
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            update_indices = None
            weights = torch.ones_like(rewards).to(self.device)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        q_values = self.model(states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values_model = self.model(next_states)
            next_actions = next_q_values_model.max(1)[1].unsqueeze(1)
            next_q_values_target = self.target_model(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values_target
        
        td_errors = torch.abs(q_values - target_q_values)
        loss = (td_errors * weights).mean() # td_errors needs to be [batch_size, 1] like weights
        
        if isinstance(self.memory, PrioritizedReplayBuffer) and update_indices is not None:
            # Pass raw TD errors (detached from graph, on CPU)
            td_errors_numpy = td_errors.detach().cpu().numpy().flatten()
            self.memory.update_priorities(update_indices, td_errors_numpy)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_model(self):
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


class NStepBuffer:
    """
    N步学习缓冲区
    存储n步序列并计算n步回报
    """
    def __init__(self, n_step=3, gamma=0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
    
    def add(self, state, action, reward, next_state, done):
        current_experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(current_experience)
        if len(self.buffer) < self.n_step:
            return None
        
        n_step_reward = 0.0
        n_step_next_state = None
        n_step_done = False
        for i in range(self.n_step):
            exp = self.buffer[i]
            n_step_reward += (self.gamma ** i) * exp.reward
            if exp.done:
                n_step_next_state = exp.next_state
                n_step_done = True
                break
            if i == self.n_step - 1:
                n_step_next_state = exp.next_state
                n_step_done = exp.done
        s0, a0, _, _, _ = self.buffer[0]
        return Experience(s0, a0, n_step_reward, n_step_next_state, n_step_done)
    
    def get_last_n_step(self):
        experiences_to_return = []
        for j in range(len(self.buffer)):
            n_step_reward = 0.0
            n_step_next_state = None
            n_step_done = False
            for i in range(self.n_step):
                current_idx_in_buffer = j + i
                if current_idx_in_buffer >= len(self.buffer):
                    if i > 0:
                        n_step_next_state = self.buffer[-1].next_state
                        n_step_done = self.buffer[-1].done
                    break
                exp = self.buffer[current_idx_in_buffer]
                n_step_reward += (self.gamma ** i) * exp.reward
                if exp.done:
                    n_step_next_state = exp.next_state
                    n_step_done = True
                    break
                if i == self.n_step - 1 or current_idx_in_buffer == len(self.buffer) - 1:
                    n_step_next_state = exp.next_state
                    n_step_done = exp.done
            start_exp = self.buffer[j]
            experiences_to_return.append(Experience(start_exp.state, start_exp.action, n_step_reward, n_step_next_state, n_step_done))
        return experiences_to_return
    
    def reset(self):
        self.buffer.clear()

class AdaptiveNStepBuffer:
    """
    Adaptive N-step Learning Buffer.
    Dynamically adjusts N based on TD error history. It stores experiences up to `max_n_step`
    to allow N to grow, but forms n-step returns based on `current_n_step`.

    Args:
        base_n_step (int): The minimum/starting value for N.
        max_n_step (int): The maximum value N can take.
        gamma (float): Discount factor.
        td_error_history_size (int): How many recent TD errors to store for averaging.
        td_error_threshold_low (float): If avg TD error is below this, N may decrease.
        td_error_threshold_high (float): If avg TD error is above this, N may increase.
        n_step_increment (int): How much to increase N by at a time.
        n_step_decrement (int): How much to decrease N by at a time.
    """
    def __init__(self, base_n_step, max_n_step, gamma,
                 td_error_history_size=100,
                 td_error_threshold_low=0.1,
                 td_error_threshold_high=0.5,
                 n_step_increment=1,
                 n_step_decrement=1):
        self.base_n_step = base_n_step
        self.max_n_step = max_n_step
        self.gamma = gamma
        self.current_n_step = base_n_step # Current N value used for forming returns

        # Internal buffer stores up to max_n_step experiences to allow N to grow.
        self.buffer = deque(maxlen=max_n_step)
        self.td_error_history = deque(maxlen=td_error_history_size) # Stores recent TD errors

        self.td_error_threshold_low = td_error_threshold_low
        self.td_error_threshold_high = td_error_threshold_high
        self.n_step_increment = n_step_increment
        self.n_step_decrement = n_step_decrement

        print(f"AdaptiveNStepBuffer initialized: base_n={base_n_step}, max_n={max_n_step}, current_n={self.current_n_step}")

    def add(self, state, action, reward, next_state, done):
        current_experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(current_experience) # Add to the right (end) of the deque

        # Check if we have enough experiences to form at least one n-step sequence
        # based on the current_n_step.
        if len(self.buffer) >= self.current_n_step:
            # Form the n-step experience from the leftmost (oldest) `current_n_step` items.
            n_step_reward = 0.0
            actual_n_used = 0 # Number of steps actually used (can be < current_n_step if episode ends)
            n_step_final_next_state = None
            n_step_final_done = False

            for i in range(self.current_n_step):
                exp = self.buffer[i] # Experience at step t+i from the start of the window
                n_step_reward += (self.gamma ** i) * exp.reward
                actual_n_used = i + 1
                if exp.done: # Episode ended within the n-step window
                    n_step_final_next_state = exp.next_state
                    n_step_final_done = True
                    break
                if i == self.current_n_step - 1: # Reached the end of the current_n_step window
                    n_step_final_next_state = exp.next_state
                    n_step_final_done = exp.done

            s0, a0, _, _, _ = self.buffer[0] # Initial state and action of the n-step sequence

            # Remove the oldest experience (s0, a0, ...) to slide the window.
            self.buffer.popleft()

            return Experience(s0, a0, n_step_reward, n_step_final_next_state, n_step_final_done)

        return None # Not enough elements yet to form an n-step sequence based on current_n_step

    def record_td_error(self, td_error):
        """Records a TD error to the history for N-step adaptation."""
        self.td_error_history.append(abs(td_error))

    def adapt_n_step(self):
        if not self.td_error_history: # or len(self.td_error_history) < some_min_samples
            return

        avg_td_error = np.mean(list(self.td_error_history))
        # The deque `td_error_history` automatically manages its size via `maxlen`.
        # No need to clear explicitly if we want a sliding window average.

        prev_n_step = self.current_n_step
        # Increase N if error is high and N is below max
        if avg_td_error > self.td_error_threshold_high and self.current_n_step < self.max_n_step:
            self.current_n_step = min(self.current_n_step + self.n_step_increment, self.max_n_step)
        # Decrease N if error is low and N is above base
        elif avg_td_error < self.td_error_threshold_low and self.current_n_step > self.base_n_step:
            self.current_n_step = max(self.current_n_step - self.n_step_decrement, self.base_n_step)

        if prev_n_step != self.current_n_step:
            print(f"AdaptiveNStep: N changed from {prev_n_step} to {self.current_n_step} (Avg TD Error: {avg_td_error:.4f})")
            # Note: Changing current_n_step affects how many items `add()` requires before returning
            # an n-step experience and how many items it pops. The internal `self.buffer`
            # (with maxlen=max_n_step) remains the source of experiences.

    def get_last_n_step(self):
        """
        Processes remaining experiences in the buffer at the end of an episode.
        For each experience that could be the start of an n-step sequence (up to current_n_step),
        it forms the n-step return.
        """
        experiences_to_return = []
        # Create a temporary list from the deque to iterate without modification issues
        # as `self.buffer` might be modified if `add` was called by an external process (not typical here).
        # However, this method is usually called after the episode is truly done.

        # The internal buffer `self.buffer` contains the last `len(self.buffer)` experiences.
        # We need to form n-step returns for sequences starting at each possible point
        # within this buffer.
        # Example: buffer = [e0, e1, e2], current_n_step = 3
        #   - Returns n-step for (e0, e1, e2)
        #   - Then, conceptually, for (e1, e2) if it were shorter than n_step
        #   - Then, for (e2)
        # This is similar to how NStepBuffer's get_last_n_step worked in the original fixed-N version.

        # We iterate through the current buffer, considering each element as a potential start.
        # The number of items to process for n-step return from that start point is
        # min(current_n_step, number_of_remaining_items_in_buffer_from_start).

        # Create a snapshot for safe iteration, as the main buffer should not be modified here.
        # The `AdaptiveNStepBuffer.add` method with `popleft` means `self.buffer` contains
        # less than `current_n_step` items typically by the time `get_last_n_step` is called,
        # because `add` only returns None if `len(buffer) < current_n_step` *after* append.
        # `get_last_n_step` is called after the *final* `done` for an episode.
        # The `store_experience` method calls `n_step_buffer.add()`, then if `done`, calls `get_last_n_step()`, then `reset()`.
        # So, `self.buffer` at this point contains the tail end experiences that did not yet form a full `current_n_step` sequence
        # and were not popped by `add`.

        # Example: current_n=3. Buffer has [e_final-1, e_final].
        # For e_final-1: n-step uses [e_final-1, e_final].
        # For e_final: n-step uses [e_final].

        temp_processing_buffer = list(self.buffer)

        for j in range(len(temp_processing_buffer)): # j is the starting index in temp_processing_buffer
            n_step_reward = 0.0
            actual_n_used = 0
            n_step_next_state = None
            n_step_done = False # Default for the transition

            # Calculate reward and find next_state/done for the transition starting at temp_processing_buffer[j]
            # The window size for these tail-end transitions should still be guided by current_n_step,
            # but practically limited by available data or episode termination.
            for i in range(self.current_n_step):
                current_idx_in_temp_buffer = j + i
                if current_idx_in_temp_buffer >= len(temp_processing_buffer):
                    # Ran out of experiences in the temp_buffer for this n-step calculation.
                    # This means the n_step_next_state and n_step_done are from the last actual experience processed.
                    if i > 0: # Ensure at least one step was processed.
                        # This state should ideally not be reached if logic is correct,
                        # as break on exp.done or end of loop should set these.
                        # Fallback to last item in temp_buffer if necessary.
                        n_step_next_state = temp_processing_buffer[-1].next_state
                        n_step_done = temp_processing_buffer[-1].done
                    break

                exp = temp_processing_buffer[current_idx_in_temp_buffer]
                n_step_reward += (self.gamma ** i) * exp.reward
                actual_n_used = i + 1

                if exp.done: # Episode terminated within this n-step window
                    n_step_next_state = exp.next_state
                    n_step_done = True
                    break

                # If loop finishes, or it's the last available item for this starting point j
                if i == self.current_n_step - 1 or current_idx_in_temp_buffer == len(temp_processing_buffer) - 1:
                    n_step_next_state = exp.next_state
                    n_step_done = exp.done

            if actual_n_used > 0: # Only add if at least one step was processed
                 start_exp = temp_processing_buffer[j]
                 experiences_to_return.append(Experience(start_exp.state, start_exp.action, n_step_reward, n_step_next_state, n_step_done))
        return experiences_to_return

    def reset(self):
        self.buffer.clear()
        self.td_error_history.clear()
        # Optional: reset n_step to base_n_step, or let it persist across episodes
        # print(f"AdaptiveNStepBuffer reset. Current N is {self.current_n_step}")


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
                 n_atoms=51, v_min=-10, v_max=10, **kwargs): # Removed n_step from here, use base_n_step
        if use_noisy:
            kwargs['epsilon_start'] = 0.0
            kwargs['epsilon_final'] = 0.0
            kwargs['epsilon_decay'] = 1
        if 'prioritized_replay' not in kwargs:
            kwargs['prioritized_replay'] = True
            
        super().__init__(model, target_model, env, device, **kwargs)
        
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
        self.training_steps_count = 0 # Counter for adapt_n_step frequency

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

        n_step_exp = self.n_step_buffer.add(current_state, current_action, current_reward, current_next_state, current_done)
        if n_step_exp is not None:
            self.memory.push(n_step_exp.state, n_step_exp.action, n_step_exp.reward, n_step_exp.next_state, n_step_exp.done)
        if done:
            remaining_exps = self.n_step_buffer.get_last_n_step()
            for exp in remaining_exps:
                self.memory.push(exp.state, exp.action, exp.reward, exp.next_state, exp.done)
            self.n_step_buffer.reset()
    
    def update_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        
        if self.use_noisy:
            if hasattr(self.model, 'sample_noise'): self.model.sample_noise()
            if hasattr(self.target_model, 'sample_noise'): self.target_model.sample_noise()
        
        if isinstance(self.memory, PrioritizedReplayBuffer):
            states, actions, rewards, next_states, dones, tree_indices, weights = self.memory.sample(self.batch_size)
            weights = weights.to(self.device)
            update_indices = tree_indices # For PER
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            update_indices = None
            weights = torch.ones_like(rewards).to(self.device)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device) # These are already n-step rewards from the buffer
        next_states = next_states.to(self.device)
        dones = dones.to(self.device) # This 'done' is the n-step done
        
        # Loss calculation uses self.n_step_buffer.current_n_step for discounting Q(s',a')
        if self.use_distributional:
            loss = self._compute_distributional_loss(states, actions, rewards, next_states, dones, weights)
        else:
            loss = self._compute_standard_loss(states, actions, rewards, next_states, dones, weights)
        
        # Calculate TD errors for PER update and N-step adaptation
        td_errors_numpy = None
        with torch.no_grad(): # All calculations for TD error should not affect gradients
            if self.use_distributional:
                # For distributional, TD error is often |E[Q(s,a)] - (r + gamma^N * E[Q(s',a')])|
                # Re-calculate current Q's expected value for selected actions
                current_dist_probs_model = self.model(states)
                current_q_selected_dist_probs = current_dist_probs_model.gather(1, actions.unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)
                current_q_selected_expected = (current_q_selected_dist_probs * self.support.unsqueeze(0)).sum(1, keepdim=True)

                # Re-calculate target Q's expected value (Double DQN style for action selection)
                next_dist_probs_model = self.model(next_states)
                next_q_values_model = (next_dist_probs_model * self.support.unsqueeze(0)).sum(2)
                next_actions = next_q_values_model.max(1)[1].unsqueeze(1)

                next_dist_probs_target_net = self.target_model(next_states)
                next_best_dist_target_net = next_dist_probs_target_net.gather(1, next_actions.unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)
                next_q_max_expected = (next_best_dist_target_net * self.support.unsqueeze(0)).sum(1, keepdim=True)

                # Target Q value for TD error calculation
                # rewards are already n-step rewards from buffer
                target_q_for_td_error = rewards + (1 - dones) * (self.gamma ** self.n_step_buffer.current_n_step) * next_q_max_expected
                td_errors = torch.abs(current_q_selected_expected - target_q_for_td_error)
            else: # Standard DQN
                current_q_model_vals = self.model(states).gather(1, actions)
                
                next_q_values_model = self.model(next_states)
                next_actions = next_q_values_model.max(1)[1].unsqueeze(1)
                next_q_target_net = self.target_model(next_states).gather(1, next_actions)

                # Target Q value for TD error calculation
                # rewards are already n-step rewards from buffer
                target_q_for_td_error = rewards + (1 - dones) * (self.gamma ** self.n_step_buffer.current_n_step) * next_q_target_net
                td_errors = torch.abs(current_q_model_vals - target_q_for_td_error)

            td_errors_numpy = td_errors.detach().cpu().numpy().flatten()

        # Update PER priorities if applicable
        if isinstance(self.memory, PrioritizedReplayBuffer) and update_indices is not None and td_errors_numpy is not None:
            self.memory.update_priorities(update_indices, td_errors_numpy)
        
        # Record TD errors for N-step adaptation
        if td_errors_numpy is not None: # Ensure td_errors were computed
             self.n_step_buffer.record_td_error(np.mean(td_errors_numpy))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()
        
        self.training_steps_count += 1
        if self.training_steps_count % self.adapt_n_step_freq == 0:
            self.n_step_buffer.adapt_n_step()
            # Potentially log self.n_step_buffer.current_n_step here

        return loss.item()
    
    def _compute_standard_loss(self, states, actions, rewards, next_states, dones, weights):
        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values_model = self.model(next_states)
            next_actions = next_q_values_model.max(1)[1].unsqueeze(1)
            next_q_values_target = self.target_model(next_states).gather(1, next_actions)
            # Use current_n_step from adaptive buffer for discounting future rewards
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step_buffer.current_n_step) * next_q_values_target
        # td_errors = torch.abs(q_values - target_q_values) # Ensure td_errors has shape [batch_size, 1]
        td_errors = q_values - target_q_values # Loss is (q_values - target_q_values).pow(2) * weights or huber_loss
        # Using Huber loss might be better, but for consistency with current td_errors = abs(...)
        # loss = (td_errors.pow(2) * weights).mean() # This is MSE like
        loss = (torch.abs(td_errors) * weights).mean() # This is MAE like, but original was also this for PER error calc
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