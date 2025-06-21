"""
Implements standard and prioritized replay buffers for reinforcement learning agents.

This module includes:
- SumTree: A binary tree data structure for efficient prioritized sampling.
- ReplayBuffer: A standard FIFO experience replay buffer.
- PrioritizedReplayBuffer (PER): A replay buffer that samples experiences based on their TD errors,
  allowing the agent to focus on more significant experiences.
- _batch_experiences_to_tensors: A helper function to convert lists of experiences
  into batched PyTorch tensors.
"""
import numpy as np
import torch
import random
from collections import deque
# Import Experience from utils.py (one level up)
from ..utils import Experience

def _batch_experiences_to_tensors(experiences: list[Experience]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts a list of Experience tuples into batched PyTorch tensors."""
    states = torch.cat([torch.FloatTensor(exp.state).unsqueeze(0) for exp in experiences])
    actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float).unsqueeze(1)
    next_states = torch.cat([torch.FloatTensor(exp.next_state).unsqueeze(0) for exp in experiences])
    dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float).unsqueeze(1)
    return states, actions, rewards, next_states, dones

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
        # self._propagate(tree_idx, change) # DIAGNOSTIC: Disable propagation
        # Propagate change to parent nodes

        # Update max_priority if a leaf's priority (and not an internal node's sum) changes to be the new max
        if priority > self._max_priority and tree_idx >= (self.capacity - 1): # tree_idx for leaves start at capacity - 1
             self._max_priority = priority
        
        # Diagnostic log for very large priorities
        if self.tree[tree_idx] > 1e10: # Log if priority exceeds 10 billion
            print(f"DEBUG: Large priority updated in SumTree: {self.tree[tree_idx]:.2e} at tree_idx {tree_idx}")
            # Potentially add sys.stdout.flush() if immediate printing is needed and not happening


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
        return _batch_experiences_to_tensors(experiences)

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

        # Add epsilon to total_p to prevent division by zero if total_p is zero
        probs = priorities_for_weights / (total_p + 1e-8) # Added epsilon to total_p

        # N for IS weight calculation: number of experiences in the buffer.
        # Using self.num_experiences is accurate for both partially and fully filled buffers.
        N = self.num_experiences

        # Add epsilon to the base of the power to prevent (0)^(-beta) -> inf
        weights_base = N * probs
        weights = (weights_base + 1e-8) ** (-beta) # Added epsilon to base

        # Normalize weights by dividing by the maximum weight for stability
        max_weight = weights.max()
        if np.isfinite(max_weight) and max_weight > 1e-8: # Check if max_weight is valid and positive
            weights /= max_weight
        else: # Handle case where all priorities (and thus probs and weights) might be zero, or max_weight is problematic
            weights = np.ones_like(weights)

        weights_tensor = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

        # Convert batch of experiences to tensors using the helper function
        states, actions, rewards, next_states, dones = _batch_experiences_to_tensors(batch_experiences)

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
