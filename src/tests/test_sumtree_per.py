import unittest
import numpy as np
import torch
import random

# Adjust import path based on where this test file is relative to src
# Assuming src/tests/test_sumtree_per.py, then src.agent should work if PYTHONPATH includes project root
try:
    from src.agent import SumTree, PrioritizedReplayBuffer, Experience
except ImportError:
    # Fallback for different execution context (e.g. if tests are run from project root)
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.agent import SumTree, PrioritizedReplayBuffer, Experience


class TestSumTree(unittest.TestCase):
    def test_initialization(self):
        tree = SumTree(capacity=16)
        self.assertEqual(tree.capacity, 16)
        self.assertEqual(len(tree.tree), 2 * 16 - 1)
        self.assertEqual(len(tree.data_indices), 16)
        self.assertEqual(tree.total_priority(), 0)

    def test_add_and_total_priority(self):
        tree = SumTree(capacity=4)
        tree.add(priority=0.5, data_buffer_idx=0)
        self.assertAlmostEqual(tree.total_priority(), 0.5)
        tree.add(priority=1.0, data_buffer_idx=1)
        self.assertAlmostEqual(tree.total_priority(), 1.5)
        tree.add(priority=0.2, data_buffer_idx=2)
        self.assertAlmostEqual(tree.total_priority(), 1.7)
        tree.add(priority=0.8, data_buffer_idx=3)
        self.assertAlmostEqual(tree.total_priority(), 2.5)
        # Test overwrite
        tree.add(priority=1.2, data_buffer_idx=4) # Overwrites data_idx 0
        self.assertAlmostEqual(tree.total_priority(), 2.5 - 0.5 + 1.2)


    def test_update_and_propagation(self):
        tree = SumTree(capacity=4)
        # Initial adds (data_buffer_idx are 0,1,2,3 for tree leaves 3,4,5,6)
        # Leaf indices: capacity-1 to 2*capacity-2
        # For capacity 4: leaves are at tree indices 3, 4, 5, 6
        tree.add(0.5, 0) # tree_idx 3 (via data_pointer 0)
        tree.add(1.0, 1) # tree_idx 4 (via data_pointer 1)
        tree.add(0.2, 2) # tree_idx 5 (via data_pointer 2)
        tree.add(0.8, 3) # tree_idx 6 (via data_pointer 3)

        initial_total = tree.total_priority() # 2.5

        # Update priority of leaf corresponding to data_pointer 1 (which was the second add)
        # data_pointer would be 0 after 4 adds.
        # The leaf that had 1.0 (data_idx 1) is at tree.data_pointer = 1 before wrapping, so tree_idx = 1 + 4 - 1 = 4
        tree_idx_to_update = 4
        tree.update(tree_idx_to_update, 0.1) # Update priority from 1.0 to 0.1

        self.assertAlmostEqual(tree.tree[tree_idx_to_update], 0.1)
        self.assertAlmostEqual(tree.total_priority(), initial_total - 1.0 + 0.1)

    def test_get_leaf(self):
        tree = SumTree(capacity=4)
        priorities = [0.5, 1.0, 0.2, 0.8]
        for i, p in enumerate(priorities):
            tree.add(p, data_buffer_idx=i)

        total_p = tree.total_priority()
        self.assertAlmostEqual(total_p, sum(priorities))

        # Test sampling (deterministic for specific s values)
        # Leaf indices for capacity 4 are 3, 4, 5, 6
        # Tree structure (example, values are priorities):
        #      2.5
        #    /     \
        #   1.5     1.0
        #  /   \   /   \
        # 0.5  1.0 0.2  0.8  (tree indices 3,4,5,6 - data_indices 0,1,2,3)

        # Sample for first element (data_idx 0, priority 0.5)
        idx, p_val, data_idx = tree.get_leaf(0.1)
        self.assertEqual(data_idx, 0)
        self.assertAlmostEqual(p_val, 0.5)

        # Sample for second element (data_idx 1, priority 1.0)
        idx, p_val, data_idx = tree.get_leaf(0.6) # 0.5 < s <= 1.5
        self.assertEqual(data_idx, 1)
        self.assertAlmostEqual(p_val, 1.0)

        # Sample for third element (data_idx 2, priority 0.2)
        idx, p_val, data_idx = tree.get_leaf(1.6) # 1.5 < s <= 1.7
        self.assertEqual(data_idx, 2)
        self.assertAlmostEqual(p_val, 0.2)

        # Sample for fourth element (data_idx 3, priority 0.8)
        idx, p_val, data_idx = tree.get_leaf(2.0) # 1.7 < s <= 2.5
        self.assertEqual(data_idx, 3)
        self.assertAlmostEqual(p_val, 0.8)

    def test_max_priority_tracking(self):
        tree = SumTree(capacity=4)
        self.assertEqual(tree.max_priority, 1.0) # Default
        tree.add(0.5, 0)
        self.assertEqual(tree.max_priority, 1.0) # Still default as 0.5 < 1.0
        tree.add(1.5, 1)
        self.assertEqual(tree.max_priority, 1.5)
        tree.add(0.2, 2)
        self.assertEqual(tree.max_priority, 1.5)
        tree.update(tree.capacity - 1 + 2, 2.0) # Update priority of data_idx 2 to 2.0
        self.assertEqual(tree.max_priority, 2.0)
        tree.add(0.1, 3) # Max priority should remain 2.0
        self.assertEqual(tree.max_priority, 2.0)


class TestPrioritizedReplayBuffer(unittest.TestCase):
    def _create_dummy_experience(self, i=0):
        state = np.full((4, 84, 84), i, dtype=np.uint8)
        action = i % 2
        reward = float(i)
        next_state = np.full((4, 84, 84), i + 1, dtype=np.uint8)
        done = False
        return state, action, reward, next_state, done

    def test_initialization(self):
        buffer = PrioritizedReplayBuffer(capacity=100)
        self.assertEqual(buffer.capacity, 100)
        self.assertEqual(len(buffer), 0)

    def test_push(self):
        buffer = PrioritizedReplayBuffer(capacity=5)
        s, a, r, ns, d = self._create_dummy_experience(0)
        buffer.push(s, a, r, ns, d)
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.buffer[0].action, a)
        # Check if tree has an entry with max priority
        self.assertAlmostEqual(buffer.tree.total_priority(), buffer.tree.max_priority)

        s1, a1, r1, ns1, d1 = self._create_dummy_experience(1)
        buffer.push(s1, a1, r1, ns1, d1)
        self.assertEqual(len(buffer), 2)
        # Total priority should be 2 * max_priority (as new items get max_priority)
        self.assertAlmostEqual(buffer.tree.total_priority(), 2 * buffer.tree.max_priority)


    def test_sample(self):
        capacity = 10
        buffer = PrioritizedReplayBuffer(capacity=capacity)
        for i in range(capacity):
            s, a, r, ns, d = self._create_dummy_experience(i)
            buffer.push(s, a, r, ns, d)

        self.assertEqual(len(buffer), capacity)

        batch_size = 4
        states, actions, rewards, next_states, dones, tree_indices, weights = buffer.sample(batch_size)

        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(len(tree_indices), batch_size)
        self.assertEqual(weights.shape[0], batch_size)
        self.assertTrue(all(w > 0 for w in weights.flatten().tolist()))

    def test_update_priorities(self):
        capacity = 10
        buffer = PrioritizedReplayBuffer(capacity=capacity)
        for i in range(capacity):
            s, a, r, ns, d = self._create_dummy_experience(i)
            buffer.push(s, a, r, ns, d) # All pushed with max_priority (likely 1.0)

        batch_size = 4
        _, _, _, _, _, tree_indices, _ = buffer.sample(batch_size)

        # Simulate some TD errors
        td_errors = np.random.rand(batch_size).astype(np.float32)
        initial_priorities_sampled = [buffer.tree.tree[idx] for idx in tree_indices]

        buffer.update_priorities(tree_indices, td_errors)

        updated_priorities_sampled = [buffer.tree.tree[idx] for idx in tree_indices]

        for i in range(batch_size):
            expected_p = (abs(td_errors[i]) + buffer.epsilon) ** buffer.alpha
            self.assertAlmostEqual(updated_priorities_sampled[i], expected_p, places=5)
            # Also check that other priorities in the tree were not unintentionally changed
            # This is harder to check exhaustively without knowing all tree indices.

    def test_beta_annealing_and_is_weights(self):
        capacity = 100
        buffer = PrioritizedReplayBuffer(capacity=capacity)
        for i in range(capacity):
            buffer.push(*self._create_dummy_experience(i))

        # Frame 1 (after init), beta should be beta_start
        # We can't directly check buffer.beta, but we can check IS weights behavior indirectly
        # or trust the sample() method's internal logic.
        # For this test, we'll mostly check that weights are valid.

        initial_beta = buffer.beta_start # Store expected initial beta
        _, _, _, _, _, tree_indices1, weights1 = buffer.sample(4)
        # After one sample, buffer.frame becomes 2. beta for next sample will be calculated using frame=1.
        # The beta used for weights1 was indeed beta_start (frame=1 was used in beta_by_frame calculation).

        # Simulate many frames passing to check beta annealing
        buffer.frame = buffer.beta_frames // 2
        beta_mid_expected = min(1.0, buffer.beta_start + (buffer.beta_frames // 2) * (1.0 - buffer.beta_start) / buffer.beta_frames)
        _, _, _, _, _, tree_indices2, weights2 = buffer.sample(4)
        # The beta used for weights2 was beta_mid_expected

        buffer.frame = buffer.beta_frames * 2 # Ensure beta reaches 1.0 (or more, will be clamped)
        beta_final_expected = 1.0
        _, _, _, _, _, tree_indices3, weights3 = buffer.sample(4)
        # The beta used for weights3 was beta_final_expected

        # Weights should be <= 1 and > 0
        self.assertTrue(torch.all(weights1 <= 1.0) and torch.all(weights1 > 0))
        self.assertTrue(torch.all(weights2 <= 1.0) and torch.all(weights2 > 0))
        self.assertTrue(torch.all(weights3 <= 1.0) and torch.all(weights3 > 0))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
