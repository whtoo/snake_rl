import unittest
import numpy as np
from collections import deque

try:
    from src.buffers.n_step_buffers import AdaptiveNStepBuffer
    from src.utils import Experience # Changed from src.agent
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.buffers.n_step_buffers import AdaptiveNStepBuffer
    from src.utils import Experience # Changed from src.agent

class TestAdaptiveNStepBuffer(unittest.TestCase):

    def _create_dummy_experience_tuples(self, num_experiences, start_val=0, done_at_idx=None):
        experiences = []
        for i in range(num_experiences):
            state = np.array([i + start_val]) # Simple state
            action = i % 2
            reward = float(i + 1) # Reward = 1, 2, 3...
            next_state = np.array([i + start_val + 1])
            done = (done_at_idx == i)
            experiences.append(Experience(state, action, reward, next_state, done))
        return experiences

    def test_initialization(self):
        buffer = AdaptiveNStepBuffer(base_n_step=3, max_n_step=10, gamma=0.99)
        self.assertEqual(buffer.current_n_step, 3)
        self.assertEqual(buffer.max_n_step, 10)
        self.assertEqual(buffer.gamma, 0.99)
        self.assertEqual(len(buffer.buffer), 0)
        self.assertEqual(len(buffer.td_error_history), 0)

    def test_add_and_n_step_return(self):
        buffer = AdaptiveNStepBuffer(base_n_step=3, max_n_step=5, gamma=0.9)

        experiences = self._create_dummy_experience_tuples(5)

        # Add 2 experiences, buffer not full enough for current_n_step=3
        self.assertIsNone(buffer.add(*experiences[0]))
        self.assertIsNone(buffer.add(*experiences[1]))

        # Add 3rd experience, should return 3-step experience
        exp_3_step = buffer.add(*experiences[2])
        self.assertIsNotNone(exp_3_step)

        # s0, a0, r0, s1, d0=F (reward 1)
        # s1, a1, r1, s2, d1=F (reward 2)
        # s2, a2, r2, s3, d2=F (reward 3)
        # Expected 3-step reward: r0 + gamma*r1 + gamma^2*r2 = 1 + 0.9*2 + 0.9^2*3 = 1 + 1.8 + 0.81*3 = 1 + 1.8 + 2.43 = 5.23
        self.assertEqual(exp_3_step.state, experiences[0].state)
        self.assertEqual(exp_3_step.action, experiences[0].action)
        self.assertAlmostEqual(exp_3_step.reward, 5.23)
        self.assertEqual(exp_3_step.next_state, experiences[2].next_state) # s3
        self.assertFalse(exp_3_step.done)

        # Add 4th experience
        exp_3_step_2 = buffer.add(*experiences[3]) # Starts from experiences[1]
        # s1, a1, r1, s2, d1=F (reward 2)
        # s2, a2, r2, s3, d2=F (reward 3)
        # s3, a3, r3, s4, d3=F (reward 4)
        # Expected 3-step reward: r1 + gamma*r2 + gamma^2*r3 = 2 + 0.9*3 + 0.9^2*4 = 2 + 2.7 + 0.81*4 = 2 + 2.7 + 3.24 = 7.94
        self.assertEqual(exp_3_step_2.state, experiences[1].state)
        self.assertAlmostEqual(exp_3_step_2.reward, 7.94)
        self.assertEqual(exp_3_step_2.next_state, experiences[3].next_state) # s4
        self.assertFalse(exp_3_step_2.done)

    def test_add_with_early_done(self):
        buffer = AdaptiveNStepBuffer(base_n_step=3, max_n_step=5, gamma=0.9)
        # s0, a0, r0, s1, d0=F (reward 1)
        # s1, a1, r1, s2, d1=True (reward 2) -> episode ends here
        # s2, a2, r2, s3, d2=F (reward 3) -> this won't be added if s1 is done
        experiences = self._create_dummy_experience_tuples(3, done_at_idx=1)

        buffer.add(*experiences[0])
        buffer.add(*experiences[1]) # Episode ends here

        # Buffer has [exp0, exp1]. current_n_step=3. len(buffer)=2.
        # The next add will make len(buffer) = 3.
        # An n-step experience starting from exp0 will be formed.
        # This experience will use exp0 and exp1 (which is done).
        # Reward = r0 + gamma*r1. Next state is exp1's next_state. Done is True.

        buffer.reset() # Clear buffer for clean test
        buffer.add(*experiences[0]) # s0, a0, r1, s1, F
        buffer.add(*experiences[1]) # s1, a1, r2, s2, T -- buffer is [e0, e1]

        # Add third experience, which makes buffer long enough (3 elements) for current_n_step=3
        n_step_exp = buffer.add(*experiences[2]) # s2, a2, r3, s3, F -- buffer becomes [e1, e2] after popleft

        self.assertIsNotNone(n_step_exp) # An experience should be returned
        self.assertEqual(n_step_exp.state, experiences[0].state) # state should be s0
        self.assertEqual(n_step_exp.action, experiences[0].action) # action should be a0
        # Reward = r0_raw + gamma * r1_raw = 1.0 + 0.9 * 2.0 = 1.0 + 1.8 = 2.8
        self.assertAlmostEqual(n_step_exp.reward, 2.8)
        self.assertEqual(n_step_exp.next_state, experiences[1].next_state) # next_state should be s2 (from exp1)
        self.assertTrue(n_step_exp.done) # done should be True (from exp1)

    def test_n_step_adaptation(self):
        buffer = AdaptiveNStepBuffer(base_n_step=2, max_n_step=5, gamma=0.99,
                                     td_error_threshold_low=0.2,
                                     td_error_threshold_high=0.8,
                                     td_error_history_size=3)
        self.assertEqual(buffer.current_n_step, 2)

        # Low TD errors -> decrease N (but capped at base_n_step)
        buffer.record_td_error(0.1)
        buffer.record_td_error(0.05)
        buffer.record_td_error(0.1)
        buffer.adapt_n_step()
        self.assertEqual(buffer.current_n_step, 2) # Should not go below base

        # High TD errors -> increase N
        buffer.td_error_history.clear()
        buffer.record_td_error(0.9)
        buffer.record_td_error(1.0)
        buffer.record_td_error(0.85)
        buffer.adapt_n_step()
        self.assertEqual(buffer.current_n_step, 2 + buffer.n_step_increment) # Expected 2+1=3

        # Increase N again
        current_n = buffer.current_n_step
        buffer.td_error_history.clear()
        buffer.record_td_error(0.9)
        buffer.record_td_error(1.0)
        buffer.record_td_error(0.85)
        buffer.adapt_n_step()
        self.assertEqual(buffer.current_n_step, min(current_n + buffer.n_step_increment, buffer.max_n_step))

        # Fill up to max_n_step
        buffer.current_n_step = buffer.max_n_step -1
        buffer.td_error_history.clear()
        buffer.record_td_error(0.9)
        buffer.record_td_error(1.0)
        buffer.record_td_error(0.85)
        buffer.adapt_n_step()
        self.assertEqual(buffer.current_n_step, buffer.max_n_step)

        # Try to increase N beyond max_n_step
        buffer.td_error_history.clear()
        buffer.record_td_error(0.9)
        buffer.record_td_error(1.0)
        buffer.record_td_error(0.85)
        buffer.adapt_n_step()
        self.assertEqual(buffer.current_n_step, buffer.max_n_step) # Should be capped at max

        # Low TD errors -> decrease N
        buffer.td_error_history.clear()
        buffer.record_td_error(0.1)
        buffer.record_td_error(0.05)
        buffer.record_td_error(0.1)
        buffer.adapt_n_step()
        self.assertEqual(buffer.current_n_step, buffer.max_n_step - buffer.n_step_decrement)


    def test_get_last_n_step(self):
        # Scenario: episode ends, buffer has some items. current_n_step = 3
        # Experiences: e0, e1 (done=True)
        buffer = AdaptiveNStepBuffer(base_n_step=3, max_n_step=5, gamma=0.9)
        exps_raw = self._create_dummy_experience_tuples(2, done_at_idx=1) # e0, e1(done)

        # Manually fill buffer as if these were the last items before episode end
        buffer.buffer.append(exps_raw[0])
        buffer.buffer.append(exps_raw[1])

        last_experiences = buffer.get_last_n_step()
        self.assertEqual(len(last_experiences), 2)

        # For e0 (j=0):
        # i=0: r0_contrib = r0_raw = 1. exp=e0. actual_n=1
        # i=1: r1_contrib = gamma*r1_raw = 0.9*2 = 1.8. exp=e1. actual_n=2. done=True. Break.
        # Total reward for e0_n_step = 1 + 1.8 = 2.8
        # Next state for e0_n_step = e1.next_state. Done for e0_n_step = e1.done
        e0_n_step = last_experiences[0]
        self.assertEqual(e0_n_step.state, exps_raw[0].state)
        self.assertAlmostEqual(e0_n_step.reward, 2.8)
        self.assertEqual(e0_n_step.next_state, exps_raw[1].next_state)
        self.assertTrue(e0_n_step.done)

        # For e1 (j=1):
        # i=0: r0_contrib = r1_raw = 2. exp=e1. actual_n=1. done=True. Break.
        # Total reward for e1_n_step = 2
        # Next state for e1_n_step = e1.next_state. Done for e1_n_step = e1.done
        e1_n_step = last_experiences[1]
        self.assertEqual(e1_n_step.state, exps_raw[1].state)
        self.assertAlmostEqual(e1_n_step.reward, 2.0)
        self.assertEqual(e1_n_step.next_state, exps_raw[1].next_state)
        self.assertTrue(e1_n_step.done)

    def test_reset(self):
        buffer = AdaptiveNStepBuffer(base_n_step=3, max_n_step=5, gamma=0.99)
        experiences = self._create_dummy_experience_tuples(3)
        for exp in experiences:
            buffer.add(*exp)
        buffer.record_td_error(0.5)

        self.assertNotEqual(len(buffer.buffer), 0)
        self.assertNotEqual(len(buffer.td_error_history), 0)

        buffer.reset()
        self.assertEqual(len(buffer.buffer), 0)
        self.assertEqual(len(buffer.td_error_history), 0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
