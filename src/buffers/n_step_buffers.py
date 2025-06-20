"""
Implements N-step learning buffers for reinforcement learning agents.

N-step learning helps bridge the gap between Monte Carlo and Temporal Difference learning
by accumulating rewards over N steps before bootstrapping.

This module includes:
- NStepBuffer: A buffer for fixed N-step returns.
- AdaptiveNStepBuffer: A buffer that dynamically adjusts N based on TD error history.
- _calculate_n_step_transition_details: A helper function to compute N-step rewards,
  next states, and done flags from a sequence of experiences.
"""
import numpy as np
from collections import deque
# Import Experience from utils.py (one level up)
from ..utils import Experience

def _calculate_n_step_transition_details(experience_window: deque, gamma: float, n_steps_to_take: int) -> tuple[float, any, bool, int]:
    """
    Calculates N-step reward, next_state, done flag, and actual N used from a window of experiences.

    Args:
        experience_window: A deque of Experience tuples.
        gamma: Discount factor.
        n_steps_to_take: The desired number of steps for the N-step return.

    Returns:
        A tuple (n_step_reward, n_step_next_state, n_step_done, actual_n_used).
    """
    n_step_reward = 0.0
    actual_n_used = 0
    n_step_final_next_state = None
    n_step_final_done = False

    for i in range(n_steps_to_take):
        if i >= len(experience_window): # Should not happen if called correctly
            break

        exp = experience_window[i]
        n_step_reward += (gamma ** i) * exp.reward
        actual_n_used = i + 1

        if exp.done:
            n_step_final_next_state = exp.next_state
            n_step_final_done = True
            break # Episode ended within the n-step window

        # If it a full n-step sequence, or the last available experience in a shorter window
        if i == n_steps_to_take - 1 or i == len(experience_window) - 1:
            n_step_final_next_state = exp.next_state
            n_step_final_done = exp.done

    return n_step_reward, n_step_final_next_state, n_step_final_done, actual_n_used

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

        # Use helper to calculate N-step details from the current buffer window (which is self.buffer)
        # The helper receives the current state of self.buffer (which has maxlen = self.n_step).
        # If self.buffer has n_step elements, the oldest one (self.buffer[0]) is the one for which
        # we are calculating the n-step return. This experience will be implicitly dropped when
        # a new experience is added later and the deque reaches its maxlen.
        n_step_calc_reward, n_step_calc_next_state, n_step_calc_done, _ = _calculate_n_step_transition_details(self.buffer, self.gamma, self.n_step)

        s0, a0, _, _, _ = self.buffer[0] # s0, a0 are from the first experience in the current n-step window
        # The self.buffer (deque with maxlen=n_step) will automatically discard the oldest element (s0, a0, ...)";
        # when the next experience is added if it is full, maintaining the sliding window.";
        return Experience(s0, a0, n_step_calc_reward, n_step_calc_next_state, n_step_calc_done)

    def get_last_n_step(self):
        experiences_to_return = []
        temp_buffer_list = list(self.buffer) # Create a stable list for iteration
        for j in range(len(temp_buffer_list)):
            # Define the window for the current starting experience
            # The window starts at temp_buffer_list[j]
            current_window = deque([temp_buffer_list[k] for k in range(j, len(temp_buffer_list))])

            # Use helper to calculate N-step details for this specific window
            # self.n_step is the max steps for this buffer type
            n_step_r, n_step_ns, n_step_d, actual_n = _calculate_n_step_transition_details(current_window, self.gamma, self.n_step)

            if actual_n > 0: # Only if some steps were processed
                start_exp = temp_buffer_list[j]
                experiences_to_return.append(Experience(start_exp.state, start_exp.action, n_step_r, n_step_ns, n_step_d))
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
            # The window for calculation is the leftmost `current_n_step` experiences from self.buffer.
            # The helper function will process up to self.current_n_step items from the start of the deque.
            n_step_r, n_step_ns, n_step_d, _ = _calculate_n_step_transition_details(self.buffer, self.gamma, self.current_n_step)

            s0, a0, _, _, _ = self.buffer[0] # Initial state and action of the n-step sequence

            self.buffer.popleft() # Remove the oldest experience to slide the window

            return Experience(s0, a0, n_step_r, n_step_ns, n_step_d)

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
        temp_processing_buffer = list(self.buffer) # Create a stable list for iteration

        for j in range(len(temp_processing_buffer)):
            # Define the window for the current starting experience
            current_window = deque([temp_processing_buffer[k] for k in range(j, len(temp_processing_buffer))])

            # Use helper to calculate N-step details for this specific window, using self.current_n_step
            n_step_r, n_step_ns, n_step_d, actual_n = _calculate_n_step_transition_details(current_window, self.gamma, self.current_n_step)

            if actual_n > 0: # Only if some steps were processed
                 start_exp = temp_processing_buffer[j]
                 experiences_to_return.append(Experience(start_exp.state, start_exp.action, n_step_r, n_step_ns, n_step_d))
        return experiences_to_return

    def reset(self):
        self.buffer.clear()
        self.td_error_history.clear()
        # Optional: reset n_step to base_n_step, or let it persist across episodes
        # print(f"AdaptiveNStepBuffer reset. Current N is {self.current_n_step}")
