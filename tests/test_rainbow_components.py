import unittest
import torch
import torch.nn as nn
import math
import numpy as np
from collections import deque

# Add src directory to sys.path to allow direct import of modules
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import NoisyLinear, RainbowDQN
from agent import NStepBuffer, RainbowAgent, Experience, DQNAgent

# Mock Environment for Agent tests
class MockEnv:
    def __init__(self, action_space_n=6, observation_shape=(4, 84, 84)):
        self.action_space = type('ActionSpace', (), {'n': action_space_n})()
        self.observation_space = type('ObservationSpace', (), {'shape': observation_shape})()
        self.observation_space.shape = observation_shape # Ensure shape is directly accessible

    def reset(self):
        # Return a tuple (observation, info) as per Gymnasium API
        return torch.zeros(self.observation_space.shape), {}

    def step(self, action):
        # Return (observation, reward, done, truncated, info)
        return torch.zeros(self.observation_space.shape), 0.0, False, False, {}

class TestNoisyLinear(unittest.TestCase):
    def test_noisy_linear_initialization(self):
        in_features, out_features = 10, 5
        layer = NoisyLinear(in_features, out_features)

        self.assertIsInstance(layer.mu_weight, nn.Parameter)
        self.assertEqual(layer.mu_weight.shape, (out_features, in_features))
        self.assertIsInstance(layer.sigma_weight, nn.Parameter)
        self.assertEqual(layer.sigma_weight.shape, (out_features, in_features))

        self.assertIsInstance(layer.mu_bias, nn.Parameter)
        self.assertEqual(layer.mu_bias.shape, (out_features,))
        self.assertIsInstance(layer.sigma_bias, nn.Parameter)
        self.assertEqual(layer.sigma_bias.shape, (out_features,))

        self.assertTrue(hasattr(layer, 'epsilon_weight'))
        self.assertEqual(layer.epsilon_weight.shape, (out_features, in_features))
        self.assertTrue(hasattr(layer, 'epsilon_bias'))
        self.assertEqual(layer.epsilon_bias.shape, (out_features,))

    def test_noisy_linear_sample_noise_factorised(self):
        in_features, out_features = 10, 5
        layer = NoisyLinear(in_features, out_features, factorised=True)

        initial_epsilon_weight = layer.epsilon_weight.clone()
        initial_epsilon_bias = layer.epsilon_bias.clone()

        layer.sample_noise()
        self.assertFalse(torch.equal(initial_epsilon_weight, layer.epsilon_weight))
        self.assertFalse(torch.equal(initial_epsilon_bias, layer.epsilon_bias))
        self.assertEqual(layer.epsilon_weight.shape, (out_features, in_features))
        self.assertEqual(layer.epsilon_bias.shape, (out_features,))

        current_epsilon_weight = layer.epsilon_weight.clone()
        layer.sample_noise()
        self.assertFalse(torch.equal(current_epsilon_weight, layer.epsilon_weight))

    def test_noisy_linear_sample_noise_independent(self):
        in_features, out_features = 10, 5
        layer = NoisyLinear(in_features, out_features, factorised=False)

        initial_epsilon_weight = layer.epsilon_weight.clone()
        initial_epsilon_bias = layer.epsilon_bias.clone()

        layer.sample_noise()
        self.assertFalse(torch.equal(initial_epsilon_weight, layer.epsilon_weight))
        self.assertFalse(torch.equal(initial_epsilon_bias, layer.epsilon_bias))
        self.assertEqual(layer.epsilon_weight.shape, (out_features, in_features))
        self.assertEqual(layer.epsilon_bias.shape, (out_features,))

        current_epsilon_weight = layer.epsilon_weight.clone()
        layer.sample_noise()
        self.assertFalse(torch.equal(current_epsilon_weight, layer.epsilon_weight))

    def test_noisy_linear_forward_pass_train_eval(self):
        in_features, out_features = 10, 5
        layer = NoisyLinear(in_features, out_features)
        dummy_input = torch.randn(1, in_features)

        # Train mode
        layer.train()
        layer.sample_noise() # Sample noise for the first forward pass
        output_train1 = layer(dummy_input)
        self.assertEqual(output_train1.shape, (1, out_features))

        # To ensure output changes, sample noise again
        layer.sample_noise()
        output_train2 = layer(dummy_input)
        self.assertFalse(torch.equal(output_train1, output_train2))

        # Eval mode
        layer.eval()
        output_eval1 = layer(dummy_input)
        self.assertEqual(output_eval1.shape, (1, out_features))

        output_eval2 = layer(dummy_input)
        self.assertTrue(torch.equal(output_eval1, output_eval2))

        # Check that eval output is different from training output (mu vs mu + sigma*eps)
        # This is highly probable but not strictly guaranteed if sigma or epsilon is zero.
        # Given initialization, it's very likely different.
        layer.train() # Back to train to use noise
        layer.sample_noise()
        output_train_compare = layer(dummy_input)
        self.assertFalse(torch.equal(output_eval1, output_train_compare), "Eval and Train outputs should differ if noise is active")

class TestNStepBuffer(unittest.TestCase):
    def test_nstep_buffer_add_and_fill(self):
        buffer = NStepBuffer(n_step=3, gamma=0.99)
        s1, a1, r1, s2, d1 = ("s1", 1, 1.0, "s2", False)
        s2_t, a2, r2, s3, d2 = ("s2", 2, 2.0, "s3", False) # s2_t to distinguish from s2 in first tuple
        s3_t, a3, r3, s4, d3 = ("s3", 3, 3.0, "s4", False)

        self.assertIsNone(buffer.add(s1, a1, r1, s2, d1))
        self.assertEqual(len(buffer.buffer), 1)
        self.assertIsNone(buffer.add(s2_t, a2, r2, s3, d2))
        self.assertEqual(len(buffer.buffer), 2)

        n_step_exp = buffer.add(s3_t, a3, r3, s4, d3)
        self.assertIsNotNone(n_step_exp)
        self.assertEqual(len(buffer.buffer), 3) # Deque is full

        expected_reward = r1 + 0.99 * r2 + (0.99**2) * r3
        self.assertEqual(n_step_exp.state, s1)
        self.assertEqual(n_step_exp.action, a1)
        self.assertAlmostEqual(n_step_exp.reward, expected_reward)
        self.assertEqual(n_step_exp.next_state, s4) # Next state from the last element
        self.assertEqual(n_step_exp.done, d3)       # Done from the last element

    def test_nstep_buffer_episode_termination_within_nstep(self):
        buffer = NStepBuffer(n_step=3, gamma=0.99)
        s1, a1, r1, s2, d1 = ("s1", 1, 1.0, "s2", False)
        s2_t, a2, r2, s3, d2_terminal = ("s2", 2, 2.0, "s3", True) # Episode ends here
        s3_t, a3, r3, s4, d3 = ("s3", 3, 3.0, "s4", False) # This would be after termination

        buffer.add(s1, a1, r1, s2, d1)
        buffer.add(s2_t, a2, r2, s3, d2_terminal) # Buffer not full yet

        # When the buffer becomes full with the third transition
        n_step_exp = buffer.add(s3_t, a3, r3, s4, d3)
        self.assertIsNotNone(n_step_exp)

        # The n-step transition should be calculated up to the point of termination (t2)
        # State, Action from t1
        # Reward = r1 + gamma * r2
        # Next state = s3 (from t2), Done = True (from t2)
        expected_reward = r1 + 0.99 * r2
        self.assertEqual(n_step_exp.state, s1)
        self.assertEqual(n_step_exp.action, a1)
        self.assertAlmostEqual(n_step_exp.reward, expected_reward)
        self.assertEqual(n_step_exp.next_state, s3) # next_state from the terminal transition t2
        self.assertTrue(n_step_exp.done)            # done from the terminal transition t2

    def test_nstep_buffer_get_last_n_step(self):
        buffer = NStepBuffer(n_step=3, gamma=0.99)
        s1, a1, r1, ns1, d1_false = ("s1", 1, 1.0, "ns1", False)
        s2, a2, r2, ns2, d2_false = ("s2", 2, 2.0, "ns2", False)

        buffer.add(s1, a1, r1, ns1, d1_false)
        buffer.add(s2, a2, r2, ns2, d2_false)

        experiences = buffer.get_last_n_step()
        self.assertEqual(len(experiences), 2)

        # Exp1: starts at (s1, a1)
        # Reward: r1 + gamma*r2
        # Next_state: ns2 (from last actual experience s2)
        # Done: d2_false (from last actual experience s2)
        exp1 = experiences[0]
        self.assertEqual(exp1.state, s1)
        self.assertEqual(exp1.action, a1)
        self.assertAlmostEqual(exp1.reward, r1 + 0.99 * r2)
        self.assertEqual(exp1.next_state, ns2)
        self.assertEqual(exp1.done, d2_false) # Corrected: should be from last exp in buffer

        # Exp2: starts at (s2, a2)
        # Reward: r2
        # Next_state: ns2
        # Done: d2_false
        exp2 = experiences[1]
        self.assertEqual(exp2.state, s2)
        self.assertEqual(exp2.action, a2)
        self.assertAlmostEqual(exp2.reward, r2)
        self.assertEqual(exp2.next_state, ns2)
        self.assertEqual(exp2.done, d2_false) # Corrected: should be from last exp in buffer

        # Test with a terminal state in the buffer
        buffer.reset()
        s1, a1, r1, ns1, d1_false = ("s1", 1, 1.0, "ns1", False)
        s2, a2, r2, ns2, d2_true = ("s2", 2, 2.0, "ns2", True) # Terminal
        buffer.add(s1, a1, r1, ns1, d1_false)
        buffer.add(s2, a2, r2, ns2, d2_true)

        experiences_term = buffer.get_last_n_step()
        self.assertEqual(len(experiences_term), 2)

        exp1_term = experiences_term[0] # s1, a1, r1 + g*r2, ns2, True
        self.assertEqual(exp1_term.state, s1)
        self.assertAlmostEqual(exp1_term.reward, r1 + 0.99*r2)
        self.assertEqual(exp1_term.next_state, ns2)
        self.assertTrue(exp1_term.done)

        exp2_term = experiences_term[1] # s2, a2, r2, ns2, True
        self.assertEqual(exp2_term.state, s2)
        self.assertAlmostEqual(exp2_term.reward, r2)
        self.assertEqual(exp2_term.next_state, ns2)
        self.assertTrue(exp2_term.done)


    def test_nstep_buffer_reset(self):
        buffer = NStepBuffer(n_step=3, gamma=0.99)
        buffer.add("s1", 1, 1.0, "s2", False)
        self.assertEqual(len(buffer.buffer), 1)
        buffer.reset()
        self.assertEqual(len(buffer.buffer), 0)

class TestRainbowDQNModel(unittest.TestCase):
    def setUp(self):
        self.input_shape = (4, 84, 84) # C, H, W
        self.n_actions = 6
        self.batch_size = 2
        self.dummy_input = torch.randn(self.batch_size, *self.input_shape)

    def test_rainbow_dqn_scalar_output(self):
        model = RainbowDQN(self.input_shape, self.n_actions, use_distributional=False, use_noisy=False)
        model.eval() # Set to eval mode
        output = model(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.n_actions))

    def test_rainbow_dqn_distributional_output(self):
        n_atoms = 51
        model = RainbowDQN(self.input_shape, self.n_actions, n_atoms=n_atoms, use_distributional=True, use_noisy=False)
        model.eval() # Set to eval mode
        output = model(self.dummy_input) # Output should be probabilities
        self.assertEqual(output.shape, (self.batch_size, self.n_actions, n_atoms))
        # Check if probabilities sum to 1 along the atoms dimension
        sum_probs = output.sum(dim=2)
        self.assertTrue(torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=1e-6))

    def test_rainbow_dqn_noisy_layers_used(self):
        model = RainbowDQN(self.input_shape, self.n_actions, use_noisy=True, use_distributional=False)
        # Check a linear layer in value stream (e.g., the last one)
        self.assertIsInstance(model.value_stream[-1], NoisyLinear)
        # Check a linear layer in advantage stream
        self.assertIsInstance(model.advantage_stream[-1], NoisyLinear)

        model_dist = RainbowDQN(self.input_shape, self.n_actions, use_noisy=True, use_distributional=True, n_atoms=51)
        self.assertIsInstance(model_dist.value_stream[-1], NoisyLinear)
        self.assertIsInstance(model_dist.advantage_stream[-1], NoisyLinear)


    def test_rainbow_dqn_sample_noise(self):
        model = RainbowDQN(self.input_shape, self.n_actions, use_noisy=True, use_distributional=False)

        # Get a NoisyLinear layer to check (e.g., first noisy layer in value stream)
        noisy_layer = None
        for layer in model.value_stream:
            if isinstance(layer, NoisyLinear):
                noisy_layer = layer
                break
        self.assertIsNotNone(noisy_layer, "No NoisyLinear layer found in value_stream for testing sample_noise")

        initial_epsilon_weight = noisy_layer.epsilon_weight.clone()
        model.sample_noise() # This should sample noise for all NoisyLinear layers
        self.assertFalse(torch.equal(initial_epsilon_weight, noisy_layer.epsilon_weight))


class TestRainbowAgent(unittest.TestCase):
    def setUp(self):
        self.input_shape = (4, 84, 84)
        self.n_actions = 3
        self.env = MockEnv(action_space_n=self.n_actions, observation_shape=self.input_shape)
        self.device = torch.device("cpu")

        # Common args for agent
        self.agent_args = {
            "env": self.env,
            "device": self.device,
            "buffer_size": 10000,
            "batch_size": 32,
            "gamma": 0.99,
            "lr": 1e-4,
            "target_update": 1000,
            "prioritized_replay": False # Keep simple for these tests
        }

    def _create_models(self, use_noisy=False, use_distributional=False, n_atoms=51, v_min=-10, v_max=10):
        model = RainbowDQN(
            input_shape=self.input_shape,
            n_actions=self.n_actions,
            use_noisy=use_noisy,
            use_distributional=use_distributional,
            n_atoms=n_atoms, v_min=v_min, v_max=v_max
        ).to(self.device)
        target_model = RainbowDQN(
            input_shape=self.input_shape,
            n_actions=self.n_actions,
            use_noisy=use_noisy,
            use_distributional=use_distributional,
            n_atoms=n_atoms, v_min=v_min, v_max=v_max
        ).to(self.device)
        target_model.load_state_dict(model.state_dict())
        return model, target_model

    def test_rainbow_agent_initialization_standard(self):
        model, target_model = self._create_models()
        agent = RainbowAgent(model, target_model, **self.agent_args,
                             n_step=3, use_noisy=False, use_distributional=False)
        self.assertFalse(agent.use_noisy)
        self.assertFalse(agent.use_distributional)
        self.assertIsNotNone(agent.n_step_buffer)
        self.assertEqual(agent.n_step, 3)
        # Epsilon should be default from DQNAgent
        self.assertEqual(agent.epsilon_start, DQNAgent.epsilon_start if 'epsilon_start' not in self.agent_args else self.agent_args['epsilon_start'])


    def test_rainbow_agent_initialization_noisy(self):
        model, target_model = self._create_models(use_noisy=True)
        agent = RainbowAgent(model, target_model, **self.agent_args,
                             n_step=3, use_noisy=True, use_distributional=False)
        self.assertTrue(agent.use_noisy)
        # Epsilon should be disabled for noisy networks
        self.assertEqual(agent.epsilon_start, 0.0)
        self.assertEqual(agent.epsilon_final, 0.0)

    def test_rainbow_agent_initialization_distributional(self):
        n_atoms, v_min, v_max = 51, -10, 10
        model, target_model = self._create_models(use_distributional=True, n_atoms=n_atoms, v_min=v_min, v_max=v_max)
        agent = RainbowAgent(model, target_model, **self.agent_args,
                             n_step=3, use_noisy=False, use_distributional=True,
                             n_atoms=n_atoms, v_min=v_min, v_max=v_max)
        self.assertTrue(agent.use_distributional)
        self.assertEqual(agent.n_atoms, n_atoms)
        self.assertEqual(agent.v_min, v_min)
        self.assertEqual(agent.v_max, v_max)
        self.assertTrue(hasattr(agent, 'support'))
        self.assertTrue(hasattr(agent, 'delta_z'))
        self.assertEqual(agent.support.shape, (n_atoms,))

    def test_rainbow_agent_select_action_noisy(self):
        model, target_model = self._create_models(use_noisy=True, use_distributional=False)
        agent = RainbowAgent(model, target_model, **self.agent_args, use_noisy=True)

        state, _ = self.env.reset()
        # With noisy nets, action selection should be greedy w.r.t. noisy Q-values
        # We are mostly testing that it runs without epsilon logic
        action = agent.select_action(state)
        self.assertIsInstance(action, int)

    def test_rainbow_agent_select_action_distributional_noisy(self):
        n_atoms, v_min, v_max = 51, -10, 10
        model, target_model = self._create_models(use_noisy=True, use_distributional=True,
                                               n_atoms=n_atoms, v_min=v_min, v_max=v_max)
        agent = RainbowAgent(model, target_model, **self.agent_args,
                             use_noisy=True, use_distributional=True,
                             n_atoms=n_atoms, v_min=v_min, v_max=v_max)
        state, _ = self.env.reset()
        action = agent.select_action(state)
        self.assertIsInstance(action, int)
        # Further checks would involve inspecting model output and expected Q-values

    def test_rainbow_agent_store_experience(self):
        model, target_model = self._create_models()
        agent = RainbowAgent(model, target_model, **self.agent_args, n_step=3)

        # Mock buffers
        agent.n_step_buffer = unittest.mock.Mock(spec=NStepBuffer)
        agent.memory = unittest.mock.Mock(spec=agent.memory) # Assuming PrioritizedReplayBuffer or ReplayBuffer

        s, a, r, ns, d = ("s", 1, 1.0, "ns", False)
        n_step_return_exp = Experience("s0", 0, 10.0, "ns_n", False)

        # Case 1: n_step_buffer.add returns None
        agent.n_step_buffer.add.return_value = None
        agent.store_experience(s,a,r,ns,d)
        agent.n_step_buffer.add.assert_called_once_with(s,a,r,ns,d)
        agent.memory.push.assert_not_called()

        # Case 2: n_step_buffer.add returns an experience
        agent.n_step_buffer.add.reset_mock()
        agent.memory.push.reset_mock()
        agent.n_step_buffer.add.return_value = n_step_return_exp
        agent.store_experience(s,a,r,ns,d)
        agent.n_step_buffer.add.assert_called_once_with(s,a,r,ns,d)
        agent.memory.push.assert_called_once_with(*n_step_return_exp)

        # Case 3: Done is True
        agent.n_step_buffer.add.reset_mock()
        agent.memory.push.reset_mock()
        agent.n_step_buffer.add.return_value = n_step_return_exp # Suppose it also returns one last full exp

        remaining_exps = [Experience("s_rem1", 1, 1.0, "ns_rem1", True), Experience("s_rem2", 2, 2.0, "ns_rem2", True)]
        agent.n_step_buffer.get_last_n_step.return_value = remaining_exps
        agent.n_step_buffer.reset.return_value = None

        agent.store_experience(s,a,r,ns,True) # done = True
        agent.n_step_buffer.add.assert_called_once_with(s,a,r,ns,True)
        agent.memory.push.assert_any_call(*n_step_return_exp) # For the one returned by add
        agent.n_step_buffer.get_last_n_step.assert_called_once()
        agent.memory.push.assert_any_call(*remaining_exps[0])
        agent.memory.push.assert_any_call(*remaining_exps[1])
        self.assertEqual(agent.memory.push.call_count, 3) # 1 from add, 2 from remaining
        agent.n_step_buffer.reset.assert_called_once()


    def test_rainbow_agent_update_model_call_order_noisy(self):
        # Use MagicMock for more flexibility with special methods if needed, and mock specific methods.
        model_mock = unittest.mock.MagicMock()
        target_model_mock = unittest.mock.MagicMock()

        # Mock methods needed by DQNAgent.__init__ for model_mock
        dummy_param = nn.Parameter(torch.randn(2,2))
        model_mock.parameters.return_value = [dummy_param]
        model_mock.to.return_value = model_mock # .to() method

        # Mock methods needed for target_model_mock
        target_model_mock.to.return_value = target_model_mock
        target_model_mock.load_state_dict.return_value = None # Mocked method
        target_model_mock.eval.return_value = None # Mocked method

        agent = RainbowAgent(model_mock, target_model_mock, **self.agent_args, use_noisy=True)

        # Mock memory to control batch sampling
        # Use MagicMock for agent.memory to correctly mock __len__
        agent.memory = unittest.mock.MagicMock(spec=agent.memory)
        dummy_experience_batch = (
            torch.randn(self.agent_args['batch_size'], *self.input_shape), # states
            torch.randint(0, self.n_actions, (self.agent_args['batch_size'], 1)), # actions
            torch.randn(self.agent_args['batch_size'], 1), # rewards
            torch.randn(self.agent_args['batch_size'], *self.input_shape), # next_states
            torch.randint(0, 2, (self.agent_args['batch_size'], 1)).float() # dones
        )
        if agent.prioritized_replay: # if it was true
             dummy_experience_batch += (torch.arange(self.agent_args['batch_size']), torch.ones(self.agent_args['batch_size'],1))

        agent.memory.sample.return_value = dummy_experience_batch

        # Mock loss computation to prevent actual computation
        agent._compute_standard_loss = unittest.mock.Mock(return_value=torch.tensor(0.1, requires_grad=True))

        # Fill memory enough to trigger update
        agent.memory.__len__.return_value = self.agent_args['batch_size'] * 2

        agent.update_model()

        model_mock.sample_noise.assert_called_once()
        target_model_mock.sample_noise.assert_called_once()
        agent._compute_standard_loss.assert_called_once() # Or distributional if that was configured

    # More detailed tests for _compute_standard_loss and _compute_distributional_loss
    # would require careful setup of mock model outputs and expected values.
    # For now, this covers the basic call structure.

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# Note: The RainbowAgent loss function tests are complex and would require significant mocking
# or very precise input/output calculations. The current tests focus on initialization,
# method calls, and interactions between components.
# Adding detailed loss tests would be a good extension if time permits.
# For example, _compute_standard_loss is relatively testable.
# _compute_distributional_loss is much harder due to the projection step.

class TestStandardLoss(unittest.TestCase):
    def setUp(self):
        self.input_shape = (4, 84, 84) # Corrected: Use a valid shape for CNN
        self.n_actions = 2
        self.n_atoms = 3
        self.v_min = 0
        self.v_max = 2
        self.batch_size = 2
        self.device = torch.device("cpu")

        # For loss computation tests, we often don't need full model instantiation if forward is mocked.
        # However, RainbowAgent init requires model and target_model.
        # So, we instantiate them but will mock their forward methods in the test.
        self.model = RainbowDQN(self.input_shape, self.n_actions, use_distributional=False, use_noisy=False).to(self.device)
        self.target_model = RainbowDQN(self.input_shape, self.n_actions, use_distributional=False, use_noisy=False).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.env = MockEnv(action_space_n=self.n_actions, observation_shape=self.input_shape)
        self.agent = RainbowAgent(
            model=self.model, target_model=self.target_model, env=self.env, device=self.device,
            batch_size=self.batch_size, n_step=1, gamma=0.9, prioritized_replay=False,
            use_distributional=False # Explicitly standard
        )

    def test_compute_standard_loss(self):
        # Create dummy batch data
        states = torch.randn(self.batch_size, *self.input_shape).to(self.device)
        actions = torch.tensor([[0], [1]]).long().to(self.device) # Batch_size x 1
        rewards = torch.tensor([[1.0], [1.0]]).float().to(self.device) # Batch_size x 1
        next_states = torch.randn(self.batch_size, *self.input_shape).to(self.device)
        dones = torch.tensor([[0.0], [0.0]]).float().to(self.device) # Batch_size x 1
        weights = torch.ones(self.batch_size, 1).to(self.device)

        # Mock model outputs for predictability
        # Q(s,a)
        self.agent.model.forward = unittest.mock.Mock(return_value=torch.tensor([[10.0, 20.0], [5.0, 15.0]]).to(self.device))
        # Q_target(s',a')
        self.agent.target_model.forward = unittest.mock.Mock(return_value=torch.tensor([[100.0, 200.0], [50.0, 150.0]]).to(self.device))
        # Q_model(s') for Double DQN action selection
        # This mock needs to be for the self.model called on next_states
        # To simplify, let's assume model(next_states) returns values leading to specific next_actions
        # Let's say model(next_states).max(1)[1] selects action 0 for first sample, action 1 for second
        mock_next_q_main_model = torch.tensor([[100.0, 0.0], [0.0, 150.0]]).to(self.device) # Max actions are 0, 1

        # Need to make sure the *correct* mock is called.
        # The agent calls self.model(states) and self.model(next_states)
        # and self.target_model(next_states)

        # Let current_q_values be based on actions [0, 1]
        # q_s_a0 = 10.0, q_s_a1 = 15.0
        # gather will pick these: [[10.0], [15.0]]

        # Target calculation:
        # model(next_states) -> mock_next_q_main_model -> next_actions = [[0],[1]]
        # target_model(next_states) -> [[100,200],[50,150]]
        # target_model(next_states).gather(1, next_actions) -> [[100],[150]]
        # target_q = rewards + (1-dones) * gamma^n_step * gathered_target_q
        # target_q_0 = 1.0 + 0.9 * 100.0 = 91.0
        # target_q_1 = 1.0 + 0.9 * 150.0 = 136.0
        # targets = [[91.0], [136.0]]

        # td_errors = abs([[10.0], [15.0]] - [[91.0], [136.0]]) = abs([[-81.0], [-121.0]]) = [[81.0], [121.0]]
        # loss = (81.0 + 121.0) / 2 = 202.0 / 2 = 101.0

        # Setup the multiple model calls properly
        def side_effect_model(*args, **kwargs):
            if torch.equal(args[0], states): # Call with states
                return torch.tensor([[10.0, 20.0], [5.0, 15.0]]).to(self.device)
            elif torch.equal(args[0], next_states): # Call with next_states for Double DQN
                return mock_next_q_main_model
            raise ValueError("Unexpected input to model mock")

        self.agent.model.forward = unittest.mock.MagicMock(side_effect=side_effect_model)
        self.agent.target_model.forward = unittest.mock.MagicMock(return_value=torch.tensor([[100.0, 200.0], [50.0, 150.0]]).to(self.device))

        loss = self.agent._compute_standard_loss(states, actions, rewards, next_states, dones, weights)
        self.assertAlmostEqual(loss.item(), 101.0, places=5)

# To run tests from within a Python script if not using pytest runner:
# if __name__ == '__main__':
#     unittest.main()

# Note: The `run_training.py` script uses `train()` from `src.train`.
# This test file should be placed in `tests/` directory.
# To run: `python -m unittest tests.test_rainbow_components` from the project root.
# Or if using pytest: `pytest tests/test_rainbow_components.py`
# Make sure __init__.py exists in tests folder if using module discovery.
# For these tests, Experience namedtuple is used by NStepBuffer.
# DQNAgent.epsilon_start needs to be accessible for one of the tests.
# It's better to define it as a class variable or make agent_args more complete.
DQNAgent.epsilon_start = 1.0 # Define for test access
DQNAgent.epsilon_final = 0.01
DQNAgent.epsilon_decay = 10000


class TestDistributionalLoss(unittest.TestCase):
    def setUp(self):
        self.input_shape = (4, 84, 84)
        self.n_actions = 2
        self.batch_size = 2 # Keep batch size small for manual calculation
        self.device = torch.device("cpu")

        # Distributional parameters
        self.n_atoms = 3 # Small n_atoms for easier manual calculation
        self.v_min = 0.0
        self.v_max = 2.0 # delta_z will be (2-0)/(3-1) = 1.0. Support: [0, 1, 2]

        self.model = RainbowDQN(
            self.input_shape, self.n_actions,
            n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max,
            use_distributional=True, use_noisy=False
        ).to(self.device)

        self.target_model = RainbowDQN(
            self.input_shape, self.n_actions,
            n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max,
            use_distributional=True, use_noisy=False
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.env = MockEnv(action_space_n=self.n_actions, observation_shape=self.input_shape)
        self.agent = RainbowAgent(
            model=self.model, target_model=self.target_model, env=self.env, device=self.device,
            batch_size=self.batch_size, n_step=1, gamma=0.9, prioritized_replay=False,
            use_distributional=True, n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max
        )
        self.support = self.agent.support
        self.delta_z = self.agent.delta_z

    def test_compute_distributional_loss(self):
        # Dummy batch data
        states = torch.randn(self.batch_size, *self.input_shape).to(self.device)
        actions = torch.tensor([[0], [1]]).long().to(self.device) # Batch_size x 1, actions for each sample
        rewards = torch.tensor([[1.0], [0.5]]).float().to(self.device)
        next_states = torch.randn(self.batch_size, *self.input_shape).to(self.device)
        # For one sample, let done=True to test that path
        dones = torch.tensor([[0.0], [1.0]]).float().to(self.device)
        weights = torch.ones(self.batch_size, 1).to(self.device)

        # Mock model outputs (probabilities from RainbowDQN's softmax)
        # Current model output for selected actions (batch_size, n_atoms)
        # These are the P(s,a) that agent._compute_distributional_loss will get after gather
        # and then apply .log() to. So, these should be probabilities.
        # model(states) will return (batch_size, n_actions, n_atoms)
        # Let's mock the full output of self.model(states)
        # Sample 0, action 0: dist [0.1, 0.6, 0.3] -> log [log(0.1), log(0.6), log(0.3)]
        # Sample 1, action 1: dist [0.4, 0.4, 0.2] -> log [log(0.4), log(0.4), log(0.2)]
        mock_current_dist_all_actions = torch.zeros(self.batch_size, self.n_actions, self.n_atoms, device=self.device)
        mock_current_dist_all_actions[0, 0, :] = torch.tensor([0.1, 0.6, 0.3])
        mock_current_dist_all_actions[0, 1, :] = torch.tensor([0.2, 0.2, 0.6]) # Other action for sample 0
        mock_current_dist_all_actions[1, 1, :] = torch.tensor([0.4, 0.4, 0.2])
        mock_current_dist_all_actions[1, 0, :] = torch.tensor([0.7, 0.1, 0.2]) # Other action for sample 1

        # Mock for Double DQN action selection on next_states: self.model(next_states)
        # Sample 0 (next_state): action 1 is best (e.g., Q-value for action 1 is higher)
        #   Q_action0 = (0.1*0 + 0.2*1 + 0.7*2) = 1.6
        #   Q_action1 = (0.6*0 + 0.1*1 + 0.3*2) = 0.7. So action 0 is best for sample 0.
        # Sample 1 (next_state): action 0 is best
        #   Q_action0 = (0.5*0 + 0.3*1 + 0.2*2) = 0.7
        #   Q_action1 = (0.2*0 + 0.4*1 + 0.4*2) = 1.2. So action 1 is best for sample 1.
        mock_next_dist_model = torch.zeros(self.batch_size, self.n_actions, self.n_atoms, device=self.device)
        mock_next_dist_model[0, 0, :] = torch.tensor([0.1, 0.2, 0.7]) # Q = 1.6
        mock_next_dist_model[0, 1, :] = torch.tensor([0.6, 0.1, 0.3]) # Q = 0.7
        mock_next_dist_model[1, 0, :] = torch.tensor([0.5, 0.3, 0.2]) # Q = 0.7
        mock_next_dist_model[1, 1, :] = torch.tensor([0.2, 0.4, 0.4]) # Q = 1.2
        # Expected next_actions from Double DQN: [0, 1]

        # Mock for target network evaluation on next_states: self.target_model(next_states)
        # These are the distributions P_target(s_next, a_selected_by_model)
        # For sample 0, next_action is 0: use dist_target_s0_a0
        # For sample 1, next_action is 1: use dist_target_s1_a1
        mock_next_dist_target_net_all_actions = torch.zeros(self.batch_size, self.n_actions, self.n_atoms, device=self.device)
        dist_target_s0_a0 = torch.tensor([0.2, 0.3, 0.5]) # For sample 0, next_action 0
        dist_target_s1_a1 = torch.tensor([0.3, 0.4, 0.3]) # For sample 1, next_action 1
        mock_next_dist_target_net_all_actions[0, 0, :] = dist_target_s0_a0
        mock_next_dist_target_net_all_actions[1, 1, :] = dist_target_s1_a1
        # Fill others just in case, though not strictly needed if gather is correct
        mock_next_dist_target_net_all_actions[0, 1, :] = torch.rand(self.n_atoms); mock_next_dist_target_net_all_actions[0,1,:] /= mock_next_dist_target_net_all_actions[0,1,:].sum()
        mock_next_dist_target_net_all_actions[1, 0, :] = torch.rand(self.n_atoms); mock_next_dist_target_net_all_actions[1,0,:] /= mock_next_dist_target_net_all_actions[1,0,:].sum()


        def side_effect_model_forward(*args, **kwargs):
            if torch.equal(args[0], states): return mock_current_dist_all_actions
            if torch.equal(args[0], next_states): return mock_next_dist_model
            raise ValueError("Unexpected input to model mock forward")

        def side_effect_target_model_forward(*args, **kwargs):
            if torch.equal(args[0], next_states): return mock_next_dist_target_net_all_actions
            raise ValueError("Unexpected input to target_model mock forward")

        self.agent.model.forward = unittest.mock.MagicMock(side_effect=side_effect_model_forward)
        self.agent.target_model.forward = unittest.mock.MagicMock(side_effect=side_effect_target_model_forward)

        # --- Manual Calculation of Expected Target Distribution ---
        # Support: [0.0, 1.0, 2.0], delta_z = 1.0

        # Sample 0: reward=1.0, done=0.0, next_action_selected=0, P(s_next,a=0)=[0.2, 0.3, 0.5]
        # Tz = reward + gamma * support = 1.0 + 0.9 * [0, 1, 2] = [1.0, 1.9, 2.8]
        # Clamped Tz = [1.0, 1.9, 2.0] (since v_max=2.0)
        # b = (Tz - v_min) / delta_z = ([1.0, 1.9, 2.0] - 0.0) / 1.0 = [1.0, 1.9, 2.0]
        # l = floor(b) = [1, 1, 2]
        # u = ceil(b) = [1, 2, 2]
        # Projected_dist_0:
        #   For atom j=0 (original prob p_j=0.2, Tz_j=1.0, l_j=1, u_j=1):
        #     m_0[1] += 0.2 * 1 = 0.2 (since l_j==u_j)
        #   For atom j=1 (original prob p_j=0.3, Tz_j=1.9, l_j=1, u_j=2):
        #     m_0[1] += 0.3 * (2 - 1.9) = 0.3 * 0.1 = 0.03
        #     m_0[2] += 0.3 * (1.9 - 1) = 0.3 * 0.9 = 0.27
        #   For atom j=2 (original prob p_j=0.5, Tz_j=2.0, l_j=2, u_j=2):
        #     m_0[2] += 0.5 * 1 = 0.5
        # target_dist_0 = [0, 0.2 + 0.03, 0.27 + 0.5] = [0, 0.23, 0.77]

        # Sample 1: reward=0.5, done=1.0
        # Tz = reward + (1-done) * gamma * support = 0.5 + 0 = 0.5
        # Clamped Tz = [0.5, 0.5, 0.5] (scalar applied to all atoms in projection)
        # b = ([0.5, 0.5, 0.5] - 0.0) / 1.0 = [0.5, 0.5, 0.5]
        # l = floor(b) = [0, 0, 0]
        # u = ceil(b) = [1, 1, 1]
        # next_dist_target for sample 1 (action 1) is [0.3, 0.4, 0.3]
        # Projected_dist_1:
        #   For atom j=0 (original prob p_j=0.3, Tz_j=0.5, l_j=0, u_j=1):
        #     m_1[0] += 0.3 * (1 - 0.5) = 0.3 * 0.5 = 0.15
        #     m_1[1] += 0.3 * (0.5 - 0) = 0.3 * 0.5 = 0.15
        #   For atom j=1 (original prob p_j=0.4, Tz_j=0.5, l_j=0, u_j=1):
        #     m_1[0] += 0.4 * (1 - 0.5) = 0.4 * 0.5 = 0.20
        #     m_1[1] += 0.4 * (0.5 - 0) = 0.4 * 0.5 = 0.20
        #   For atom j=2 (original prob p_j=0.3, Tz_j=0.5, l_j=0, u_j=1):
        #     m_1[0] += 0.3 * (1 - 0.5) = 0.3 * 0.5 = 0.15
        #     m_1[1] += 0.3 * (0.5 - 0) = 0.3 * 0.5 = 0.15
        # target_dist_1 = [0.15+0.20+0.15, 0.15+0.20+0.15, 0] = [0.5, 0.5, 0.0]

        expected_target_dist = torch.tensor([
            [0.0, 0.23, 0.77],
            [0.5, 0.5, 0.0]
        ], device=self.device)

        # --- Manual Loss Calculation ---
        # log_probs_selected for sample 0 (action 0): log([0.1, 0.6, 0.3])
        # log_probs_selected for sample 1 (action 1): log([0.4, 0.4, 0.2])
        log_p_s0_a0 = torch.log(torch.tensor([0.1, 0.6, 0.3])).to(self.device)
        log_p_s1_a1 = torch.log(torch.tensor([0.4, 0.4, 0.2])).to(self.device)

        loss0 = -(expected_target_dist[0] * log_p_s0_a0).sum() # KL Div
        loss1 = -(expected_target_dist[1] * log_p_s1_a1).sum()
        expected_loss = (loss0 + loss1) / 2.0 # Averaged over batch, weights are 1.0

        actual_loss = self.agent._compute_distributional_loss(states, actions, rewards, next_states, dones, weights)

        self.assertAlmostEqual(actual_loss.item(), expected_loss.item(), places=5)
