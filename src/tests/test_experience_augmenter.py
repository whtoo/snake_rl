import unittest
import numpy as np

try:
    from src.utils import ExperienceAugmenter
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.utils import ExperienceAugmenter

class TestExperienceAugmenter(unittest.TestCase):

    def test_initialization(self):
        augmenter = ExperienceAugmenter()
        self.assertIsNotNone(augmenter.augmentation_config)
        self.assertEqual(len(augmenter.augmentation_config), 0)

        config = {'add_noise': {'scale': 0.05}}
        augmenter_with_config = ExperienceAugmenter(augmentation_config=config)
        self.assertEqual(augmenter_with_config.augmentation_config, config)

    def test_add_gaussian_noise(self):
        augmenter = ExperienceAugmenter() # Scale will be passed directly for helper

        # Test with a single image (H, W)
        state_single = np.full((84, 84), 128, dtype=np.uint8)
        scale = 10.0 # std dev for noise on uint8 image
        augmented_state_single = augmenter._add_gaussian_noise(state_single.copy(), scale)

        self.assertEqual(augmented_state_single.shape, state_single.shape)
        self.assertTrue(np.any(augmented_state_single != state_single)) # Check if changed
        self.assertTrue(np.all(augmented_state_single >= 0) and np.all(augmented_state_single <= 255))
        self.assertEqual(augmented_state_single.dtype, np.uint8)

        # Test with a stack of images (C, H, W)
        state_stack = np.full((4, 84, 84), 64, dtype=np.uint8)
        augmented_state_stack = augmenter._add_gaussian_noise(state_stack.copy(), scale)

        self.assertEqual(augmented_state_stack.shape, state_stack.shape)
        self.assertTrue(np.any(augmented_state_stack != state_stack))
        self.assertTrue(np.all(augmented_state_stack >= 0) and np.all(augmented_state_stack <= 255))
        self.assertEqual(augmented_state_stack.dtype, np.uint8)

        # Test if mean of added noise is somewhat close to 0 and std dev is somewhat close to scale
        # This is a stochastic test, so it might be flaky.
        # For a large enough image, it should hold better.
        noise_added = augmented_state_stack.astype(np.float32) - state_stack.astype(np.float32)
        # Clip noise_added because augmented_state_stack was clipped.
        # This isn't perfect for checking std dev of original noise, but gives an idea.
        # A better way would be to generate noise separately and check its properties.
        # For this test, let's just check it's different and within bounds.

        # Test with zero scale (should return original or very close to it due to float conversions)
        augmented_state_zero_scale = augmenter._add_gaussian_noise(state_stack.copy(), 0.0)
        np.testing.assert_array_almost_equal(augmented_state_zero_scale, state_stack, decimal=0)


    def test_augment_method_with_noise(self):
        config = {'add_noise': {'scale': 10.0}}
        augmenter = ExperienceAugmenter(augmentation_config=config)

        state = np.full((4, 84, 84), 128, dtype=np.uint8)
        action = 1
        reward = 1.0
        next_state = np.full((4, 84, 84), 100, dtype=np.uint8)
        done = False

        aug_s, aug_a, aug_r, aug_ns, aug_d = augmenter.augment(
            state.copy(), action, reward, next_state.copy(), done
        )

        self.assertEqual(aug_s.shape, state.shape)
        self.assertTrue(np.any(aug_s != state)) # State should be augmented
        self.assertTrue(np.all(aug_s >= 0) and np.all(aug_s <= 255))

        self.assertEqual(aug_ns.shape, next_state.shape)
        self.assertTrue(np.any(aug_ns != next_state)) # Next state should be augmented
        self.assertTrue(np.all(aug_ns >= 0) and np.all(aug_ns <= 255))

        # Action, reward, done should not be changed
        self.assertEqual(aug_a, action)
        self.assertEqual(aug_r, reward)
        self.assertEqual(aug_d, done)

    def test_augment_method_without_noise_config(self):
        augmenter = ExperienceAugmenter(augmentation_config={}) # Empty config

        state = np.full((4, 84, 84), 128, dtype=np.uint8)
        action = 1
        reward = 1.0
        next_state = np.full((4, 84, 84), 100, dtype=np.uint8)
        done = False

        aug_s, aug_a, aug_r, aug_ns, aug_d = augmenter.augment(
            state.copy(), action, reward, next_state.copy(), done
        )
        # Should return original states if 'add_noise' not in config
        np.testing.assert_array_equal(aug_s, state)
        np.testing.assert_array_equal(aug_ns, next_state)

    def test_augment_method_with_zero_scale_noise(self):
        config = {'add_noise': {'scale': 0.0}} # Zero scale
        augmenter = ExperienceAugmenter(augmentation_config=config)

        state = np.full((4, 84, 84), 128, dtype=np.uint8)
        action = 1
        reward = 1.0
        next_state = np.full((4, 84, 84), 100, dtype=np.uint8)
        done = False

        aug_s, aug_a, aug_r, aug_ns, aug_d = augmenter.augment(
            state.copy(), action, reward, next_state.copy(), done
        )
        # Should return original states due to zero scale
        np.testing.assert_array_almost_equal(aug_s, state, decimal=0)
        np.testing.assert_array_almost_equal(aug_ns, next_state, decimal=0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
