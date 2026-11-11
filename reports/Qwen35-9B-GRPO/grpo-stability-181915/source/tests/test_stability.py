import math
import unittest

import torch

from src.stability import assess_batch, make_schedule, stop_reason, tis_weights


class StabilityTests(unittest.TestCase):
    def test_schedule_is_seeded_and_disjoint(self):
        keys = list(map(str, range(200)))
        result = make_schedule(keys, seed=42, heldout=32)
        self.assertEqual(result, make_schedule(list(reversed(keys)), seed=42, heldout=32))
        self.assertEqual([len(x) for x in result['training']], [16] * 10)
        self.assertEqual(len(set(sum(result['training'], []) + result['heldout'])), 192)
        with self.assertRaises(ValueError):
            make_schedule(keys[:100], seed=42)
        with self.assertRaises(ValueError):
            make_schedule(keys + keys[:1], seed=42)

    def test_tis_scales_gradient_without_differentiating_weight(self):
        current = torch.tensor([0., 0., 0.], requires_grad=True)
        sampled = torch.tensor([-math.log(4), 0., math.log(2)])
        weights, clipped, errors = tis_weights(current, sampled)
        self.assertFalse(weights.requires_grad)
        torch.testing.assert_close(weights, torch.tensor([2., 1., .5]))
        self.assertEqual(clipped.tolist(), [True, False, False])
        torch.testing.assert_close(errors, sampled.abs())
        loss = -((current - current.detach()).exp() * weights).sum()
        loss.backward()
        torch.testing.assert_close(current.grad, -weights)

    def test_api_failure_does_not_become_a_zero_reward(self):
        rewards = [1., 0., None, 1., 1., 1.]
        advantages, metrics = assess_batch(['a'] * 3 + ['b'] * 3, rewards)
        self.assertEqual(metrics['scored_trajectories'], 5)
        self.assertEqual(metrics['group_mixed'], 1)
        self.assertEqual(metrics['group_all_correct'], 1)
        self.assertEqual(metrics['reward_mean'], .8)
        self.assertIsNone(advantages[2])
        self.assertAlmostEqual(advantages[0], 1.)
        self.assertEqual(advantages[3:], [0., 0., 0.])

    def test_stops_use_prior_steps_only_and_two_increases(self):
        stable = {'entropy': 2., 'grad_norm': 1., 'loss': 0., 'empty_or_invalid_rate': .03}
        rising = dict(stable, empty_or_invalid_rate=.09)
        now = dict(stable, empty_or_invalid_rate=.11)
        self.assertIsNone(stop_reason([rising], now))
        self.assertEqual(stop_reason([stable, rising], now), 'empty_or_format_rate')
        self.assertEqual(stop_reason([stable], dict(stable, grad_norm=11.)), 'gradient_norm_spike')
        self.assertEqual(stop_reason([stable], dict(stable, entropy=.99)), 'entropy_collapse')
        self.assertEqual(stop_reason([], dict(stable, loss=float('nan'))), 'nonfinite_loss_or_gradient')
