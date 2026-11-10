import unittest

from probe.single_step_old import DETACHED_MODE, pre_update_probability_mode


class SingleStepOldTests(unittest.TestCase):
    def test_detach_mode_requires_exactly_one_reward_step(self):
        self.assertEqual(pre_update_probability_mode(DETACHED_MODE,('reward',)),DETACHED_MODE)
        for phases in (('zero_advantage','reward'),('reward','reward'),()):
            with self.assertRaises(ValueError):
                pre_update_probability_mode(DETACHED_MODE,phases)

    def test_unrecognized_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            pre_update_probability_mode('guess',('reward',))


if __name__=='__main__':
    unittest.main()
