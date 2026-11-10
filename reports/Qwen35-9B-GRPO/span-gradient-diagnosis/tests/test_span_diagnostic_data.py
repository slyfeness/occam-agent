import unittest

from probe.span_diagnostic_data import boundary_cases, verify_native_contexts


def call(prompt, action):
    return {'prompt_ids': prompt, 'completion_ids': action, 'vllm_logprobs': [-1.] * len(action)}


class DiagnosticDataTests(unittest.TestCase):
    def test_maps_divergent_original_token_and_next_action_separately(self):
        calls = [call([1, 2], [3, 4]), call([1, 2, 13, 4, 5], [6, 7])]
        self.assertEqual(boundary_cases(calls), [{'before_call': 1, 'after_call': 2,
            'common_prefix_tokens': 2, 'previous_action_index': 0, 'next_action_index': 2}])

    def test_tool_context_is_checked_even_when_action_ids_match(self):
        calls = [call([1], [2]), call([1, 2, 3], [4])]
        span = {'prompt_ids': [1], 'completion_ids': [2, 3, 4],
                'action_offsets': [0, 2], 'vllm_logprobs': [-1., -1.]}
        self.assertEqual(verify_native_contexts(calls, [span]), 2)
        with self.assertRaises(ValueError):
            verify_native_contexts(calls, [{**span, 'completion_ids': [2, 30, 4]}])


if __name__ == '__main__':
    unittest.main()
