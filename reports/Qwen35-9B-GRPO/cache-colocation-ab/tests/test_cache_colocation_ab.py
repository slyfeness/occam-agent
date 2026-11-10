import unittest

from cache_colocation_ab import headroom, read_counters, summarize_trial, verify_response


class CacheColocationTests(unittest.TestCase):
    def test_prefix_mismatch_is_rejected(self):
        response = {'prompt_token_ids': [2, 4], 'usage': {'completion_tokens': 3}}
        self.assertEqual(verify_response(response, [2, 4]), 3)
        with self.assertRaisesRegex(ValueError, 'prefix_mismatch'):
            verify_response(response, [2, 5])

    def test_counter_deltas_and_denominators(self):
        before = read_counters('vllm:prefix_cache_hits_total{engine="0"} 10\n')
        after = read_counters('\n'.join([
            'vllm:prefix_cache_hits_total{engine="0"} 60',
            'vllm:prefix_cache_queries_total 200',
            'vllm:num_preemptions_total 0',
            'vllm:request_prefill_time_seconds_sum 1.5',
            'vllm:request_prefill_time_seconds_count 3',
        ]))
        result = summarize_trial(before, after, 3, 120, 4)
        self.assertEqual(result['prefix_hit_rate'], 0.25)
        self.assertEqual(result['prefill_mean_seconds'], 0.5)
        self.assertEqual(result['output_tokens_per_second'], 30)
        self.assertEqual(result['preemptions'], 0)

    def test_missing_metrics_are_not_reported_as_zero(self):
        result = summarize_trial({}, {}, 3, 120, 4)
        self.assertIsNone(result['preemptions'])
        self.assertIsNone(result['prefix_hit_rate'])

    def test_memory_budget_uses_total_not_free(self):
        self.assertEqual(headroom(800, 635, .75), 35)
