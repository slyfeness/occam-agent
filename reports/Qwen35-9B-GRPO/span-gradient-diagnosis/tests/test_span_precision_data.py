import unittest

from probe.span_precision_data import noise_halves, short_reference_items


def item(index, query, reward, length=3):
    def call(prompt, answer):
        return {'prompt_ids':prompt,'completion_ids':answer,'vllm_logprobs':[-1.] * len(answer)}
    prefix = list(range(1,length+1))
    return {'id':str(index),'query_id':query,'reward':reward,'advantage':reward-.5,
            'segments':[call(prefix,[100]),call(prefix+[100,101],[102])]}


class PrecisionSelectionTests(unittest.TestCase):
    def test_reference_uses_short_extendable_trajectories(self):
        pool = [item(i, 'q', i%2, length) for i,length in enumerate([8,4,6,3,5])]
        chosen = short_reference_items(pool)
        self.assertEqual([x['id'] for x in chosen], ['3','1','4'])

    def test_noise_has_same_queries_disjoint_actions_and_centered_advantages(self):
        pool = [item(q*8+i,str(q),i%2) for q in range(5) for i in range(8)]
        a,b = noise_halves(pool)
        self.assertEqual(len(a),12)
        self.assertEqual(len(b),12)
        self.assertFalse({x['id'] for x in a}&{x['id'] for x in b})
        self.assertEqual({x['query_id'] for x in a},{x['query_id'] for x in b})
        for part in (a,b):
            for query in {x['query_id'] for x in part}:
                self.assertAlmostEqual(sum(x['advantage'] for x in part if x['query_id']==query),0.)


if __name__ == '__main__':
    unittest.main()
