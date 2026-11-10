"""Select and certify native actions for the read-only scoring diagnostic."""
from .native_spans import merge_exact_segments


def boundary_cases(calls):
    cases = []
    offset = 0
    for i, (before, after) in enumerate(zip(calls, calls[1:]), 1):
        expected = before['prompt_ids'] + before['completion_ids']
        actual = after['prompt_ids']
        common = next((k for k, (left, right) in enumerate(zip(expected, actual)) if left != right),
                      min(len(expected), len(actual)))
        if common < len(expected):
            previous = offset + common - len(before['prompt_ids']) if common >= len(before['prompt_ids']) else None
            cases.append({'before_call': i, 'after_call': i+1, 'common_prefix_tokens': common,
                          'previous_action_index': previous,
                          'next_action_index': offset + len(before['completion_ids'])})
        offset += len(before['completion_ids'])
    return cases


def verify_native_contexts(calls, spans):
    remaining = iter(calls)
    certified = 0
    for span in spans:
        start = len(span['prompt_ids'])
        sequence = span['prompt_ids'] + span['completion_ids']
        positions = span['action_offsets']
        offset = 0
        while offset < len(positions):
            call = next(remaining, None)
            if call is None:
                raise ValueError('Span introduced extra actions')
            prompt, actions = call['prompt_ids'], call['completion_ids']
            end = offset + len(actions)
            action_start = len(prompt) - start
            if (positions[offset:end] != list(range(action_start, action_start+len(actions)))
                    or sequence[:len(prompt)+len(actions)] != prompt+actions
                    or span['vllm_logprobs'][offset:end] != call['vllm_logprobs']):
                raise ValueError('A native causal history, action mask or probability changed')
            offset = end
            certified += len(actions)
    if next(remaining, None) is not None:
        raise ValueError('Span dropped original actions')
    return certified


def longest(item):
    return max(len(r['prompt_ids']) + len(r['completion_ids']) for r in item['segments'])


def select_trajectories(batch, size=8):
    ordered = sorted(enumerate(batch, 1), key=lambda pair: (longest(pair[1]), pair[0]))
    changed = [pair for pair in ordered if boundary_cases(pair[1]['segments'])]
    exact = [pair for pair in ordered if not boundary_cases(pair[1]['segments'])]
    changed_count = min(len(changed), max(size//2, size-len(exact)))

    def quantiles(items, count):
        if not count:
            return []
        if count == 1:
            return [items[len(items)//2]]
        return [items[i*(len(items)-1)//(count-1)] for i in range(count)]

    chosen = sorted(quantiles(changed, changed_count) + quantiles(exact, size-changed_count))
    if len(chosen) != size:
        raise ValueError('Unable to form the declared representative subset')
    return chosen


def certify(trajectories):
    return sum(verify_native_contexts(t['segments'], merge_exact_segments(t['segments']))
               for t in trajectories)
