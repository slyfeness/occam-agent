"""Collapse exact native extensions, retaining the action identity of each token."""


def merge_exact_segments(segments):
    """Return maximal spans. Gaps in action_offsets are observed context."""
    spans = []
    for segment in segments:
        prompt = segment['prompt_ids']
        actions = segment['completion_ids']
        probabilities = segment['vllm_logprobs']
        if not prompt or not actions or len(probabilities) != len(actions):
            raise ValueError('Expected a nonempty native call with one probability per action')
        if spans:
            span = spans[-1]
            prefix = span['prompt_ids'] + span['completion_ids']
            if prompt[:len(prefix)] == prefix:
                offset = len(prompt) - len(span['prompt_ids'])
                span['completion_ids'].extend(prompt[len(prefix):])
                span['completion_ids'].extend(actions)
                span['vllm_logprobs'].extend(probabilities)
                span['action_offsets'].extend(range(offset, offset + len(actions)))
                continue
        spans.append({'prompt_ids': list(prompt), 'completion_ids': list(actions),
                      'vllm_logprobs': list(probabilities), 'action_offsets': list(range(len(actions)))})
    return spans


def action_records(trajectories, merge=False):
    records = []
    for trajectory in trajectories:
        calls = trajectory['segments']
        action_count = sum(len(call['completion_ids']) for call in calls)
        spans = merge_exact_segments(calls) if merge else calls
        records.extend(dict(span, trajectory_id=trajectory['id'], advantage=trajectory['advantage'],
                            action_count=action_count) for span in spans)
    return records


def action_offsets(record):
    return record.get('action_offsets', range(len(record['completion_ids'])))
