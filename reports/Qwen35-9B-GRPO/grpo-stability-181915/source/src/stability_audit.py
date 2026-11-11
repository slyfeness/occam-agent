"""Audit completed GRPO steps and summarize private tool timings without exporting raw traces."""

import argparse
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from statistics import mean, quantiles

from .stability import read_json, write_json


def distribution(values):
    return {'count': len(values), 'mean_seconds': mean(values) if values else None,
            'p95_seconds': (quantiles(values, n=100, method='inclusive')[94] if len(values) > 1
                            else values[0] if values else None)}


def audit(run):
    status = read_json(run / 'STATUS.json')
    plan = read_json(run / 'RUN_PLAN.json')
    all_ids = [qid for group in plan['training'] for qid in group]
    if len(all_ids) != 160 or len(set(all_ids)) != 160 or [len(g) for g in plan['training']] != [16]*10:
        raise ValueError('Frozen query schedule changed')
    results = {}
    persisted = sorted(path.name for path in (run / 'checkpoints').glob('step-*') if path.is_dir())
    if set(persisted) - {'step-05', 'step-10'}:
        raise ValueError('Unexpected optimizer checkpoint')
    for path in sorted(run.glob('step-*/STEP_RESULT.json')):
        row = read_json(path)
        step = row['step']
        group = plan['training'][step-1]
        if row['query_ids'] != group:
            raise ValueError(f'Query identity mismatch at step {step}')
        attempt = sorted(path.parent.glob('rollout-*/ROLLOUT_RESULT.json'))[-1].parent
        trajectory = read_json(attempt / 'TRAJECTORY_METRICS.json')
        api = read_json(attempt / 'API_JUDGE.json')
        expected = {qid: 8 for qid in group}
        if len(api) != 128 or Counter(r['query_id'] for r in trajectory) != expected:
            raise ValueError(f'Incomplete 16-by-8 batch at step {step}')
        if len({r['id'] for r in trajectory}) != 128:
            raise ValueError(f'Duplicate trajectories at step {step}')
        if len(list((attempt / 'api-verdicts').glob('*.json'))) != 128:
            raise ValueError(f'Incomplete API verdict receipts at step {step}')
        if any(r['retries'] > 3 or not r['attempts'] for r in api):
            raise ValueError(f'API retry rule was not met at step {step}')
        marks = [(record[key], delta) for record in api
                 for key, delta in (('started_at', 1), ('ended_at', -1))]
        running = peak_concurrency = 0
        for _, delta in sorted(marks):
            running += delta
            peak_concurrency = max(running, peak_concurrency)
        if running != 0 or peak_concurrency > 16:
            raise ValueError(f'API concurrency exceeded the configured limit at step {step}')
        if Counter(r['terminal'] for r in trajectory) != row['termination']:
            raise ValueError(f'Termination counts differ at step {step}')
        for r, verdict in zip(trajectory, api, strict=True):
            if not all(isinstance(r[key], int) and r[key] >= 0 for key in (
                    'generated_tokens', 'context_tokens_sum', 'context_tokens_max', 'tool_calls', 'rounds')):
                raise ValueError(f'Missing trajectory metrics at step {step}')
            if r['rounds'] > 20 or r['context_tokens_max'] > 65536:
                raise ValueError(f'Trajectory exceeded frozen limits at step {step}')
            if r['terminal'] != 'MODEL_FINAL_STOP' and r['reward'] != 0:
                raise ValueError(f'Terminal reward override changed at step {step}')
            if r['terminal'] == 'MODEL_FINAL_STOP':
                expected_reward = None if verdict['correct'] is None else float(verdict['correct'])
                if r['reward'] != expected_reward or ((r['advantage'] is None) != (expected_reward is None)):
                    raise ValueError(f'API reward exclusion changed at step {step}')
        calls = read_json(attempt / 'RETRIEVAL_TIMING.json')
        events = [r for r in calls if r['event_id'] is not None]
        if len(events) != sum(r['tool_calls'] for r in trajectory) or len(events) != row['retrieval_calls']:
            raise ValueError(f'Retrieval event counts differ at step {step}')
        train = row['train']
        completed = row.get('policy_synced', False)
        if completed:
            if not train['optimizer_step'] or train['adam_state_step_max'] != step:
                raise ValueError(f'Optimizer state is not continuous at step {step}')
            if len(row['train_peak_nvml_mib']) != 4 or len(row['actor_stats']) != 4:
                raise ValueError(f'GPU measurement coverage is incomplete at step {step}')
            if not all(math.isfinite(train[k]) for k in ('loss', 'entropy', 'grad_norm', 'tis_clip_fraction')):
                raise ValueError(f'Nonfinite training result at step {step}')
            archive = read_json(run / 'policy-snapshots' / f'step-{step}.json')
            weights = run / 'policy-snapshots' / archive['snapshot']
            size = weights.stat()
            if (not archive['file_sha256'] or size.st_size != archive['verified_byte_count']
                    or size.st_blocks * 512 < size.st_size):
                raise ValueError(f'Verified weight archive missing at step {step}')
        results[str(step)] = {
            'optimizer_and_sync_verified': completed,
            'trajectories': len(trajectory),
            'output_tokens': sum(r['generated_tokens'] for r in trajectory),
            'context_tokens_sum': sum(r['context_tokens_sum'] for r in trajectory),
            'longest_context_tokens': max(r['context_tokens_max'] for r in trajectory),
            'api_verdicts': len(api),
            'api_peak_concurrency': peak_concurrency,
            'retrieval_event_wall': distribution([r['event_wall_seconds'] for r in events]),
            'retrieval_server_queue': distribution([r['server_queue_seconds'] for r in events
                                                    if r['server_queue_seconds'] is not None]),
            'retrieval_client_to_search': distribution([r['pre_search_wait_seconds'] for r in events
                                                       if r.get('pre_search_wait_seconds') is not None]),
        }
    checkpoints = {}
    for name in persisted:
        folder = run / 'checkpoints' / name
        step = int(name.split('-')[1])
        if read_json(folder / 'COMPLETE.json') != {'step': step, 'world_size': 4}:
            raise ValueError(f'Checkpoint completion receipt differs at step {step}')
        paths = sorted(folder.glob('checkpoint-rank*.pt'))
        if len(paths) != 4:
            raise ValueError(f'Checkpoint does not contain four rank shards at step {step}')
        for path in paths:
            with zipfile.ZipFile(path) as archive:
                if not any(name.endswith('/data.pkl') for name in archive.namelist()):
                    raise ValueError(f'Checkpoint structure is invalid at step {step}')
            stat = path.stat()
            if stat.st_blocks * 512 < stat.st_size:
                raise ValueError(f'Sparse optimizer checkpoint at step {step}')
        checkpoints[str(step)] = {'rank_shards': 4, 'total_bytes': sum(p.stat().st_size for p in paths)}
    if status['state'] == 'complete':
        if status['optimizer_steps'] != 10 or len(results) != 10 or set(checkpoints) != {'5', '10'}:
            raise ValueError('Ten-step completion lacks required artifacts')
        if not all(r['optimizer_and_sync_verified'] for r in results.values()):
            raise ValueError('A completed step did not verify policy synchronization')
    return {'job': status['job'], 'state': status['state'], 'unique_scheduled_queries': len(all_ids),
            'steps': results, 'checkpoints': checkpoints}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    write_json(args.output, result)
    print(json.dumps({'state': result['state'], 'audited_steps': len(result['steps']),
                      'checkpoint_steps': sorted(result['checkpoints'])}))
