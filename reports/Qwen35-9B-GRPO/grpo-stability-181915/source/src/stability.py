"""Shared schedule, reward, importance sampling and stop rules for the ten-step run."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def write_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_bytes())


def make_schedule(ids, seed, heldout=0):
    if len(set(ids)) != len(ids) or len(ids) < 160 + heldout:
        raise ValueError(f'Expected at least {160 + heldout} unique authorized queries')
    ordered = sorted(ids, key=lambda key: (hashlib.sha256(f'{seed}\n{key}'.encode()).digest(), key))
    return {'seed': seed, 'training': [ordered[i:i + 16] for i in range(0, 160, 16)],
            'heldout': ordered[160:160 + heldout]}


def freeze(pool, destination, seed):
    raw = pool.read_bytes()
    source = json.loads(raw)
    rows = source['train']
    questions = {}
    for row in rows:
        qid = str(row['query_id'])
        question, gold = row['question'], row['gold']
        if qid in questions or not isinstance(question, str) or not question.strip() or not gold:
            raise ValueError('Training pool has duplicate IDs or incomplete questions and answers')
        questions[qid] = {'query_id': qid, 'question': question, 'gold': gold,
                          'question_sha256': hashlib.sha256(question.encode()).hexdigest()}
    if len({r['question_sha256'] for r in questions.values()}) != len(questions):
        raise ValueError('Training pool has duplicate question text')
    schedule = make_schedule(list(questions), seed)
    schedule['pool_sha256'] = hashlib.sha256(raw).hexdigest()
    schedule['questions'] = {qid: questions[qid] for group in schedule['training'] for qid in group}
    schedule['heldout_reason'] = 'Optional endpoint evaluations omitted to keep the ten-step stability run bounded.'
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, schedule)
    return {'query_count': 160, 'steps': 10, 'source_sha256': schedule['pool_sha256'], 'seed': seed}


def assess_batch(ids, rewards):
    groups = defaultdict(list)
    for index, (qid, reward) in enumerate(zip(ids, rewards, strict=True)):
        groups[qid]
        if reward is not None:
            if reward not in (0., 1.):
                raise ValueError('Nonbinary reward')
            groups[qid].append((index, reward))
    counts = dict(group_all_correct=0, group_all_wrong=0, group_mixed=0, group_unscored=0)
    advantages = [None] * len(rewards)
    for group in groups.values():
        values = [reward for _, reward in group]
        if not values:
            counts['group_unscored'] += 1
            continue
        mean = sum(values) / len(values)
        std = math.sqrt(sum((value - mean)**2 for value in values) / len(values))
        kind = 'all_correct' if mean == 1 else 'all_wrong' if mean == 0 else 'mixed'
        counts['group_' + kind] += 1
        for index, value in group:
            advantages[index] = (value - mean) / (std + 1e-8)
    scored = [reward for reward in rewards if reward is not None]
    return advantages, dict(counts, scored_trajectories=len(scored),
                            reward_mean=sum(scored) / len(scored) if scored else None)


def tis_weights(current, sampled):
    difference = current.detach() - sampled
    log_cap = math.log(2.)
    return difference.clamp(max=log_cap).exp(), difference > log_cap, difference.abs()


def stop_reason(previous, current):
    if not all(math.isfinite(current[key]) for key in ('loss', 'grad_norm', 'entropy')):
        return 'nonfinite_loss_or_gradient'
    if previous and current['grad_norm'] > 10 * median(row['grad_norm'] for row in previous):
        return 'gradient_norm_spike'
    if previous and current['entropy'] < previous[0]['entropy'] / 2:
        return 'entropy_collapse'
    invalid = current['empty_or_invalid_rate']
    if (len(previous) >= 2 and invalid > .10
            and previous[-2]['empty_or_invalid_rate'] < previous[-1]['empty_or_invalid_rate'] < invalid):
        return 'empty_or_format_rate'
    if current.get('repetition_triggered'):
        return 'repeated_output'
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pool', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--seed', type=int, default=20261110)
    args = parser.parse_args()
    print(json.dumps(freeze(args.pool, args.destination, args.seed)))
