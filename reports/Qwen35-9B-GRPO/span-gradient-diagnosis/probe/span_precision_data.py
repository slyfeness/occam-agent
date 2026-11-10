"""Frozen sample selections for reference precision and rollout split diagnostics."""
import collections
import math
import random

from .native_spans import merge_exact_segments
from .span_diagnostic_data import longest


def short_reference_items(batch, limit=8192):
    eligible = sorted((t for t in batch if t['advantage'] != 0 and longest(t) <= limit),
                      key=lambda t:(longest(t),t['id']))
    if len(eligible) < 2:
        raise ValueError('Fewer than two short trajectories with a useful gradient')
    chosen = eligible[:3]

    def extends(t):
        return len(merge_exact_segments(t['segments'])) < len(t['segments'])

    if not any(map(extends,chosen)):
        candidate = next((t for t in eligible if extends(t)),None)
        if candidate is None:
            raise ValueError('The FP32 sample needs a contributing prefix extension')
        chosen[-1] = candidate
    return chosen


def noise_halves(batch, seed=20261110, queries=3):
    groups = collections.defaultdict(list)
    for row in batch:
        groups[row['query_id']].append(row)
    if len(groups) < queries or any(len(rows)!=8 for rows in groups.values()):
        raise ValueError('Expected complete groups of eight saved trajectories')
    randomizer = random.Random(seed)
    chosen = randomizer.sample(sorted(groups),queries)
    halves = [[],[]]
    for query in chosen:
        records = sorted(groups[query],key=lambda t:t['id'])
        randomizer.shuffle(records)
        for j,rows in enumerate((records[:4],records[4:])):
            rewards = [float(t['reward']) for t in rows]
            if not all(map(math.isfinite,rewards)):
                raise ValueError('Invalid saved reward')
            center = sum(rewards)/len(rewards)
            std = math.sqrt(sum((r-center)**2 for r in rewards)/len(rewards))
            halves[j].extend(dict(t,advantage=(r-center)/(std+1e-8)) for t,r in zip(rows,rewards,strict=True))
    return halves
