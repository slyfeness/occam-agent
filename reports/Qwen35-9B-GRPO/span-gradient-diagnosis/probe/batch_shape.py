"""Batch shapes authorized for the two one-update GRPO experiments."""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchShape:
    query_count: int
    samples_per_query: int

    def __post_init__(self):
        if (self.query_count, self.samples_per_query) not in {(4, 4), (16, 8)}:
            raise ValueError('Expected a 4x4 or 16x8 GRPO batch')

    @classmethod
    def from_config(cls, config):
        if 'query_count' not in config and 'samples_per_query' not in config:
            return cls(4, 4)
        return cls(config.get('query_count'), config.get('samples_per_query'))

    @property
    def trajectories(self):
        return self.query_count * self.samples_per_query

    def verify_groups(self, query_ids):
        counts = Counter(query_ids)
        if len(counts) != self.query_count or any(count != self.samples_per_query for count in counts.values()):
            raise ValueError('GRPO query groups do not match the frozen batch shape')
