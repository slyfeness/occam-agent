"""Compare BF16 tensors in two sealed archives without publishing tensor values."""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


def header(path):
    with path.open('rb') as handle:
        size = struct.unpack('<Q', handle.read(8))[0]
        raw = handle.read(size)
    data = json.loads(raw)
    data.pop('__metadata__', None)
    return size + 8, raw, data


def compare(first, second):
    first_base, first_header, original = header(first)
    second_base, second_header, replayed = header(second)
    if original.keys() != replayed.keys():
        raise ValueError('Tensor identities differ')
    values = changed = tensors_changed = 0
    sum_squared = max_absolute = 0.
    nonfinite_changed = 0
    for name, entry in original.items():
        other = replayed[name]
        if entry['shape'] != other['shape'] or entry['dtype'] != other['dtype'] or entry['dtype'] != 'BF16':
            raise ValueError('Expected identical BF16 tensor layouts')
        start, end = entry['data_offsets']
        alt, alt_end = other['data_offsets']
        size = (end-start)//2
        if end-start != alt_end-alt or size != math.prod(entry['shape']):
            raise ValueError('Tensor extent differs')
        left = np.memmap(first, mode='r', dtype='<u2', offset=first_base+start, shape=(size,))
        right = np.memmap(second, mode='r', dtype='<u2', offset=second_base+alt, shape=(size,))
        changes = 0
        for offset in range(0, size, 4*2**20):
            a, b = left[offset:offset+4*2**20], right[offset:offset+4*2**20]
            mask = a != b
            count = int(np.count_nonzero(mask))
            changes += count
            if count:
                x = (a[mask].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
                y = (b[mask].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
                finite = np.isfinite(x) & np.isfinite(y)
                nonfinite_changed += int(np.count_nonzero(~finite))
                delta = x[finite] - y[finite]
                if delta.size:
                    sum_squared += float(np.sum(delta * delta))
                    max_absolute = max(max_absolute, float(np.max(np.abs(delta))))
        values += size
        changed += changes
        tensors_changed += int(changes != 0)
    return {'same_raw_header': first_header == second_header, 'total_tensors': len(original),
            'changed_tensors': tensors_changed, 'total_values': values, 'changed_values': changed,
            'changed_fraction': changed/values, 'max_absolute_difference': max_absolute,
            'rms_difference': math.sqrt(sum_squared/values), 'nonfinite_changed_values': nonfinite_changed}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.first, args.second)))
