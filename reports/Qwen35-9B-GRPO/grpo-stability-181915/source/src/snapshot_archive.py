"""Compare a safetensors archive against its exact in-memory tensor bytes."""

import hashlib
import json
import os
import struct


def write_archive(path, tensors, shapes, metadata):
    header = {'__metadata__': metadata}
    offset = 0
    for name, tensor in tensors.items():
        end = offset + len(tensor)
        header[name] = {'dtype': 'BF16', 'shape': shapes[name], 'data_offsets': [offset, end]}
        offset = end
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    with path.open('xb') as archive:
        archive.write(struct.pack('<Q', len(raw)))
        archive.write(raw)
        for tensor in tensors.values():
            for start in range(0, len(tensor), 16 * 2**20):
                archive.write(tensor[start:start + 16 * 2**20])
        archive.flush()
        os.fsync(archive.fileno())


def verify_archive(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as archive:
        prefix = archive.read(8)
        if len(prefix) != 8:
            raise ValueError('Invalid tensor archive header length')
        size = struct.unpack('<Q', prefix)[0]
        if not 0 < size <= 16 * 2**20:
            raise ValueError('Invalid tensor archive header size')
        raw_header = archive.read(size)
        if len(raw_header) != size:
            raise ValueError('Truncated tensor archive header')
        header = json.loads(raw_header)
        header.pop('__metadata__', None)
        if set(header) != set(expected):
            raise ValueError('Tensor archive names differ from the export')
        digest.update(prefix)
        digest.update(raw_header)
        available = os.fstat(archive.fileno()).st_size - size - 8
        cursor = 0
        for name, entry in sorted(header.items(), key=lambda item: item[1]['data_offsets']):
            start, end = entry['data_offsets']
            tensor = expected[name]
            if (entry['dtype'] != 'BF16' or start != cursor or
                    end-start != len(tensor) or end > available):
                raise ValueError(f'Invalid tensor extent for {name}')
            tensor_digest = hashlib.sha256()
            remaining = end - start
            while remaining:
                part = archive.read(min(8 * 2**20, remaining))
                if not part:
                    raise ValueError(f'Truncated tensor extent for {name}')
                tensor_digest.update(part)
                digest.update(part)
                remaining -= len(part)
            if tensor_digest.digest() != hashlib.sha256(tensor).digest():
                raise ValueError(f'On-disk tensor bytes differ for {name}')
            cursor = end
        if cursor != available:
            raise ValueError('Unexpected bytes after the tensor extents')
    return digest.hexdigest()
