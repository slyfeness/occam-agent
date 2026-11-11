import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from src.snapshot_archive import verify_archive, write_archive


class SnapshotArchiveTests(unittest.TestCase):
    def test_streamed_archive_verifies_against_source(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'policy.safetensors'
            tensors = {'a': memoryview(b'1234'), 'b': memoryview(b'5678')}
            write_archive(path, tensors, {'a': [2], 'b': [2]}, {'fixture': 'true'})
            self.assertEqual(verify_archive(path, tensors), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_verifies_each_tensor_and_file_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'policy.safetensors'
            header = json.dumps({
                'a': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]},
                'b': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [4, 8]},
                '__metadata__': {},
            }).encode()
            expected = {'a': memoryview(b'1234'), 'b': memoryview(b'5678')}
            whole = struct.pack('<Q', len(header)) + header + b'12345678'
            path.write_bytes(whole)
            self.assertEqual(verify_archive(path, expected), hashlib.sha256(whole).hexdigest())
            path.write_bytes(whole[:-4] + b'\x00' * 4)
            with self.assertRaisesRegex(ValueError, 'tensor bytes'):
                verify_archive(path, expected)
            path.write_bytes(whole[:-4])
            with self.assertRaisesRegex(ValueError, 'tensor extent'):
                verify_archive(path, expected)
