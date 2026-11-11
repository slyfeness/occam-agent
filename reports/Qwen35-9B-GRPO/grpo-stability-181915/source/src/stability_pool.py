"""Add 60 unused BrowseComp-Plus rows to the authorized historical 100."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

from .stability import freeze, read_json, write_json


def main():
    import pyarrow.parquet as parquet

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-manifest', type=Path, required=True)
    parser.add_argument('--legacy-results', type=Path, required=True)
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--decrypt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    seed = 20261110
    original = read_json(args.legacy_manifest)['selected']
    old_gold = {}
    for line in args.legacy_results.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        old_gold[str(row['query_id'])] = row['gold']
    old_ids = {str(row['query_id']) for row in original}
    if len(original) != 100 or len(old_ids) != 100 or not old_ids <= old_gold.keys():
        raise ValueError('Legacy 100-query pool identity did not match')
    decrypt = runpy.run_path(str(args.decrypt))
    password = decrypt['DEFAULT_CANARY']
    plain = decrypt['decrypt_string']
    assets = sorted(args.parquet.glob('test-*.parquet'))
    if not assets:
        raise ValueError('No original local BrowseComp-Plus shards')
    official = {}
    for asset in assets:
        for row in parquet.read_table(asset, columns=['query_id', 'query', 'answer']).to_pylist():
            qid = str(row['query_id'])
            if not qid.isascii() or not qid.isdecimal() or qid in official:
                raise ValueError('Invalid or duplicate original query ID')
            official[qid] = {'query_id': qid, 'question': plain(row['query'], password),
                             'gold': plain(row['answer'], password)}
    legacy = []
    for row in original:
        qid = str(row['query_id'])
        current = official[qid]
        if (current['question'] != row['question'] or current['gold'] != old_gold[qid]
                or hashlib.sha256(current['question'].encode()).hexdigest() != row['question_sha256']):
            raise ValueError('Original asset and legacy query identity disagree')
        legacy.append(current)
    normalize = lambda question: ' '.join(question.casefold().split())
    used = {normalize(row['question']) for row in legacy}
    additional = []
    for qid in sorted(official.keys() - old_ids,
                      key=lambda key: (hashlib.sha256(f'{seed}:additional:{key}'.encode()).digest(), key)):
        row = official[qid]
        text = normalize(row['question'])
        if text and text not in used and row['gold'].strip():
            used.add(text)
            additional.append(row)
            if len(additional) == 60:
                break
    if len(additional) != 60:
        raise ValueError('Original pool does not contain 60 distinct unused questions')
    args.output.mkdir(mode=0o700, exist_ok=False)
    pool = args.output / 'PRIVATE_POOL.json'
    plan = args.output / 'PLAN.json'
    write_json(pool, {'train': legacy + additional, 'historical_count': 100, 'added_count': 60,
                      'original_split': 'test', 'original_rows': len(official),
                      'original_shards_sha256': {asset.name: hashlib.file_digest(asset.open('rb'), 'sha256').hexdigest()
                                                 for asset in assets}})
    receipt = freeze(pool, plan, seed)
    write_json(args.output / 'POOL_RECEIPT.json', dict(receipt, historical_count=100, added_count=60,
                                                      original_rows=len(official)))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
