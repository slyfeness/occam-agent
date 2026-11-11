"""Replay the 181505 actor/search protocol with online judging and tool timing."""

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from .batch_shape import BatchShape
from .rollout import HISTORY, runtime
from .stability import assess_batch, read_json, write_json
from .stability_judge import summarize


def instrument_search_timing(source):
    anchor = ' import hmac\n'
    if source.count(anchor) != 1:
        raise RuntimeError('Frozen retriever timing boundary changed')
    injection = ''' from contextvars import ContextVar
 from fastmcp.server.middleware import Middleware
 timing = ContextVar('tool_timing', default=None)
 original_search = searcher.search
 def timed_search(*args, **kwargs):
  entry=timing.get()
  if entry is None:return original_search(*args, **kwargs)
  entry['search_started']=time.monotonic()
  try:return original_search(*args, **kwargs)
  finally:entry['search_ended']=time.monotonic()
 searcher.search=timed_search
 class ToolTiming(Middleware):
  async def on_call_tool(self, context, call_next):
   arguments=dict(context.message.arguments or {})
   trace=arguments.pop('_occam_timing',None)
   if trace is None:return await call_next(context)
   entry={'trace':trace,'handler_started':time.monotonic()}
   token=timing.set(entry)
   try:
    message=context.message.model_copy(update={'arguments':arguments})
    return await call_next(context.copy(message=message))
   finally:
    timing.reset(token)
    entry['handler_ended']=time.monotonic()
    with (folder/'TOOL_TIMING.jsonl').open('a') as handle:
     handle.write(json.dumps(entry)+'\\n')
 mcp.add_middleware(ToolTiming())
'''
    return source.replace(anchor, injection + anchor, 1)


def worker():
    from mcp.client.session import ClientSession

    output = Path(os.environ['SMOKE_RUN'])
    qid, url, server, port = sys.argv[2:]
    trajectory = output / 'trajectories' / qid
    original_call = ClientSession.call_tool
    call_count = 0

    async def timed_call(self, name, arguments=None, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        trace = f'{qid}:{call_count}'
        arguments = dict(arguments or {}, _occam_timing=trace)
        started = time.monotonic()
        try:
            return await original_call(self, name, arguments, *args, **kwargs)
        finally:
            with (trajectory / 'TOOL_TIMING.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'trace': trace, 'sent': started, 'received': time.monotonic()}) + '\n')

    ClientSession.call_tool = timed_call
    sys.argv[1:] = ['worker', qid, url, server, port]
    from .rollout import main as legacy_worker
    legacy_worker(keep_services=True)


def read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def union_seconds(intervals):
    total = 0.
    end = 0.
    for left, right in sorted(intervals):
        total += max(0., right - max(left, end))
        end = max(end, right)
    return total


def retrieval_timing(client, server):
    search_start = server.get('search_started')
    search_end = server.get('search_ended')
    if (search_start is None) != (search_end is None):
        raise RuntimeError('Incomplete timing inside the retriever')
    return dict(client, **server, client_wall_seconds=client['received']-client['sent'],
                entered_search=search_start is not None,
                handler_seconds=server['handler_ended']-server['handler_started'],
                handler_dispatch_seconds=server['handler_started']-client['sent'],
                pre_search_wait_seconds=None if search_start is None else search_start-client['sent'],
                server_queue_seconds=None if search_start is None else search_start-server['handler_started'],
                search_seconds=None if search_start is None else search_end-search_start)


def join_tool_timings(events, client, server):
    dispatched = sum(bool(event.get('raw_mcp_response_path')) for event in events)
    aligned = dispatched == len(client)
    wire = iter(client)
    rows = []
    for event in events:
        sent = bool(event.get('raw_mcp_response_path'))
        start = datetime.fromisoformat(event['dispatch_utc']).timestamp()
        end = datetime.fromisoformat(event['return_utc']).timestamp()
        row = {'event_id': event['event_id'], 'event_start_utc_seconds': start, 'event_end_utc_seconds': end,
               'event_wall_seconds': end-start, 'dispatched_to_server': sent,
               'entered_search': None if sent else False, 'server_queue_seconds': None,
               'wire_association_missing': sent and not aligned}
        if sent and aligned:
            item = next(wire)
            remote = server.pop(item['trace'], None)
            if remote is not None:
                row.update(retrieval_timing(item, remote))
            else:
                row.update(item, wire_association_missing=True)
        rows.append(row)
    if not aligned:
        for item in wire:
            remote = server.pop(item['trace'], None)
            row = retrieval_timing(item, remote) if remote is not None else dict(item)
            rows.append(dict(row, event_id=None, wire_association_missing=True))
    return rows


def trajectory_row(task, trajectory):
    rounds = trajectory['rounds']
    observed = [r for r in rounds if r.get('response_received')]
    outcome = trajectory['metadata']['terminal_status']
    segments = []
    repeated = False
    contexts = []
    for row in observed:
        tokens = row['token_accounting']
        prompt, completion = tokens['prompt_token_ids'], tokens['completion_token_ids']
        if not tokens['exact_prompt_token_ids_equal'] or completion is None:
            raise RuntimeError('Exact native token accounting missing')
        contexts.append(len(prompt))
        if not completion:
            continue
        probabilities = row['raw_response']['choices'][0]['logprobs']['content']
        if len(probabilities) != len(completion):
            raise RuntimeError('Native vLLM probability coverage is incomplete')
        tail = completion[-128:]
        repeated |= len(tail) == 128 and any(tail[period:] == tail[:-period] for period in range(1, 17))
        segments.append({'prompt_ids': prompt, 'completion_ids': completion,
                         'vllm_logprobs': [item['logprob'] for item in probabilities]})
    final = observed[-1].get('content') or '' if observed else ''
    return {'id': task['run_id'], 'query_id': task['query_id'], 'terminal': outcome,
            'final': final, 'segments': segments, 'generated_tokens': sum(len(s['completion_ids']) for s in segments),
            'context_tokens_sum': sum(contexts), 'context_tokens_max': max(contexts, default=0),
            'tool_calls': len(trajectory['tool_events']), 'tool_events': trajectory['tool_events'],
            'rounds': len(rounds),
            'repeated_output': repeated}


def run_rollout(output, step, rows, seed, servers, port, search_journal, judge):
    output.mkdir(mode=0o700)
    shape = BatchShape(16, 8)
    tasks = [
        {'query_id': row['query_id'], 'repetition_id': rep+1, 'seed': seed + (step-1)*128 + i*8 + rep,
         'run_id': f"{row['query_id']}__rep{rep+1}", 'question_sha256': row['question_sha256']}
        for i, row in enumerate(rows) for rep in range(8)
    ]
    private = {row['query_id']: row for row in rows}
    write_json(output / 'sample_manifest.json', {'selected': [
        {k: row[k] for k in ('query_id', 'question', 'question_sha256')} for row in rows]})
    write_json(output / 'task_manifest.json', {'tasks': tasks})
    freeze = read_json(HISTORY / 'EXECUTION_FREEZE.json')
    write_json(output / 'EXECUTION_FREEZE.json', {'actor': freeze['actor'], 'tasks': tasks})
    write_json(output / 'config_manifest.json', {'query_count': 16, 'samples_per_query': 8, 'context': 65536,
               'max_tokens_per_call': 10000, 'max_actor_calls': 20, 'temperature': 1., 'top_p': 1., 'top_k': 0})
    offset = search_journal.stat().st_size if search_journal.exists() else 0
    env = dict(os.environ, SMOKE_RUN=str(output), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    env.pop('OCCAM_ACTOR_ADMIN', None)
    env.pop('JUDGE_API_KEY', None)
    judges = {}
    verdict_directory = output / 'api-verdicts'
    verdict_directory.mkdir()

    def execute(index, task):
        actor = servers[index % len(servers)]
        run_id = task['run_id']
        with (output / (run_id + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-u', '-m', 'src.stability_rollout', 'worker',
                                     run_id, actor.url, str(actor.folder), str(port)],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('A native rollout worker failed')
        trajectory = read_json(output / 'trajectories' / run_id / 'trajectory.json')
        row = trajectory_row(task, trajectory)
        question = private[task['query_id']]
        future = judge.submit(question['question'], row['final'], question['gold'],
                              verdict_directory / f'{index:03d}.json')
        judges[index] = future
        return row

    started = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as pool:
            futures = [pool.submit(execute, i, task) for i, task in enumerate(tasks)]
            records = [future.result() for future in futures]
    except Exception:
        for future in judges.values():
            future.cancel()
        raise
    generated = time.monotonic()
    answers = [judges[i].result() for i in range(len(tasks))]
    end = time.monotonic()
    timing = {}
    if search_journal.exists():
        with search_journal.open('rb') as handle:
            handle.seek(offset)
            timing = {row['trace']: row for row in map(json.loads, handle)}
    calls = []
    for record in records:
        client = read_lines(output / 'trajectories' / record['id'] / 'TOOL_TIMING.jsonl')
        calls.extend(dict(row, trajectory_id=record['id'])
                     for row in join_tool_timings(record['tool_events'], client, timing))
    write_json(output / 'RETRIEVAL_TIMING.json', calls)
    if timing:
        write_json(output / 'UNMATCHED_RETRIEVAL_TIMING.json', list(timing.values()))
    rewards = [float(answer['correct']) if answer['correct'] is not None else None
               for answer in answers]
    for i, record in enumerate(records):
        if record['terminal'] != 'MODEL_FINAL_STOP':
            rewards[i] = 0.
    advantages, metrics = assess_batch([t['query_id'] for t in tasks], rewards)
    shape.verify_groups(t['query_id'] for t in tasks)
    batch = [dict(row, reward=reward, advantage=advantage)
             for row, reward, advantage in zip(records, rewards, advantages, strict=True)]
    for row in batch:
        del row['final']
        del row['tool_events']
    write_json(output / 'TRAIN_BATCH.json', {'trajectories': batch, 'shape': [16, 8]})
    write_json(output / 'API_JUDGE.json', answers)
    if step == 1:
        write_json(output / 'LOCAL_JUDGE_INPUT.json', [
            {'question': private[record['query_id']]['question'], 'gold': private[record['query_id']]['gold'],
             'final': record['final'], 'termination': record['terminal'], 'api_correct': answer['correct']}
            for record, answer in zip(records, answers, strict=True)])
    write_json(output / 'TRAJECTORY_METRICS.json', [
        {key: value for key, value in row.items() if key not in ('segments', 'final')} for row in batch])
    counts = Counter(row['terminal'] for row in records)
    events = [call for call in calls if call['event_id'] is not None]
    retrieval_wall = union_seconds((call['event_start_utc_seconds'], call['event_end_utc_seconds'])
                                   for call in events)
    repeat_rate = sum(row['repeated_output'] for row in records) / 128
    invalid = counts['EMPTY_MODEL_STOP'] + counts['FORMAT_ERROR']
    metrics.update(termination=dict(counts), normal_count=counts['MODEL_FINAL_STOP'],
                   truncation_rate=(128-counts['MODEL_FINAL_STOP'])/128, empty_or_invalid_rate=invalid/128,
                   repetition_rate=repeat_rate, repetition_triggered=repeat_rate > .10,
                   mean_tool_calls=sum(row['tool_calls'] for row in records)/128,
                   mean_rounds=sum(row['rounds'] for row in records)/128,
                   generation_seconds=generated-started, generation_start=started, generation_end=generated,
                   judge_tail_seconds=end-generated, api=summarize(answers),
                   reward_excluded_count=sum(reward is None for reward in rewards),
                   retrieval_calls=len(events), retrieval_wall_seconds=retrieval_wall,
                   retrieval_local_only=sum(not call['dispatched_to_server'] for call in events),
                   retrieval_rejected_before_search=sum(call['dispatched_to_server'] and
                                                        call['entered_search'] is False for call in events),
                   retrieval_timing_unmatched=sum(call['wire_association_missing'] for call in calls) + len(timing),
                   retrieval_sum_seconds=sum(call['event_wall_seconds'] for call in events),
                   retrieval_fraction_of_generation=retrieval_wall/(generated-started))
    write_json(output / 'ROLLOUT_RESULT.json', metrics)
    return metrics


if __name__ == '__main__':
    worker()
