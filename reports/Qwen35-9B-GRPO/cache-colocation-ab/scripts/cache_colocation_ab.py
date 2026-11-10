"""Replay saved native chat prefixes with only the colocated vLLM budget changed."""

import argparse
import hashlib
import json
import os
import random
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def write_json(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def load_json(path):
    return json.loads(path.read_bytes())


def headroom(total, free, utilization):
    return free - total * utilization


def read_counters(metrics):
    totals = {}
    for name, value in re.findall(r'^vllm:([\w:]+)(?:\{[^\n]*\})?\s+([\d.eE+\-]+)$', metrics, re.M):
        totals[name] = totals.get(name, 0.0) + float(value)
    return totals


def summarize_trial(before, after, requests, tokens, elapsed):
    fields = {
        'prefix_lookups': 'prefix_cache_queries_total',
        'prefix_hits': 'prefix_cache_hits_total',
        'preemptions': 'num_preemptions_total',
        'engine_prompt_tokens': 'prompt_tokens_total',
        'engine_output_tokens': 'generation_tokens_total',
        'engine_requests': 'e2e_request_latency_seconds_count',
        'prefill_computed_tokens': 'request_prefill_kv_computed_tokens_sum',
    }
    for phase in ('prefill', 'decode', 'queue'):
        fields[f'{phase}_sum_seconds'] = f'request_{phase}_time_seconds_sum'
        fields[f'{phase}_count'] = f'request_{phase}_time_seconds_count'
    result = {key: after[name] - before.get(name, 0) if name in after else None
              for key, name in fields.items()}
    hits, lookups = result['prefix_hits'], result['prefix_lookups']
    result['prefix_hit_rate'] = hits / lookups if hits is not None and lookups else None
    for phase in ('prefill', 'decode', 'queue'):
        total, count = result[f'{phase}_sum_seconds'], result[f'{phase}_count']
        result[f'{phase}_mean_seconds'] = total / count if total is not None and count else None
    return dict(result, requests=requests, output_tokens=tokens, replay_seconds=elapsed,
                output_tokens_per_second=tokens / elapsed)


def load_requests(history):
    tasks = load_json(history / 'task_manifest.json')['tasks']
    if len(tasks) != 128 or len({task['run_id'] for task in tasks}) != 128:
        raise ValueError('unexpected_frozen_task_manifest')
    chains = []
    fingerprint = hashlib.sha256()
    historical_tokens = 0
    for task in tasks[1::4]:
        root = history / 'trajectories' / task['run_id']
        chain = []
        for step in load_json(root / 'trajectory.json')['rounds']:
            if not step['request_sent']:
                continue
            if not step['response_received']:
                raise ValueError('incomplete_saved_request')
            frozen = step['raw_response']['prompt_token_ids']
            accounting = step['token_accounting']
            if not frozen or frozen != accounting['prompt_token_ids']:
                raise ValueError('unverified_saved_prefix')
            body = (root / f"round_{step['round_id']:03d}" / 'raw_request.json').read_bytes()
            request = json.loads(body)
            if request.get('stream') or request.get('return_token_ids') is not True:
                raise ValueError('missing_token_id_capture')
            if (request['temperature'], request['top_p'], request['top_k']) != (1, 1, 0):
                raise ValueError('saved_sampling_configuration_changed')
            fingerprint.update(hashlib.sha256(body).digest())
            fingerprint.update(hashlib.sha256(json.dumps(frozen).encode()).digest())
            historical_tokens += step['raw_response']['usage']['completion_tokens']
            chain.append((body, frozen))
        if not chain:
            raise ValueError('empty_saved_trajectory')
        fingerprint.update(b'chain-end')
        chains.append(chain)
    lengths = [len(prefix) for chain in chains for _, prefix in chain]
    return chains, {'historical_job': 181505, 'historical_actor': 1, 'chains': len(chains),
                    'requests': len(lengths), 'prompt_tokens': sum(lengths),
                    'min_prompt_tokens': min(lengths), 'max_prompt_tokens': max(lengths),
                    'historical_output_tokens': historical_tokens, 'input_sha256': fingerprint.hexdigest()}


def verify_response(response, expected_prefix):
    if response.get('prompt_token_ids') != expected_prefix:
        raise ValueError('prefix_mismatch')
    count = response['usage']['completion_tokens']
    if not isinstance(count, int) or count < 0:
        raise ValueError('invalid_completion_count')
    return count


def http(url, body=None):
    request = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1800 if body is not None else 10) as response:
        return response.read()


class GPUSampler:
    def __init__(self, uuid):
        self.samples = []
        self.process = subprocess.Popen(
            ['nvidia-smi', '--id=' + uuid, '--query-gpu=memory.used',
             '--format=csv,noheader,nounits', '-lms', '250'], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        for line in self.process.stdout:
            value = line.strip()
            if value.isdigit():
                self.samples.append((time.monotonic(), int(value)))

    def peak(self, start=0, end=float('inf')):
        return max((value for ts, value in self.samples if start <= ts <= end), default=None)

    def close(self):
        self.process.terminate()
        self.process.wait(timeout=10)
        self.thread.join(timeout=10)


def replay(server, chains, total, progress):
    gate = threading.Event()
    stop = threading.Event()
    lock = threading.Lock()
    counter = [0, 0]

    def run_chain(chain):
        gate.wait()
        try:
            for body, prefix in chain:
                if stop.is_set():
                    return
                tokens = verify_response(json.loads(http(server.url + '/v1/chat/completions', body)), prefix)
                with lock:
                    counter[0] += 1
                    counter[1] += tokens
                    if counter[0] % 16 == 0 or counter[0] == total:
                        write_json(progress, dict(completed=counter[0], total=total, output_tokens=counter[1]))
        except Exception:
            stop.set()
            raise

    with ThreadPoolExecutor(max_workers=len(chains)) as pool:
        futures = [pool.submit(run_chain, chain) for chain in chains]
        start = time.monotonic()
        gate.set()
        for future in futures:
            future.result()
        end = time.monotonic()
    return start, end, counter


def encoder_worker(model, ready):
    import torch
    from tevatron.retriever.driver.encode import DenseModel

    initial_free, _ = torch.cuda.mem_get_info()
    encoder = DenseModel.load(str(model), pooling='eos', normalize=True, lora_name_or_path=None,
                              torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    device = torch.cuda.get_device_properties(0)
    write_json(ready, {'initial_free_bytes': initial_free, 'free_bytes': free, 'total_bytes': total,
                       'encoder_allocated_bytes': torch.cuda.memory_allocated(), 'device': device.name,
                       'uuid': str(device.uuid)})
    while encoder is not None:
        time.sleep(5)


def run_arm(server_class, output, gpu, uuid, utilization, chains, count):
    monitor = GPUSampler(uuid)
    server = None
    try:
        server = server_class(output, 65536, gpu=gpu, memory=utilization)
        server.wait()
        before = read_counters(http(server.url + '/metrics').decode())
        if before.get('prefix_cache_queries_total', 0):
            raise ValueError('cache_already_used_before_replay')
        start, end, (completed, tokens) = replay(server, chains, count, output / 'progress.json')
        for _ in range(20):
            after = read_counters(http(server.url + '/metrics').decode())
            if after.get('e2e_request_latency_seconds_count') == completed:
                break
            time.sleep(0.25)
        else:
            raise ValueError('engine_request_counter_mismatch')
        log = (output / 'server.log').read_text(encoding='utf-8', errors='replace')
        if 'enable_prefix_caching=True' not in log:
            raise ValueError('prefix_cache_not_enabled')
        graph = [float(seconds) for seconds in re.findall(r'Graph capturing finished in ([\d.]+) secs', log)]
        result = summarize_trial(before, after, completed, tokens, end - start)
        running = [int(n) for n in re.findall(r'Running: (\d+) reqs', log)]
        kv = []
        for sample in server.kv_samples:
            values = read_counters('\n'.join(sample['raw_metric_lines']))
            if 'num_requests_running' in values:
                running.append(int(values['num_requests_running']))
            if 'kv_cache_usage_perc' in values:
                kv.append(values['kv_cache_usage_perc'])
        return dict(result, status='completed', memory_utilization=utilization,
                    exact_prefix_matches=completed, peak_gpu_mib=monitor.peak(),
                    replay_peak_gpu_mib=monitor.peak(start, end), gpu_samples=len(monitor.samples),
                    startup_seconds=server.ready_seconds, graph_capture_seconds=sum(graph) if graph else None,
                    peak_running_sampled=max(running, default=None), peak_kv_fraction=max(kv, default=None))
    finally:
        try:
            if server is not None:
                server.metrics('final')
                server.stop()
        finally:
            monitor.close()


def main(args):
    from src.rollout import runtime
    import importlib.metadata

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    chains, manifest = load_requests(args.history)
    launch = args.history / 'phase10_server_graph_prefix_1.py'
    source = launch.read_bytes()
    manifest['historical_launcher_sha256'] = hashlib.sha256(source).hexdigest()
    manifest['probe_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(output / 'INPUT.json', manifest)
    result = {'input': manifest, 'slurm_job': os.environ.get('SLURM_JOB_ID'),
              'vllm_version': importlib.metadata.version('vllm'), 'trials': []}
    runtime(output, resident=True)  # Register the frozen runtime import locations.
    server_module = {'__name__': 'cache_ab_server', '__file__': str(launch)}
    from src.resident_rollout import resident_actor_env
    server_module['resident_actor_env'] = resident_actor_env
    exec(compile(source, str(launch), 'exec'), server_module)
    snapshots = output / 'empty_snapshots'
    snapshots.mkdir()
    os.environ.update(OCCAM_ACTOR_ADMIN=secrets.token_urlsafe(32),
                      OCCAM_ACTOR_GATE=str(output / 'unused_gate'), OCCAM_SNAPSHOT_DIR=str(snapshots))
    gpu, = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    ready = output / 'encoder_ready.json'
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(args.retrieval_compat), env.get('PYTHONPATH', '')))
    with (output / 'encoder.log').open('w', encoding='utf-8') as log:
        encoder = subprocess.Popen([sys.executable, __file__, '--encoder-worker', str(args.encoder), str(ready)],
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while not ready.exists():
                if encoder.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('encoder_startup_failed')
                time.sleep(1)
            preflight = load_json(ready)
            uuid = preflight.pop('uuid')
            if not uuid.startswith('GPU-'):
                uuid = 'GPU-' + uuid
            if 'A800' not in preflight['device']:
                raise RuntimeError('unexpected_gpu_type')
            result['memory_preflight'] = dict(preflight, minimum_idle_headroom_bytes=2 * 2**30,
                                             mode='encoder_loaded_idle')
            # Fix the single-pair order independently of any observed result.
            order = [('A', .50), ('B', .75)]
            random.Random(181505).shuffle(order)
            result['order'] = [label for label, _ in order]
            write_json(output / 'RESULT.json', result)
            for label, utilization in order:
                available = headroom(preflight['total_bytes'], preflight['free_bytes'], utilization)
                print(f'{label}: headroom after vLLM budget {available / 2**30:.2f} GiB', flush=True)
                if available < 2 * 2**30:
                    trial = dict(status='insufficient_idle_headroom', memory_utilization=utilization)
                else:
                    try:
                        trial = run_arm(server_module['Server'], output / label, gpu, uuid, utilization,
                                        chains, manifest['requests'])
                    except Exception as error:
                        trial = dict(status='error', error_type=type(error).__name__, memory_utilization=utilization)
                        result['trials'].append(dict(trial, arm=label, estimated_idle_headroom_bytes=available))
                        write_json(output / 'RESULT.json', result)
                        raise
                result['trials'].append(dict(trial, arm=label, estimated_idle_headroom_bytes=available))
                write_json(output / 'RESULT.json', result)
                print(f'{label}: {trial["status"]}', flush=True)
        finally:
            encoder.terminate()
            encoder.wait(timeout=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder-worker', nargs=2, type=Path)
    for name in ('history', 'output', 'encoder', 'retrieval-compat'):
        parser.add_argument('--' + name, type=Path)
    options = parser.parse_args()
    if options.encoder_worker:
        encoder_worker(*options.encoder_worker)
    else:
        if not all((options.history, options.output, options.encoder, options.retrieval_compat)):
            parser.error('controller requires history, output, encoder and retrieval-compat')
        main(options)
