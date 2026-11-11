"""Run the frozen four-A800, ten-update GRPO stability experiment."""

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .resident_controller import fingerprint, generated_probe
from .resident_rollout import ActorAdmin, ResidentActors
from .rollout import STORE, graph_server_class, runtime, stop_services
from .stability import read_json, write_json
from .stability_judge import FlashJudge
from .stability_rollout import instrument_search_timing, run_rollout


OOM = re.compile(r'out of memory|OutOfMemoryError|less than desired GPU memory utilization', re.I)
COUNTERS = ('prefix_cache_queries_total', 'prefix_cache_hits_total', 'num_preemptions_total',
            'prompt_tokens_total', 'generation_tokens_total', 'request_prefill_time_seconds_sum',
            'request_prefill_time_seconds_count')


def query_metrics(server):
    with urllib.request.urlopen(server.url + '/metrics', timeout=10) as response:
        data = response.read().decode()
    found = {field: [] for field in COUNTERS}
    for field in COUNTERS:
        for value in re.findall(r'^vllm:' + field + r'(?:\{[^\n]*\})?\s+([\d.eE+\-]+)$', data, re.M):
            found[field].append(float(value))
    return {field: sum(values) if values else None for field, values in found.items()}


def metrics_delta(before, after):
    result = {field: value - (before[field] or 0) if value is not None else None
              for field, value in after.items()}
    queries, hits = result['prefix_cache_queries_total'], result['prefix_cache_hits_total']
    result['prefix_hit_rate'] = hits / queries if hits is not None and queries else None
    return result


class GPUMonitor:
    def __init__(self, ids):
        self.samples = {uuid: [] for uuid in ids}
        self.process = subprocess.Popen(
            ['nvidia-smi', '--id=' + ','.join(ids), '--query-gpu=uuid,memory.used',
             '--format=csv,noheader,nounits', '-lms', '250'], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()

    def collect(self):
        for line in self.process.stdout:
            uuid, _, size = line.strip().partition(', ')
            if uuid in self.samples and size.isdigit():
                self.samples[uuid].append((time.monotonic(), int(size)))

    def peaks(self, start, end):
        return [max((value for ts, value in samples if start <= ts <= end), default=None)
                for samples in self.samples.values()]

    def close(self):
        self.process.terminate()
        self.process.wait(timeout=10)
        self.thread.join(timeout=10)


class Services:
    def __init__(self, root, devices, secret, gate):
        self.root, self.devices, self.secret, self.gate = root, devices, secret, gate
        self.servers, self.retriever, self.log, self.port = [], None, None, None
        self.actors = self.search = None

    def start(self, memory, version):
        self.root.mkdir(mode=0o700)
        namespace = runtime(self.root, resident=True, retriever_instrumenter=instrument_search_timing)
        self.retriever, self.port, self.log = namespace['start_retriever'](self.root / 'retriever', self.devices[1])
        factory = graph_server_class(namespace, self.root, prefix_cache=True, resident=True)
        for i, gpu in enumerate(self.devices):
            self.servers.append(factory(self.root / f'actor{i}', 65536, gpu=gpu, memory=memory if i == 1 else .75))
        for server in self.servers:
            server.wait()
        workers = [ActorAdmin(server.url, self.secret) for server in self.servers]
        self.actors = ResidentActors(workers, self.gate)
        self.search = ActorAdmin(f'http://127.0.0.1:{self.port}', self.secret)
        if version is not None:
            self.actors.park()
            self.actors.publish(version['snapshot'], version['fingerprint'], validate=generated_probe)
        elif len(set(self.actors.all(fingerprint))) != 1:
            raise RuntimeError('Initial actor weights disagree')

    def offsets(self):
        return {log: log.stat().st_size for log in self.root.glob('*/server.log')}

    def memory_failures(self, offsets):
        failed = set()
        for log in self.root.glob('*/server.log'):
            with log.open('rb') as handle:
                handle.seek(offsets.get(log, 0))
                data = handle.read().decode(errors='replace')
            if OOM.search(data):
                failed.add(log.parent.name)
        return failed

    def close(self):
        stop_services(self.servers, self.retriever, self.log)
        self.gate.unlink(missing_ok=True)


def read_hardware():
    command = ("import json,torch; print(json.dumps([{'name':p.name,'bytes':p.total_memory,'uuid':str(p.uuid)} "
               "for p in (torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count()))]))")
    hardware = json.loads(subprocess.check_output([sys.executable, '-c', command], text=True))
    if len(hardware) != 4 or any('A800' not in item['name'] or item['bytes'] < 79*2**30 for item in hardware):
        raise RuntimeError('Expected four A800 80 GB devices')
    ids = [item['uuid'] if item['uuid'].startswith('GPU-') else 'GPU-'+item['uuid'] for item in hardware]
    return ids, [{k: item[k] for k in ('name', 'bytes')} for item in hardware]


def poll_training(process, result):
    while not result.is_file():
        if process.poll() is not None:
            if result.is_file():
                break
            raise RuntimeError('Four-rank training exited before writing its receipt')
        time.sleep(2)
    return read_json(result)


def main():
    key = os.environ.pop('JUDGE_API_KEY', '')
    if not key:
        raise RuntimeError('JUDGE_API_KEY missing from Slurm job environment')
    job = os.environ['SLURM_JOB_ID']
    root = STORE / 'runs' / f'grpo-stability-{job}'
    root.mkdir(mode=0o700)
    checkpoints, snapshots = root / 'checkpoints', root / 'policy-snapshots'
    checkpoints.mkdir()
    snapshots.mkdir()
    plan = read_json(Path(os.environ['OCCAM_STABILITY_PLAN']))
    ids = [qid for group in plan['training'] for qid in group]
    if len(ids) != 160 or len(set(ids)) != 160 or any(len(group) != 16 for group in plan['training']):
        raise ValueError('Frozen ten-step query schedule is incomplete')
    write_json(root / 'RUN_PLAN.json', {k: v for k, v in plan.items() if k != 'questions'})
    secret = secrets.token_urlsafe(32)
    gate = root / 'actor-gate'
    os.environ.update(OCCAM_ACTOR_ADMIN=secret, OCCAM_ACTOR_GATE=str(gate), OCCAM_SNAPSHOT_DIR=str(snapshots))
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    uuids, hardware = read_hardware()
    write_json(root / 'HARDWARE.json', hardware)
    monitor = GPUMonitor(uuids)
    judge = FlashJudge(key)
    del key
    trainer = services = None
    previous = []
    version = None
    memory = .75
    stage = 'startup'
    current = 0
    status = 'failed'
    started = time.monotonic()
    try:
        for step, group in enumerate(plan['training'], 1):
            current = step
            step_start = time.monotonic()
            phase = root / f'step-{step:02d}'
            phase.mkdir()
            retriever_startup = 0.
            fallback = []
            for attempt in (1, 2):
                stage = 'service_startup'
                offsets = {}
                try:
                    if services is None:
                        services = Services(phase / f'services-{attempt}', devices, secret, gate)
                        boot = time.monotonic()
                        services.start(memory, version)
                        retriever_startup += time.monotonic()-boot
                    offsets = services.offsets()
                    metrics_start = [query_metrics(server) for server in services.servers]
                    stage = 'rollout_and_api_judge'
                    rollout_dir = phase / f'rollout-{attempt}'
                    rows = [plan['questions'][qid] for qid in group]
                    rollout = run_rollout(rollout_dir, step, rows, 2026110900, services.servers, services.port,
                                          services.root / 'retriever/TOOL_TIMING.jsonl', judge)
                    memory_errors = services.memory_failures(offsets)
                    if memory_errors:
                        raise RuntimeError('Memory failure reported during the rollout')
                    actor_stats = [metrics_delta(begin, query_metrics(server))
                                   for begin, server in zip(metrics_start, services.servers, strict=True)]
                    online_peaks = monitor.peaks(rollout['generation_start'], rollout['generation_end'])
                    if any(peak is None for peak in online_peaks):
                        raise RuntimeError('Online GPU memory samples are missing')
                    break
                except Exception:
                    failures = services.memory_failures(offsets)
                    fallback_ok = failures and failures <= {'actor1', 'retriever'} and memory == .75 and attempt == 1
                    if not fallback_ok:
                        raise
                    fallback.append({'from': .75, 'to': .65, 'stage': stage, 'affected': sorted(failures),
                                     'seconds_before_retry': time.monotonic()-step_start})
                    write_json(phase / 'OOM_FALLBACK.json', fallback)
                    services.close()
                    services = None
                    memory = .65

            stage = 'park_services'
            park = time.monotonic()
            original_search = services.search.post('/occam-residency', {'action': 'probe'})
            services.actors.park()
            if services.search.post('/occam-residency', {'action': 'cpu'})['device'] != 'cpu':
                raise RuntimeError('Retriever failed to offload')
            park_seconds = time.monotonic()-park
            private_env = dict(os.environ)
            private_env.pop('OCCAM_ACTOR_ADMIN', None)
            local_seconds = 0.
            agreement = None
            if step == 1:
                stage = 'local_judge'
                local_start = time.monotonic()
                judge_env = dict(private_env, CUDA_VISIBLE_DEVICES=','.join(devices[:2]),
                                 OCCAM_LOCAL_JUDGE_INPUT=str(rollout_dir / 'LOCAL_JUDGE_INPUT.json'),
                                 OCCAM_LOCAL_JUDGE_MODEL=str(STORE.parents[1] / 'models/Qwen3-32B'))
                with (phase / 'local_judge.log').open('w', encoding='utf-8') as log:
                    subprocess.run([sys.executable, '-u', '-m', 'src.stability_judge'], env=judge_env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                pairs = read_json(rollout_dir / 'LOCAL_JUDGE.json')
                if len(pairs) != 128:
                    raise RuntimeError('Local judge did not cover all 128 answers')
                valid = [row for row in pairs if row['agree'] is not None]
                normal = [row for row in valid if row['termination'] == 'MODEL_FINAL_STOP']
                agreement = {'attempted': 128, 'comparable': len(valid), 'agreements': sum(r['agree'] for r in valid),
                             'normal_comparable': len(normal), 'normal_agreements': sum(r['agree'] for r in normal)}
                local_seconds = time.monotonic()-local_start
                write_json(phase / 'JUDGE_AGREEMENT.json', agreement)

            stage = 'fsdp_update'
            train_start = time.monotonic()
            if trainer is None:
                with (root / 'trainer.log').open('w', encoding='utf-8') as log:
                    trainer = subprocess.Popen(
                        [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                         '-m', 'src.stability_train'], stdout=log, stderr=subprocess.STDOUT,
                        env=dict(private_env, OCCAM_STABILITY_RUN=str(root)), start_new_session=True)
            write_json(root / f'train-{step:02d}.json', {
                'step': step, 'batch': str(rollout_dir / 'TRAIN_BATCH.json'), 'previous': previous,
                'empty_or_invalid_rate': rollout['empty_or_invalid_rate'],
                'repetition_triggered': rollout['repetition_triggered'],
            })
            trained = poll_training(trainer, root / f'trained-{step:02d}.json')
            train_end = time.monotonic()
            train_peaks = monitor.peaks(train_start, train_end)
            if any(peak is None for peak in train_peaks):
                raise RuntimeError('Training GPU memory samples are missing')
            report = dict(rollout, step=step, query_ids=group, train=trained, actor_stats=actor_stats,
                          memory_utilization_colocated=memory, colocated_peak_mib=online_peaks[1],
                          train_peak_nvml_mib=train_peaks, startup_seconds=retriever_startup,
                          park_seconds=park_seconds, local_judge_seconds=local_seconds,
                          train_wall_seconds=train_end-train_start, oom_fallbacks=fallback, agreement=agreement)
            write_json(phase / 'STEP_RESULT.json', report)
            if trained['stop_reason']:
                status = 'early_stop'
                break
            previous.append({key: trained[key] for key in ('grad_norm', 'entropy', 'empty_or_invalid_rate')})

            stage = 'policy_sync'
            sync_start = time.monotonic()
            version = read_json(snapshots / f'step-{step}.json')
            services.actors.publish(version['snapshot'], version['fingerprint'], validate=generated_probe)
            report['sync_seconds'] = time.monotonic()-sync_start
            stage = 'restore_search'
            if services.search.post('/occam-residency', {'action': 'cuda'})['device'] != 'cuda':
                raise RuntimeError('Retriever failed to return to GPU')
            restored = services.search.post('/occam-residency', {'action': 'probe'})
            if original_search['result_fingerprint'] != restored['result_fingerprint']:
                raise RuntimeError('Retriever probe changed after residency transfer')
            report['wall_seconds'] = time.monotonic()-step_start
            report['other_seconds'] = (report['wall_seconds'] - report['generation_seconds'] -
                                       report['judge_tail_seconds'] - report['local_judge_seconds'] -
                                       trained['forward_backward_seconds'] - trained['optimizer_seconds'] -
                                       trained['checkpoint_seconds'] - trained['export_seconds'] - report['sync_seconds'])
            report['policy_synced'] = True
            write_json(phase / 'STEP_RESULT.json', report)
            print(json.dumps({'step': step, 'elapsed_seconds': report['wall_seconds'],
                              'reward': report['reward_mean'], 'policy_synced': True}), flush=True)
        else:
            status = 'complete'
    finally:
        write_json(root / 'STATUS.json', {'state': status, 'job': job, 'current_step': current, 'stage': stage,
                                         'optimizer_steps': len(previous),
                                         'seconds': time.monotonic()-started})
        if trainer is not None and trainer.poll() is None:
            os.killpg(trainer.pid, signal.SIGTERM)
            trainer.wait(timeout=90)
        if services is not None:
            services.close()
        judge.close()
        monitor.close()


if __name__ == '__main__':
    main()
