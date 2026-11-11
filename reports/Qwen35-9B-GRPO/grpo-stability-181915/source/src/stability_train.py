"""Four-rank worker for one update per batch with Adam and RNG retained in RAM."""

import gc
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .checkpoint import save_rank, sharded_state
from .native_spans import action_offsets, action_records
from .policy_snapshot import export_policy_snapshot
from .span_diagnostic_data import certify
from .stability import read_json, stop_reason, tis_weights, write_json
from .train_smoke import MODEL, make_model, trajectory_loss


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def sync_max(value, rank):
    tensor = torch.tensor(value, device=rank, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def run_step(rank, world, command, saved, snapshots, checkpoints):
    started = time.monotonic()
    batch = read_json(Path(command['batch']))
    included = [t for t in batch['trajectories'] if t['advantage'] is not None]
    count = len(included)
    certify(included)
    records = action_records(included, merge=True)
    if not records:
        raise RuntimeError('No action tokens with a valid reward')
    for record in records:
        probs = record.get('vllm_logprobs')
        if probs is None or len(probs) != len(action_offsets(record)) or not all(map(math.isfinite, probs)):
            raise RuntimeError('Missing or nonfinite native action probability')
    local = records[rank::world]
    number = (len(records) + world - 1) // world
    fallback = min(records, key=lambda r: len(r['prompt_ids']) + len(r['completion_ids']))
    torch.cuda.reset_peak_memory_stats()
    model, trainable, kernels, calls = make_model(rank)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6, weight_decay=0.)
    if saved:
        actor_state, optimizer_state, cpu_rng, cuda_rng = saved
        with sharded_state(model):
            model.load_state_dict(actor_state)
        optimizer.load_state_dict(optimizer_state)
        restored_steps = [int(state['step'].item()) for state in optimizer.state.values()]
        if not restored_steps or max(restored_steps) != command['step'] - 1:
            raise RuntimeError('Retained Adam state does not match the preceding update')
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        del actor_state, optimizer_state
        saved.clear()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    compute_start = time.monotonic()
    diagnostic = []
    sums = torch.zeros(4, device=rank, dtype=torch.float64)
    for index in range(number):
        valid = index < len(local)
        record = local[index] if valid else fallback
        prompt, completion = record['prompt_ids'], record['completion_ids']
        ids = torch.tensor([prompt + completion], device=rank, dtype=torch.long)
        positions = torch.tensor(action_offsets(record), device=rank, dtype=torch.long) + len(prompt)
        current, entropy = model(ids, positions, observe_entropy=True)
        sampled = torch.tensor(record['vllm_logprobs'], device=rank, dtype=current.dtype)
        importance, clipped, errors = tis_weights(current, sampled)
        advantage = record['advantage'] if valid else 0.
        loss = trajectory_loss(current, current.detach(), advantage, world, count,
                               record['action_count'], importance=importance)
        finite = torch.isfinite(loss) & torch.isfinite(entropy).all() & torch.isfinite(errors).all()
        if sync_max(~finite, rank):
            return {'stop_reason': 'nonfinite_loss_or_entropy', 'optimizer_step': False}, []
        loss.backward()
        if valid:
            diagnostic.append(errors.detach().cpu())
            sums[0] += loss.detach().double()
            sums[1] += entropy.detach().sum(dtype=torch.float64)
            sums[2] += entropy.numel()
            sums[3] += clipped.sum()
        del loss, current, entropy, ids, positions, sampled, importance, clipped, errors, finite
    grad_norm = model.clip_grad_norm_(1.).item()
    torch.cuda.synchronize()
    forward_backward = sync_max(time.monotonic() - compute_start, rank)
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    loss_sum, entropy_sum, token_count, clipped_count = sums.tolist()
    q_errors = torch.cat(diagnostic) if diagnostic else torch.empty(0)
    parts = [None] * world if rank == 0 else None
    dist.gather_object(q_errors, parts, dst=0)
    metrics = {'loss': loss_sum / world, 'entropy': entropy_sum / token_count, 'grad_norm': grad_norm,
               'action_tokens': int(token_count), 'tis_clip_fraction': clipped_count / token_count,
               'empty_or_invalid_rate': command['empty_or_invalid_rate'],
               'repetition_triggered': command['repetition_triggered'],
               'forward_backward_seconds': forward_backward,
               'native_spans': len(records), 'trainable_parameters': trainable}
    if rank == 0:
        errors = torch.cat(parts).double()
        if errors.numel() != token_count:
            raise RuntimeError('TIS diagnostics lost action tokens')
        metrics['tis_absolute_logprob_error'] = {'mean': errors.mean().item(),
                                                 'p99': torch.quantile(errors, .99).item(),
                                                 'max': errors.max().item()}
    reason = stop_reason(command['previous'], metrics)
    train_peaks = [{'rank': rank, 'allocated_bytes': torch.cuda.max_memory_allocated(),
                    'reserved_bytes': torch.cuda.max_memory_reserved()}]
    if not reason:
        update_start = time.monotonic()
        optimizer.step()
        torch.cuda.synchronize()
        metrics['optimizer_seconds'] = sync_max(time.monotonic() - update_start, rank)
        optimizer_steps = [int(state['step'].item()) for state in optimizer.state.values()]
        metrics['adam_state_step_min'] = min(optimizer_steps)
        metrics['adam_state_step_max'] = max(optimizer_steps)
        train_peaks[0].update(allocated_bytes=torch.cuda.max_memory_allocated(),
                              reserved_bytes=torch.cuda.max_memory_reserved())
    peaks = [None] * world if rank == 0 else None
    dist.gather_object(train_peaks[0], peaks, dst=0)
    metrics.update(optimizer_step=reason is None, stop_reason=reason, peak_memory=peaks,
                   kernels=kernels, kernel_calls=calls)
    if reason:
        return metrics, []

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    step = command['step']
    if step in (5, 10):
        save_start = time.monotonic()
        folder = checkpoints / f'step-{step:02d}'
        folder.mkdir(exist_ok=True)
        save_rank(model, optimizer, folder / f'checkpoint-rank{rank}.pt', rank, world, step)
        dist.barrier()
        if rank == 0:
            write_json(folder / 'COMPLETE.json', {'step': step, 'world_size': world})
        metrics['checkpoint_seconds'] = sync_max(time.monotonic() - save_start, rank)
    else:
        metrics['checkpoint_seconds'] = 0.
    export_start = time.monotonic()
    export_policy_snapshot(model, MODEL, snapshots, step, rank, verify_disk=True)
    dist.barrier()
    metrics['export_seconds'] = sync_max(time.monotonic() - export_start, rank)
    if step < 10:
        with sharded_state(model):
            actor_state = model.state_dict()
        next_state = [actor_state, cpu_copy(optimizer.state_dict()), torch.get_rng_state(), torch.cuda.get_rng_state()]
    else:
        next_state = []
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    metrics['worker_wall_seconds'] = time.monotonic() - started
    return metrics, next_state


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError('Expected the frozen four-rank training topology')
    torch.manual_seed(20261109)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    root = Path(os.environ['OCCAM_STABILITY_RUN'])
    saved = []
    try:
        for step in range(1, 11):
            command = root / f'train-{step:02d}.json'
            while not command.exists():
                time.sleep(1)
            payload = read_json(command)
            if payload['step'] != step:
                raise RuntimeError('Unexpected optimizer step identity')
            output, saved = run_step(rank, world, payload, saved, root / 'policy-snapshots', root / 'checkpoints')
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
            if rank == 0:
                write_json(root / f'trained-{step:02d}.json', output)
            if not output['optimizer_step']:
                break
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
