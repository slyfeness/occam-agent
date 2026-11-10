"""Portable export of the measured GRPO training entry point."""
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .batch_shape import BatchShape
from .checkpoint import load_rank, save_rank
from .native_spans import action_offsets, action_records
from .policy_snapshot import export_policy_snapshot
from .probe_runtime import MODEL, make_model, stats, trajectory_loss
from .single_step_old import DETACHED_MODE, SEPARATE_MODE, old_probability, pre_update_probability_mode
from .span_diagnostic_data import certify


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    world = dist.get_world_size()
    assert world == 4
    torch.manual_seed(20261109)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    batch = json.loads(Path(os.environ['SMOKE_BATCH']).read_text())
    shape = BatchShape.from_config(batch)
    shape.verify_groups(t['query_id'] for t in batch['trajectories'])
    output = Path(os.environ['SMOKE_TRAIN_OUTPUT'])
    output.mkdir(parents=True,exist_ok=True)
    speed_retest = os.environ.get('SMOKE_SPEED_RETEST') == '1'
    optimizer_phases = ('reward',) if speed_retest else ('zero_advantage','reward')
    old_mode = pre_update_probability_mode(os.environ.get('GRPO_OLD_PROBABILITY_MODE',SEPARATE_MODE),optimizer_phases)
    fused_old = old_mode == DETACHED_MODE
    scoring_mode = os.environ.get('GRPO_SCORING_MODE', 'native_calls')
    if scoring_mode not in ('native_calls', 'exact_native_spans_v1'):
        raise ValueError(f'Unsupported GRPO scoring mode: {scoring_mode}')
    if scoring_mode == 'exact_native_spans_v1':
        certify(batch['trajectories'])
    records = action_records(batch['trajectories'], merge=scoring_mode == 'exact_native_spans_v1')
    for record in records:
        q = record.get('vllm_logprobs')
        if q is None or len(q) != len(action_offsets(record)) or not all(map(math.isfinite,q)):
            raise RuntimeError('Missing or nonfinite native vLLM probability diagnostic')
    local = records[rank::world]
    count = (len(records)+world-1)//world
    fsdp, trainable, kernels, kernel_calls = make_model(rank)
    measurements = []
    def measured(name, function, **metadata):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.time()
        try:
            return function()
        finally:
            torch.cuda.synchronize()
            row = {'phase':name,'start_unix':begin,'end_unix':time.time(),
                   'peak_allocated':torch.cuda.max_memory_allocated(),
                   'peak_reserved':torch.cuda.max_memory_reserved(), **metadata}
            measurements.append(row)
            with (output/f'TIMING-rank{rank}.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
    def inputs(record):
        prompt, response = record['prompt_ids'],record['completion_ids']
        assert prompt and response
        ids = torch.tensor([prompt+response],device=rank,dtype=torch.long)
        offsets = record.get('action_offsets')
        positions = (torch.arange(len(prompt),len(prompt)+len(response),device=rank) if offsets is None else
                     torch.tensor(offsets,device=rank,dtype=torch.long) + len(prompt))
        return ids, positions
    def evaluate(return_probe=False, phase='scoring'):
        result = []
        probes = []
        with torch.no_grad():
            for i in range(count):
                record = local[i] if i < len(local) else records[0]
                scored = measured(phase,lambda:fsdp(*inputs(record),return_probe=return_probe))
                lp = (scored[0] if return_probe else scored).detach().cpu()
                if i < len(local):
                    result.append(lp)
                    if return_probe:
                        probes.append(scored[1])
        return (result,probes) if return_probe else result
    start = time.monotonic()
    old = None if fused_old else evaluate(phase='old_logprob')
    repeated = None if speed_retest else evaluate(phase='noise_logprob')
    noise = [] if repeated is None else torch.cat([(b-a).abs() for a,b in zip(old,repeated)]).tolist()
    baseline = {name:tensor_sha(p) for name,p in fsdp.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([p for p in fsdp.parameters() if p.requires_grad],lr=1e-6,weight_decay=0.)
    diag = {'rank':rank,'trainable_text_parameters':trainable,'vision_trainable_parameters':0,
            'kernel_modules':kernels,'fsdp_decoder_class':'Qwen3_5DecoderLayer',
            'precision':'FP32 master parameters/Adam; BF16 forward; FP32 gradient reduction',
            'scoring_mode':scoring_mode,'scoring_records':len(records),
            'old_logprob_source':'detached reward forward' if fused_old else 'same training engine, pre-update',
            'old_probability_mode':old_mode,'vllm_logprob_usage':'diagnostics only',
            'optimizer_steps_per_batch':len(optimizer_phases),
            'noise':stats(noise),'max_native_sequence':max(len(x['prompt_ids'])+len(x['completion_ids']) for x in records)}
    q_errors = []
    def collect_q_error(record,lp):
        q = torch.tensor(record['vllm_logprobs'],dtype=torch.float64)
        q_errors.extend((lp.detach().cpu().double()-q).abs().tolist())
    def save_probability_diagnostic():
        if len(q_errors) != sum(len(action_offsets(record)) for record in local):
            raise RuntimeError('Incomplete action probability diagnostic before optimizer step')
        diag['vllm_vs_training_logprob_absolute_error'] = stats(q_errors)
        torch.save({'absolute_q_p_errors':q_errors,'absolute_repeat_errors':noise},output/f'probability-diagnostics-rank{rank}.pt')
    if not fused_old:
        for record, lp in zip(local,old,strict=True):
            collect_q_error(record,lp)
        save_probability_diagnostic()
    first = local[0]
    response_mask = [0]*len(first['completion_ids'])
    for offset in action_offsets(first):
        response_mask[offset] = 1
    diag['response_mask_example'] = {'prompt_tokens':len(first['prompt_ids']),
        'response_tokens':sum(response_mask), 'mask':[0]*len(first['prompt_ids'])+response_mask}
    print('TRAINING_OLD_PROBABILITY_MODE',rank,old_mode,diag['noise'],flush=True)
    step_counts = {'reward':0,'zero_advantage':0}
    for phase in optimizer_phases:
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        opt.zero_grad(set_to_none=True)
        losses = []
        for i in range(count):
            valid = i < len(local)
            record = local[i] if valid else records[0]
            metadata = {'input_tokens':len(record['prompt_ids'])+len(record['completion_ids']),
                        'action_tokens':len(action_offsets(record)), 'padding_record':not valid}
            lp = measured(phase+'_forward',lambda:fsdp(*inputs(record)),**metadata)
            prior = old_probability(lp,old[i].to(rank) if valid and old is not None else None)
            if fused_old and valid:
                collect_q_error(record,lp)
            adv = record['advantage'] if phase == 'reward' and valid else 0.
            loss = trajectory_loss(lp,prior,adv,world,shape.trajectories,record['action_count'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            measured(phase+'_backward',loss.backward,**metadata)
            losses.append(float(loss.detach()))
        if fused_old:
            save_probability_diagnostic()
        norm = fsdp.clip_grad_norm_(1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient')
        if phase == 'zero_advantage' and norm.item() != 0.:
            raise RuntimeError('Zero-advantage control has nonzero gradient')
        if fused_old and any(step_counts.values()):
            raise RuntimeError('Detached old probabilities cannot commit a second optimizer step')
        measured(phase+'_optimizer_step',opt.step)
        step_counts[phase] += 1
        hashes = {name:tensor_sha(p) for name,p in fsdp.named_parameters() if p.requires_grad}
        if phase == 'zero_advantage' and hashes != baseline:
            raise RuntimeError('Zero-advantage control changed parameters')
        diag[phase] = {'gradient_norm':norm.item(),'loss_sum':sum(losses),
                       'seconds':time.monotonic()-begin,'parameters_unchanged':hashes==baseline,
                       'peak_allocated':max(x['peak_allocated'] for x in measurements if x['phase'].startswith(phase)),
                       'peak_reserved':max(x['peak_reserved'] for x in measurements if x['phase'].startswith(phase))}
        if phase == 'zero_advantage':
            # Discard control Adam moments and step counter; main experiment starts fresh.
            opt = torch.optim.AdamW([p for p in fsdp.parameters() if p.requires_grad],lr=1e-6,weight_decay=0.)
    # Persist the one real update before any subsequent scoring can fail.
    path = output/f'checkpoint-rank{rank}.pt'
    torch.cuda.empty_cache()
    measured('checkpoint_save', lambda:save_rank(fsdp,opt,path,rank,world,1))
    dist.barrier()
    if rank == 0:
        (output/'CHECKPOINT_COMPLETE.json').write_text(json.dumps({'step':1,'world_size':world}))
    snapshot_directory = os.environ.get('OCCAM_SNAPSHOT_DIR')
    if snapshot_directory:
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        measured('export_inference_weights',
                 lambda: export_policy_snapshot(fsdp, MODEL, snapshot_directory, 1, rank))
    if speed_retest:
        diag['actual_kernel_calls'] = kernel_calls
        diag['total_seconds'] = time.monotonic()-start
        (output/f'RESULT-rank{rank}.json').write_text(json.dumps(diag,indent=2))
        dist.barrier()
        if rank == 0:
            (output/'COMPLETE.json').write_text(json.dumps({'optimizer_steps':step_counts['reward'],
                'zero_control_steps':step_counts['zero_advantage'],'world_size':world}))
        dist.destroy_process_group()
        return
    after, before_reload_probes = evaluate(return_probe=True,phase='post_update_logprob')
    diag['direction'] = [{'trajectory_id':r['trajectory_id'],'advantage':r['advantage'],
                          'tokens':a.numel(),'before_mean':a.mean().item(),'after_mean':b.mean().item(),
                          'change_sum':(b-a).double().sum().item(),'noise_abs_sum':(c-a).abs().double().sum().item()}
                         for r,a,b,c in zip(local,old,after,repeated)]
    (output/f'UPDATE-rank{rank}.json').write_text(json.dumps(diag,indent=2))
    with torch.no_grad():
        for parameter in fsdp.parameters():
            if parameter.requires_grad and parameter.numel():
                parameter.view(-1)[0].add_(1.)
                break
    measured('checkpoint_reload', lambda:load_rank(fsdp,opt,path,rank,world,1))
    reloaded, reloaded_probes = evaluate(return_probe=True,phase='reloaded_logprob')
    diag['reload_logprob_error'] = stats(torch.cat([(b-a).abs() for a,b in zip(after,reloaded)]).tolist())
    diag['reload_full_vocab_last_action_logits_error'] = stats(torch.cat([(b-a).abs() for a,b in zip(before_reload_probes,reloaded_probes)]).tolist())
    diag['cpu_process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    diag['actual_kernel_calls'] = kernel_calls
    diag['total_seconds'] = time.monotonic()-start
    (output/f'RESULT-rank{rank}.json').write_text(json.dumps(diag,indent=2))
    dist.barrier()
    if rank == 0:
        (output/'COMPLETE.json').write_text(json.dumps({'optimizer_steps':step_counts['reward'],
            'zero_control_steps':step_counts['zero_advantage'],'world_size':world}))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
