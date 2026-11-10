"""Measure forward and gradient differences on frozen native traces, without updating weights."""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .native_spans import action_offsets, action_records
from .span_diagnostic_data import boundary_cases, certify, longest, select_trajectories
from .probe_runtime import indexed, make_model, stats, trajectory_loss


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))['trajectories']


def selected(path, historical):
    original = load(path)
    if historical:
        chosen = [(i, t) for i, t in enumerate(original, 1) if boundary_cases(t['segments'])]
    else:
        if len(original) != 128:
            raise ValueError('Expected the frozen 128-trajectory batch')
        chosen = select_trajectories(original)
    return [dict(t, id=str(i)) for i, t in chosen]


def prepare(items, merge, rank, world):
    records = list(indexed(action_records(items, merge=merge)))
    local = records[rank::world]
    padding = min(records, key=lambda row: len(row[3]['prompt_ids'])+len(row[3]['completion_ids']))
    slots = []
    for i in range((len(records)+world-1)//world):
        valid = i < len(local)
        key, start, count, record = local[i] if valid else padding
        prompt = record['prompt_ids']
        ids = torch.tensor([prompt+record['completion_ids']], dtype=torch.long, device=rank)
        positions = torch.tensor(action_offsets(record), dtype=torch.long, device=rank)+len(prompt)
        slots.append((key, start, count, record, ids, positions, valid))
    return slots, len(records)


def synchronized_time():
    torch.cuda.synchronize()
    dist.barrier()
    return time.monotonic()


def score(model, slots, items, world):
    observations = []
    start = synchronized_time()
    with torch.no_grad():
        for key, offset, _, _, ids, positions, valid in slots:
            output = model(ids, positions)
            if valid:
                observations.append((key, offset, output.cpu().tolist()))
    elapsed = synchronized_time()-start
    shards = [None]*world
    dist.all_gather_object(shards, observations)
    result = {t['id']: torch.full((sum(len(x['completion_ids']) for x in t['segments']),), torch.nan) for t in items}
    for shard in shards:
        for key, offset, values in shard:
            target = result[key][offset:offset+len(values)]
            if target.numel() != len(values) or not target.isnan().all():
                raise ValueError('A scoring record overlapped or had invalid bounds')
            target.copy_(torch.tensor(values))
    if not all(torch.isfinite(row).all() for row in result.values()):
        raise ValueError('Incomplete or nonfinite action probability coverage')
    return result, elapsed


def baseline_behavior(items):
    return {t['id']: torch.tensor([lp for call in t['segments'] for lp in call['vllm_logprobs']]) for t in items}


def compare_scores(items, native, other, window=32, details=False):
    pools = {'all': [], 'near': [], 'away': []}
    cases, calls = [], []
    for t in items:
        key = t['id']
        left, right = native[key], other[key]
        delta = (left-right).abs()
        nearby = torch.zeros(len(delta), dtype=torch.bool)
        for case in boundary_cases(t['segments']):
            row = dict(case, trajectory_ordinal=int(key), anchors=[])
            for label in ('previous_action_index', 'next_action_index'):
                index = case[label]
                if index is None:
                    continue
                lo, hi = max(0,index-window), min(len(delta),index+window+1)
                nearby[lo:hi] = True
                row['anchors'].append({'label': label, 'action_index': index,
                                       'native_logprob': left[index].item(), 'other_logprob': right[index].item(),
                                       'window_absolute_error': stats(delta[lo:hi].tolist())})
            if details:
                cases.append(row)
        pools['all'].append(delta)
        pools['near'].append(delta[nearby])
        pools['away'].append(delta[~nearby])
        if details:
            offset = 0
            for i, call in enumerate(t['segments'], 1):
                count = len(call['completion_ids'])
                piece = delta[offset:offset+count]
                position = int(piece.argmax())
                calls.append({'trajectory_ordinal': int(key), 'call': i, 'native_prompt_tokens': len(call['prompt_ids']),
                              'max_absolute_error': piece[position].item(), 'action_position_in_call': position,
                              'near_boundary': nearby[offset+position].item(),
                              'native_logprob': left[offset+position].item(), 'other_logprob': right[offset+position].item()})
                offset += count
    result = {k: stats(torch.cat(pieces).tolist()) for k,pieces in pools.items()}
    if details:
        result.update(cases=cases, largest_call_errors=sorted(calls,key=lambda x:x['max_absolute_error'],reverse=True)[:12])
    return result


def gradient(model, slots, old, mode, rank, world, trajectories, tokens, calls):
    model.zero_grad(set_to_none=True)
    total = torch.zeros((), dtype=torch.float64, device=rank)
    start = synchronized_time()
    for key, offset, count, record, ids, positions, valid in slots:
        current = model(ids, positions)
        prior = old[key][offset:offset+count].to(rank) if valid else current.detach()
        if mode == 'global_token':
            denominator = tokens
        elif mode == 'trajectory_mean':
            denominator = trajectories*record['action_count']
        elif mode == 'call_mean':
            denominator = calls*count
        else:
            raise ValueError(mode)
        loss = trajectory_loss(current, prior, record['advantage'] if valid else 0., world, denominator, 1)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite diagnostic loss')
        loss.backward()
        total += loss.detach().double()
    elapsed = synchronized_time()-start
    dist.all_reduce(total)
    return {'loss': total.item()/world, 'forward_backward_seconds': elapsed}


def save_gradient(model):
    return {name: p.grad.detach().cpu() for name,p in model.named_parameters() if p.grad is not None}


def compare_gradient(model, reference, rank):
    current = {name: p.grad for name,p in model.named_parameters() if p.grad is not None}
    if current.keys() != reference.keys():
        raise ValueError('Gradient parameter coverage differs')
    result = torch.zeros(5,dtype=torch.float64)
    for name, before in reference.items():
        after = current[name].detach().cpu()
        for a,b in zip(before.flatten().split(1_000_000),after.flatten().split(1_000_000),strict=True):
            if not a.numel():
                continue
            a,b = a.double(),b.double()
            delta = b-a
            result[:4] += torch.stack((a.dot(a),b.dot(b),a.dot(b),delta.dot(delta)))
            result[4] = max(result[4],delta.abs().max())
    result = result.to(rank)
    dist.all_reduce(result[:4])
    dist.all_reduce(result[4:],op=dist.ReduceOp.MAX)
    aa,bb,ab,dd,peak = result.tolist()
    if not all(map(math.isfinite,(aa,bb,ab,dd,peak))) or aa <= 0 or bb <= 0:
        raise ValueError('Nonfinite or zero reference gradient')
    return {'reference_norm': aa**.5,'candidate_norm':bb**.5,'relative_l2':(dd/aa)**.5,
            'cosine':ab/(aa*bb)**.5,'max_absolute_error':peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--history',type=Path,nargs=2,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    hardware = torch.cuda.get_device_properties(rank)
    if 'A800' not in hardware.name or hardware.total_memory < 79*2**30:
        raise ValueError('Expected A800 80GB')
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    world = dist.get_world_size()
    if world != 4:
        raise ValueError('Expected four ranks')
    torch.manual_seed(20261109)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    items = selected(args.batch,False)
    native, calls = prepare(items,False,rank,world)
    merged, spans = prepare(items,True,rank,world)
    actions = certify(items)
    model, trainable, kernels, _ = make_model(rank)
    report = {'source_commit':args.source_commit,'weights':'frozen original step 0 model','trainable_parameters':trainable,
              'linear_attention_kernels':kernels,'optimizer_steps':0,'calls':calls,'spans':spans,'actions':actions,
              'selection':[{'ordinal':int(t['id']),'longest_tokens':longest(t),'boundaries':len(boundary_cases(t['segments']))} for t in items],
              'original_reduction':'sum of token loss divided by trajectory count times per-trajectory action count',
              'window_radius_action_tokens':32,'historical_vllm_hf_reference':{'mean':.0102,'p99':.1245}}
    def emit(name):
        if rank == 0:
            with (args.output/name).open('x') as target:
                json.dump(report,target,indent=2)
    old, native_time = score(model,native,items,world)
    again, native_repeat = score(model,native,items,world)
    merged_score, merged_time = score(model,merged,items,world)
    merged_again, merged_repeat = score(model,merged,items,world)
    difference = compare_scores(items,old,merged_score,details=True)
    report['forward'] = {'native_seconds':native_time,'native_repeat_seconds':native_repeat,'merged_seconds':merged_time,
                         'merged_repeat_seconds':merged_repeat,'merged_vs_native':difference,
                         'native_repeat':compare_scores(items,old,again),'merged_repeat':compare_scores(items,merged_score,merged_again),
                         'saved_vllm_vs_native':compare_scores(items,old,baseline_behavior(items)), 'historical_boundaries':{}}
    for job,path,count in zip((180554,181174),args.history,(3,1),strict=True):
        histories = selected(path,True)
        if sum(len(boundary_cases(t['segments'])) for t in histories) != count:
            raise ValueError('Historical boundary identity differs')
        certify(histories)
        h0,h1 = [prepare(histories,merge,rank,world)[0] for merge in (False,True)]
        p0,t0 = score(model,h0,histories,world)
        p1,t1 = score(model,h1,histories,world)
        report['forward']['historical_boundaries'][job] = {'native_seconds':t0,'merged_seconds':t1,
            **compare_scores(histories,p0,p1,details=True)}
    measured = difference['all']
    report['forward']['advance_to_gradients'] = measured['mean'] <= .0102 and measured['p99'] <= .1245
    emit('FORWARD.json')
    if not report['forward']['advance_to_gradients']:
        dist.destroy_process_group()
        return

    timings, comparisons = {},{}
    for label,slots,mode in (
        ('native_global',native,'global_token'),('native_global_repeat',native,'global_token'),
        ('merged_global',merged,'global_token'),('native_call_mean',native,'call_mean'),
        ('native_trajectory_mean',native,'trajectory_mean'),('merged_trajectory_mean',merged,'trajectory_mean'),
    ):
        timings[label] = gradient(model,slots,old,mode,rank,world,len(items),actions,calls)
        if label == 'native_global':
            global_grad = save_gradient(model)
        else:
            comparisons[label+'_vs_native_global'] = compare_gradient(model,global_grad,rank)
        if label == 'native_trajectory_mean':
            trajectory_grad = save_gradient(model)
        if label == 'merged_trajectory_mean':
            comparisons['merged_vs_native_original_trajectory_mean'] = compare_gradient(model,trajectory_grad,rank)
        if rank == 0:
            print(json.dumps({'completed':label,**timings[label]}),flush=True)
    report['gradient'] = {'timing':timings,'comparisons':comparisons,
        'fixed_denominator':actions,'fixed_old_logprob':'native step 0 scoring shared by all modes',
        'native_global_scoring_plus_forward_backward_seconds':native_time+timings['native_global']['forward_backward_seconds'],
        'merged_global_scoring_plus_forward_backward_seconds':merged_time+timings['merged_global']['forward_backward_seconds']}
    emit('RESULT.json')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
