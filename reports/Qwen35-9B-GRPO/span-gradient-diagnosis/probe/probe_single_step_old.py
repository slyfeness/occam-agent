"""Pair separate and detached old probabilities on the same exact native spans."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .probe_span_cause import compare_gradient, gradient, prepare, save_gradient, score, selected
from .single_step_old import DETACHED_MODE, old_probability, pre_update_probability_mode
from .span_diagnostic_data import certify, longest
from .probe_runtime import make_model, trajectory_loss


def check_detach_math():
    for advantage in (-2.,2.):
        lp = torch.tensor([-2.,-1.],requires_grad=True)
        prior = old_probability(lp,None)
        torch.testing.assert_close((lp-prior).exp(),torch.ones_like(lp),atol=0,rtol=0)
        torch.testing.assert_close(torch.autograd.grad(trajectory_loss(lp,prior,advantage,1,1,2),lp)[0],
                                   torch.full_like(lp,-advantage/2),atol=0,rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    props = torch.cuda.get_device_properties(rank)
    if 'A800' not in props.name or props.total_memory < 79*2**30:
        raise ValueError('Expected four A800 80GB devices')
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    world = dist.get_world_size()
    if world!=4:
        raise ValueError('Expected four FSDP ranks')
    torch.manual_seed(20261109)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    pre_update_probability_mode(DETACHED_MODE,('reward',))
    check_detach_math()
    items = selected(args.batch,False)
    action_count = certify(items)
    spans,records = prepare(items,True,rank,world)
    model,_,_,_ = make_model(rank)
    old,old_seconds = score(model,spans,items,world)
    metrics = {}
    for name,frozen in (('separate',old),('separate_repeat',old),('detached',None)):
        metrics[name] = gradient(model,spans,frozen,'trajectory_mean',rank,world,len(items),action_count,records)
        if name=='separate':
            ref = save_gradient(model)
        else:
            metrics[name]['gradient_vs_separate'] = compare_gradient(model,ref,rank)
        if rank==0:
            print(json.dumps({'phase':name,**metrics[name]}),flush=True)
    repeat = metrics['separate_repeat']['gradient_vs_separate']
    detached = metrics['detached']['gradient_vs_separate']
    limit = max(.03,3*repeat['relative_l2'])
    verified = detached['relative_l2'] <= limit and detached['cosine'] >= .99
    if rank==0:
        result = {'source_commit':args.source_commit,'optimizer_steps':0,'trajectories':len(items),
                  'longest_native_sequence':max(map(longest,items)),'action_tokens':action_count,'scoring_records':records,
                  'loss_reduction':'mean of per-trajectory action means','old_scoring_seconds':old_seconds,
                  'separate_total_seconds':old_seconds+metrics['separate']['forward_backward_seconds'],
                  'detached_total_seconds':metrics['detached']['forward_backward_seconds'],
                  'gradient_checks':metrics,'relative_l2_limit':limit,'cosine_minimum':.99,'verified':verified}
        with args.output.open('x') as handle:
            json.dump(result,handle,indent=2)
    dist.destroy_process_group()
    if not verified:
        raise RuntimeError('Offline detached gradient verification failed')


if __name__=='__main__':
    main()
