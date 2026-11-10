"""Compare native-span precision without optimizer steps on frozen GRPO inputs."""
import argparse
import gc
import json
import os
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from .probe_span_cause import compare_gradient, compare_scores, gradient, load, prepare, save_gradient, score, selected
from .span_diagnostic_data import certify, longest
from .span_precision_data import noise_halves, short_reference_items
from .probe_runtime import make_model


def mixers(model):
    result = [m for m in model.modules() if isinstance(m,qwen.Qwen3_5GatedDeltaNet)]
    if not result:
        raise ValueError('No Qwen Gated DeltaNet modules were found')
    return result


def require_fp32_reference(model):
    """Use PyTorch reference math throughout, including the Gated DeltaNet."""
    hf = model.module.model
    hf.set_attn_implementation('eager')
    if hf.config.text_config._attn_implementation != 'eager':
        raise ValueError('The FP32 reference must use eager attention')
    activations = set()

    def audit(module, inputs, output):
        if output.dtype != torch.float32:
            raise ValueError(f'Unexpected reference activation dtype in {type(module).__name__}: {output.dtype}')
        activations.add(type(module).__name__)

    for module in mixers(model):
        module.causal_conv1d_fn = None
        module.causal_conv1d_update = qwen.torch_causal_conv1d_update
        module.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
        module.recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule
        module.norm.variance_epsilon = module.layer_norm_epsilon
        module.norm.forward = MethodType(qwen.Qwen3_5RMSNormGated.forward,module.norm)
    for module in model.modules():
        if isinstance(module,(qwen.Qwen3_5DecoderLayer,qwen.Qwen3_5GatedDeltaNet,torch.nn.Linear,torch.nn.Embedding)):
            module.register_forward_hook(audit)
    return activations


def enable_fp32_gate_projections(model):
    names = []

    def project(module, inputs):
        bias = None if module.bias is None else module.bias.float()
        with torch.autocast('cuda',enabled=False):
            output = F.linear(inputs.float(),module.weight.float(),bias)
        if output.dtype != torch.float32:
            raise ValueError('Gate projection did not produce FP32')
        return output

    def beta_projection(module, inputs):
        # Preserve the baseline dtype of beta and the FLA kernel inputs.
        return project(module,inputs).to(inputs.dtype)

    for layer in mixers(model):
        layer.in_proj_a.forward = MethodType(project,layer.in_proj_a)
        layer.in_proj_b.forward = MethodType(beta_projection,layer.in_proj_b)
        names.extend(f'layers.{layer.layer_idx}.linear_attn.{name}' for name in ('in_proj_a','in_proj_b'))
    return names


def peak(rank,world):
    local = [torch.cuda.max_memory_allocated(rank),torch.cuda.max_memory_reserved(rank)]
    rows = [None]*world
    dist.all_gather_object(rows,local)
    return {'allocated_bytes_by_rank':[row[0] for row in rows],'reserved_bytes_by_rank':[row[1] for row in rows]}


def measure_score(model, layout, items, rank, world):
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    values, seconds = score(model,layout,items,world)
    return values,{'seconds':seconds,**peak(rank,world)}


def measure_gradient(model, layout, old, items, rank, world, calls, tokens):
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    timing = gradient(model,layout,old,'global_token',rank,world,len(items),tokens,calls)
    return {**timing,**peak(rank,world)}


def announce(rank,label,data):
    if rank==0:
        print(json.dumps({'stage':label,**data}),flush=True)


def layout(model, items, plan, old, rank, world, calls, tokens):
    values,scoring = measure_score(model,plan,items,rank,world)
    backward = measure_gradient(model,plan,values if old is None else old,items,rank,world,calls,tokens)
    return values,{'scoring':scoring,'gradient':backward,
                   'scoring_plus_forward_backward_seconds':scoring['seconds']+backward['forward_backward_seconds']}


def difference(items, left, right):
    return compare_scores(items,left,right)['all']


def reference_experiment(batch,rank,world):
    items = short_reference_items(batch)
    tokens = certify(items)
    native,calls = prepare(items,False,rank,world)
    spans,span_count = prepare(items,True,rank,world)
    result = {'trajectory_lengths':[longest(t) for t in items],'actions':tokens,'native_calls':calls,'spans':span_count}
    announce(rank,'a_selection',result)
    model,_,_,_ = make_model(rank,precision=torch.float32)
    activation_classes = require_fp32_reference(model)
    old,result['fp32_native'] = layout(model,items,native,None,rank,world,calls,tokens)
    reference = save_gradient(model)
    announce(rank,'a_fp32_reference',result['fp32_native'])
    fp32_merged,result['fp32_spans_control'] = layout(model,items,spans,old,rank,world,calls,tokens)
    result['fp32_spans_control']['gradient_vs_fp32_native'] = compare_gradient(model,reference,rank)
    result['fp32_spans_control']['logprob_vs_fp32_native'] = difference(items,old,fp32_merged)
    result['observed_fp32_activation_classes'] = sorted(activation_classes)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model,_,_,_ = make_model(rank)
    for label,plan in (('bf16_native',native),('bf16_spans',spans)):
        values,result[label] = layout(model,items,plan,old,rank,world,calls,tokens)
        result[label]['gradient_vs_fp32_native'] = compare_gradient(model,reference,rank)
        result[label]['logprob_vs_fp32_native'] = difference(items,old,values)
        announce(rank,'a_'+label,result[label])
    native_error = result['bf16_native']['gradient_vs_fp32_native']['relative_l2']
    merged_error = result['bf16_spans']['gradient_vs_fp32_native']['relative_l2']
    result['span_to_native_relative_l2_ratio'] = merged_error/native_error if native_error else None
    result['noninferior_at_1_2x'] = merged_error <= 1.2*native_error
    return result


def gate_experiment(path,rank,world):
    items = selected(path,False)
    tokens = certify(items)
    native,calls = prepare(items,False,rank,world)
    spans,span_count = prepare(items,True,rank,world)
    model,_,_,_ = make_model(rank)
    result = {'trajectory_lengths':[longest(t) for t in items],'actions':tokens,'native_calls':calls,'spans':span_count}
    old,result['bf16_native'] = layout(model,items,native,None,rank,world,calls,tokens)
    baseline = save_gradient(model)
    announce(rank,'b_bf16_native',result['bf16_native'])
    base_span,result['bf16_spans'] = layout(model,items,spans,old,rank,world,calls,tokens)
    result['bf16_spans']['gradient_vs_bf16_native'] = compare_gradient(model,baseline,rank)
    result['bf16_spans']['logprob_vs_bf16_native'] = difference(items,old,base_span)
    announce(rank,'b_bf16_spans',result['bf16_spans'])
    result['fp32_projection_modules'] = enable_fp32_gate_projections(model)
    result['beta_boundary_dtype'] = 'FP32 projection; cast to input BF16 before sigmoid'
    patched_native,result['gate_fp32_native'] = layout(model,items,native,old,rank,world,calls,tokens)
    result['gate_fp32_native']['gradient_vs_bf16_native'] = compare_gradient(model,baseline,rank)
    del baseline
    reference = save_gradient(model)
    announce(rank,'b_gate_fp32_native',result['gate_fp32_native'])
    patched_span,result['gate_fp32_spans'] = layout(model,items,spans,old,rank,world,calls,tokens)
    result['gate_fp32_spans']['gradient_vs_gate_fp32_native'] = compare_gradient(model,reference,rank)
    result['gate_fp32_spans']['logprob_vs_gate_fp32_native'] = difference(items,patched_native,patched_span)
    announce(rank,'b_gate_fp32_spans',result['gate_fp32_spans'])
    return result


def noise_experiment(batch,rank,world):
    model,_,_,_ = make_model(rank)
    result = {'seed':20261110,'queries':3,'trajectories_per_query_per_half':4,
              'halves':[],'advantages':'recomputed within each saved-reward half and query'}
    for i,items in enumerate(noise_halves(batch)):
        tokens = certify(items)
        native,calls = prepare(items,False,rank,world)
        _,row = layout(model,items,native,None,rank,world,calls,tokens)
        row.update(trajectories=len(items),actions=tokens,calls=calls,
                   nonzero_advantages=sum(t['advantage']!=0 for t in items))
        result['halves'].append(row)
        if i==0:
            reference = save_gradient(model)
        announce(rank,f'noise_half_{i+1}',row)
    if all(h['nonzero_advantages'] for h in result['halves']):
        result['half_2_vs_half_1'] = compare_gradient(model,reference,rank)
        row = result['half_2_vs_half_1']
        row['symmetric_relative_l2'] = 2*row['relative_l2']*row['reference_norm']/(row['reference_norm']+row['candidate_norm'])
    else:
        result['half_2_vs_half_1'] = None
        result['zero_gradient_half'] = [i+1 for i,h in enumerate(result['halves']) if not h['nonzero_advantages']]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=('a','b','noise'),required=True)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    gpu = torch.cuda.get_device_properties(rank)
    if 'A800' not in gpu.name or gpu.total_memory < 79*2**30:
        raise ValueError('Expected A800 80GB')
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    world = dist.get_world_size()
    if world != 4:
        raise ValueError('Expected four FSDP ranks')
    torch.manual_seed(20261109)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    batch = load(args.batch) if args.experiment!='b' else None
    if batch is not None and len(batch)!=128:
        raise ValueError('Expected the frozen 181229 batch of 128 trajectories')
    if args.experiment=='a':
        result = reference_experiment(batch,rank,world)
    elif args.experiment=='b':
        result = gate_experiment(args.batch,rank,world)
    else:
        result = noise_experiment(batch,rank,world)
    result.update(experiment=args.experiment,source_commit=args.source_commit,optimizer_steps=0)
    if rank==0:
        with args.output.open('x') as out:
            json.dump(result,out,indent=2)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
