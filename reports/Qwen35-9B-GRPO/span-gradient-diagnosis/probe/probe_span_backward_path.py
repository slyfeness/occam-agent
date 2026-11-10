"""Isolate sequence shape, checkpointing and the live Gated DeltaNet kernel."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .native_spans import merge_exact_segments
from .probe_span_cause import compare_gradient, save_gradient, selected, synchronized_time
from .probe_runtime import MODEL, make_model, stats


def pair(item, call_number, action_position):
    calls = item['segments']
    call = calls[call_number-1]
    n = len(call['completion_ids'])
    short = call['prompt_ids']+call['completion_ids']
    prior = sum(len(c['completion_ids']) for c in calls[:call_number-1])
    offset = 0
    for span in merge_exact_segments(calls):
        end = offset+len(span['action_offsets'])
        if prior < end:
            long = span['prompt_ids']+span['completion_ids']
            if long[:len(short)] != short or prior+n > end:
                raise ValueError('Fixture does not preserve its native causal prefix')
            lo,hi = max(0,action_position-8),min(n,action_position+9)
            return short,long,list(range(len(call['prompt_ids'])+lo,len(call['prompt_ids'])+hi))
        offset = end
    raise ValueError('Call did not map to a span')


def fixture(items, outliers):
    by_id = {int(t['id']):t for t in items}
    for row in outliers:
        example = pair(by_id[row['trajectory_ordinal']],row['call'],row['action_position_in_call'])
        if len(example[0]) < len(example[1]):
            return example, row
    for item in items:
        for i in range(1,len(item['segments'])+1):
            example = pair(item,i,0)
            if len(example[0]) < len(example[1]):
                return example, {'trajectory_ordinal':int(item['id']),'call':i}
    raise ValueError('No prefix extension was available')


def short_fixture(items):
    candidates = []
    for item in items:
        for i,call in enumerate(item['segments'],1):
            if len(call['prompt_ids'])+len(call['completion_ids']) > 3072:
                continue
            short,long,positions = pair(item,i,len(call['completion_ids'])//2)
            long = long[:4096]
            if len(short) < len(long):
                candidates.append((short,long,positions))
    return max(candidates,key=lambda value:len(value[0])) if candidates else None


def tensor_pair(values, rank):
    short,long,positions = values
    return [torch.tensor([sequence],dtype=torch.long,device=rank) for sequence in (short,long)],torch.tensor(positions,dtype=torch.long,device=rank)


def full_backward(model, ids, positions):
    model.zero_grad(set_to_none=True)
    start = synchronized_time()
    logp = model(ids,positions)
    logp.float().mean().backward()
    return synchronized_time()-start


def traces(model, inputs, positions, rank):
    current = {}
    hooks = []
    operands = {}
    capturing = False
    linear = next(m for m in model.modules() if m.__class__.__name__ == 'Qwen3_5GatedDeltaNet')
    function = linear.chunk_gated_delta_rule

    def capture(*args,**kw):
        if rank == 0:
            values = (*args,kw['g'],kw['beta'])
            operands.setdefault('sampled',[]).append([x[:,positions-1].detach().cpu() for x in values])
            if capturing:
                operands['tensors'] = [x.detach().cpu() for x in values]
                operands['kwargs'] = {key:value for key,value in kw.items() if key not in ('g','beta')}
        return function(*args,**kw)

    def collect(label):
        def hook(module, args, output):
            if rank == 0:
                current[label] = output[0,positions-1].detach().cpu().float()
        return hook

    layer_count = 0
    for module in model.modules():
        if module.__class__.__name__ == 'Qwen3_5DecoderLayer':
            label = f'decoder_{layer_count:02d}_{module.layer_type}'
            hooks.append(module.register_forward_hook(collect(label)))
            if hasattr(module,'linear_attn'):
                hooks.append(module.linear_attn.register_forward_hook(collect(label+'_mixer')))
            layer_count += 1
    linear.chunk_gated_delta_rule = capture

    def capture_input(module, args, kwargs):
        if rank == 0:
            hidden = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            current['first_mixer_input'] = hidden[0,positions-1].detach().cpu().float()
            if capturing:
                operands['mixer_input'] = hidden[:,:4096].detach().cpu()

    hooks.append(linear.register_forward_pre_hook(capture_input, with_kwargs=True))
    hooks.append(linear.in_proj_qkv.register_forward_hook(collect('first_qkv_projection')))
    scores, layer = [],[]
    try:
        with torch.no_grad():
            for i,ids in enumerate(inputs):
                current = {}
                capturing = i == 1
                scores.append(model(ids,positions).detach().cpu())
                layer.append(current)
    finally:
        linear.chunk_gated_delta_rule = function
        for hook in hooks:
            hook.remove()
    differences = {label: (layer[1][label]-value).abs().max().item() for label,value in layer[0].items()}
    if rank==0:
        operands['sampled_operand_max_errors'] = {
            name:(a.float()-b.float()).abs().max().item() for name,a,b in
            zip(('q','k','v','g','beta'),*operands['sampled'],strict=True)}
    return stats((scores[0]-scores[1]).abs().tolist()),differences,operands


def isolated_fla(operands, cutoff, positions, rank):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    effects = []
    for cap in (cutoff,operands['tensors'][0].shape[1]):
        q,k,v,g,beta = [x[:,:cap].contiguous().to(rank).requires_grad_() for x in operands['tensors']]
        out,_ = chunk_gated_delta_rule(q,k,v,g=g,beta=beta,**operands['kwargs'])
        probe = out[:,positions-1].float()
        grads = torch.autograd.grad(probe.mean(),(q,k,v,g,beta))
        effects.append((probe.detach().cpu(),[g[:,:cutoff].cpu() for g in grads],
                        max((g[:,cutoff:].abs().max().item() for g in grads if g.shape[1]>cutoff),default=0.)))
    left,right = effects
    metrics = {}
    for name,a,b in zip(('q','k','v','g','beta'),left[1],right[1],strict=True):
        a,b = a.float(),b.float()
        norm = a.norm().item()
        metrics[name] = {'relative_l2':(b-a).norm().item()/norm if norm else None,'max':(b-a).abs().max().item()}
    return {'output_absolute_error':stats((left[0]-right[0]).abs().flatten().tolist()),
            'input_gradient_difference':metrics,'largest_future_operand_gradient':right[2]}


def sharding_ablation(hidden, rank):
    from safetensors import safe_open
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    model_path = Path(MODEL)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True).text_config
    index = json.loads((model_path/'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.language_model.layers.0.linear_attn.'
    files = {}
    for name, filename in index.items():
        if name.startswith(prefix):
            files.setdefault(filename,[]).append(name)
    weights = {}
    for filename,names in files.items():
        with safe_open(str(model_path/filename),framework='pt',device='cpu') as archive:
            for name in names:
                weights[name.removeprefix(prefix)] = archive.get_tensor(name).float()
    max_len = hidden.shape[1]
    min_len = max_len//2
    positions = torch.arange(min_len-32,min_len-16,device=rank)
    modes = {}
    for strategy in (ShardingStrategy.FULL_SHARD,ShardingStrategy.NO_SHARD):
        # Match the full model's FP32 master parameters before FSDP casts its forward.
        module = Qwen3_5GatedDeltaNet(config,layer_idx=0).float()
        module.load_state_dict(weights,strict=True)
        model = FSDP(module,device_id=rank,use_orig_params=True,sharding_strategy=strategy,
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,
                                                    buffer_dtype=torch.bfloat16))
        samples = []
        for cap in (min_len,max_len):
            model.zero_grad(set_to_none=True)
            output = model(hidden[:,:cap].contiguous().to(rank))[:,positions].float()
            output.mean().backward()
            with FSDP.summon_full_params(model,with_grads=True):
                samples.append((output.detach().cpu(),torch.cat([p.grad.detach().cpu().flatten() for p in model.parameters() if p.grad is not None])))
        modes[strategy.name] = samples
        del model,module
    def diff(a,b):
        return {'output_absolute_error':stats((a[0]-b[0]).abs().flatten().tolist()),
                'gradient_relative_l2':(a[1]-b[1]).norm().item()/a[1].norm().item()}
    shard,unshard = modes['FULL_SHARD'],modes['NO_SHARD']
    return {'lengths':[min_len,max_len],'full_shard_short_vs_long':diff(*shard),'no_shard_short_vs_long':diff(*unshard),
            'same_short_full_vs_no_shard':diff(shard[0],unshard[0]),'same_long_full_vs_no_shard':diff(shard[1],unshard[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('batch','forward','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    device = torch.cuda.get_device_properties(rank)
    if 'A800' not in device.name or device.total_memory < 79*2**30:
        raise ValueError('Expected A800 80GB')
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    if dist.get_world_size()!=4:
        raise ValueError('Expected four FSDP ranks')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261109)
    items = selected(args.batch,False)
    first = json.loads(args.forward.read_text())
    values,case = fixture(items,first['forward']['merged_vs_native']['largest_call_errors'])
    inputs,positions = tensor_pair(values,rank)
    model,_,kernels,_ = make_model(rank)
    start = synchronized_time()
    score,layers,operands = traces(model,inputs,positions,rank)
    trace_seconds = synchronized_time()-start
    result = {'source_commit':args.source_commit,'case':case,'lengths':[len(x) for x in values[:2]],
              'same_lm_head_positions':len(values[2]),'same_cotangent':'mean selected log-probs',
              'layer_output_max_error':layers,'matched_projection_logprob_difference':score,
              'linear_kernels':kernels,'model_sharding':str(model.sharding_strategy),'trace_seconds':trace_seconds,
              'optimizer_steps':0}
    if rank==0:
        result['first_fla_sampled_operand_max_errors'] = operands['sampled_operand_max_errors']
        print(json.dumps({'matched_forward':result}),flush=True)
        start = time.monotonic()
        torch.cuda.empty_cache()
        result['isolated_fla'] = isolated_fla(operands,len(values[0]),positions,rank)
        result['isolated_fla_seconds'] = time.monotonic()-start
        print(json.dumps({'isolated_fla':result['isolated_fla']}),flush=True)
    dist.barrier()
    short_time = full_backward(model,inputs[0],positions)
    short_grad = save_gradient(model)
    long_time = full_backward(model,inputs[1],positions)
    result['fixed_cotangent'] = {**compare_gradient(model,short_grad,rank),'short_seconds':short_time,'long_seconds':long_time}
    if rank==0:
        print(json.dumps({'fixed_cotangent':result['fixed_cotangent']}),flush=True)
    del short_grad

    reduced = short_fixture(items)
    if reduced:
        inputs,positions = tensor_pair(reduced,rank)
        on_short = full_backward(model,inputs[0],positions)
        on_grad = save_gradient(model)
        on_long = full_backward(model,inputs[1],positions)
        on_difference = compare_gradient(model,on_grad,rank)
        model.module.model.gradient_checkpointing_disable()
        if any(getattr(module,'gradient_checkpointing',False) for module in model.module.model.modules()):
            raise RuntimeError('Transformer checkpointing toggle did not take effect')
        off_short = full_backward(model,inputs[0],positions)
        checkpoint_effect = compare_gradient(model,on_grad,rank)
        del on_grad
        off_grad = save_gradient(model)
        off_long = full_backward(model,inputs[1],positions)
        result['checkpoint_ablation'] = {'lengths':[len(x) for x in reduced[:2]],'transformer_checkpoint_on':on_difference,
            'transformer_checkpoint_off':compare_gradient(model,off_grad,rank),'toggle_on_identical_short_input':checkpoint_effect,
            'seconds':[on_short,on_long,off_short,off_long]}
        del off_grad
        if rank==0:
            print(json.dumps({'checkpoint_ablation':result['checkpoint_ablation']}),flush=True)
    model.zero_grad(set_to_none=True)
    del model
    torch.cuda.empty_cache()
    distributed_input = [operands.get('mixer_input')]
    dist.broadcast_object_list(distributed_input,src=0,device=torch.device('cuda',rank))
    start = synchronized_time()
    result['first_linear_mixer_sharding_ablation'] = sharding_ablation(distributed_input[0],rank)
    result['sharding_ablation_seconds'] = synchronized_time()-start
    if rank==0:
        with args.output.open('x') as out:
            json.dump(result,out,indent=2)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
