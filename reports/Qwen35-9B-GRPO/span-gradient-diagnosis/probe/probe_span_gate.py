"""Reproduce the first DeltaNet gate discrepancy without FSDP or backward."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.nn import functional as F
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from .probe_span_backward_path import fixture, sharding_ablation, tensor_pair
from .probe_span_cause import selected
from .probe_runtime import MODEL, stats


def gate_projection(args):
    started = time.monotonic()
    data = json.loads(args.forward.read_text())
    values,case = fixture(selected(args.batch,False),data['forward']['merged_vs_native']['largest_call_errors'])
    inputs,positions = tensor_pair(values,0)
    root = Path(MODEL)
    config = AutoConfig.from_pretrained(root,local_files_only=True).text_config
    index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    def weight(name):
        key = 'model.language_model.'+name
        with safe_open(str(root/index[key]),framework='pt',device='cpu') as archive:
            return archive.get_tensor(key).to(device=0,dtype=torch.bfloat16)
    embedding = weight('embed_tokens.weight')
    norm = Qwen3_5RMSNorm(config.hidden_size,config.rms_norm_eps).to(device=0,dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(weight('layers.0.input_layernorm.weight'))
        hidden = [norm(F.embedding(ids,embedding)) for ids in inputs]
        del embedding
        projection = weight('layers.0.linear_attn.in_proj_a.weight')
        bias = weight('layers.0.linear_attn.dt_bias')
        decay = weight('layers.0.linear_attn.A_log').float().exp()
        short = inputs[0].shape[1]
        def gate(a):
            return -decay*F.softplus(a.float()+bias)
        def compare(a,b):
            delta = (a.float()-b[:,:short].float()).abs()
            return {'prefix_max':delta.max().item(),'prefix_mean':delta.mean().item(),
                    'matched_positions':stats(delta[:,positions-1].flatten().tolist())}
        result = {'source_commit':args.source_commit,'case':case,'lengths':[x.shape[1] for x in inputs],
                  'optimizer_steps':0,'normalized_input_prefix_exact':torch.equal(hidden[0],hidden[1][:,:short])}
        for dtype in (torch.bfloat16,torch.float32):
            a,b = [F.linear(x.to(dtype),projection.to(dtype)) for x in hidden]
            result[str(dtype)] = {'gate_projection':compare(a,b),'decay_gate':compare(gate(a),gate(b)),
                                  'decay_gate_with_identical_projected_prefix':compare(gate(a),gate(torch.cat((a,b[:,short:]),dim=1)))}
    torch.cuda.synchronize()
    result['gate_elapsed_seconds'] = time.monotonic()-started
    return result,hidden[1][:,:4096].cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('batch','forward','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    if 'A800' not in torch.cuda.get_device_name(rank):
        raise ValueError('Expected an A800 GPU')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    if dist.get_world_size()!=4:
        raise ValueError('Expected four ranks for the isolated sharding comparison')
    result,hidden = gate_projection(args) if rank==0 else (None,None)
    if rank==0:
        print(json.dumps({'gate':result}),flush=True)
    shared = [hidden]
    dist.broadcast_object_list(shared,src=0,device=torch.device('cuda',rank))
    started = time.monotonic()
    sharding = sharding_ablation(shared[0],rank)
    torch.cuda.synchronize()
    dist.barrier()
    if rank==0:
        result['sharding'] = sharding
        result['sharding_elapsed_seconds'] = time.monotonic()-started
        with args.output.open('x') as out:
            json.dump(result,out,indent=2)
        print(json.dumps({'sharding':sharding}),flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
