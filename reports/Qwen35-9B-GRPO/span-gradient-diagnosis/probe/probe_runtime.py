"""Shared model and loss helpers extracted from the measured diagnostic runtime."""
import collections
import functools
import os
from typing import Callable, Iterable

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer


MODEL = os.environ.get('OCCAM_STEP0_MODEL')


def instrument_linear_attention(modules: Iterable[object]) -> tuple[dict[str, int], Callable[[], None]]:
    """Count calls through the functions bound on live Gated DeltaNet modules."""
    counters = {"chunk_gated_delta_rule": 0, "recurrent_gated_delta_rule": 0}
    originals: list[tuple[object, str, object]] = []
    for module in modules:
        for name in counters:
            function = getattr(module, name, None)
            if function is None:
                continue
            originals.append((module, name, function))

            def counted(*args: object, _function: object = function, _name: str = name, **kwargs: object) -> object:
                counters[_name] += 1
                return _function(*args, **kwargs)  # type: ignore[operator]

            setattr(module, name, counted)

    def restore() -> None:
        for module, name, function in originals:
            setattr(module, name, function)

    return counters, restore


class ActionScorer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, positions, return_probe=False):
        hidden = self.model.model(input_ids=ids, use_cache=False)[0][0]
        results = []
        for pos in positions.split(32):
            # Full vocabulary normalization, only at causal action positions.
            def project(h, target):
                logits = self.model.lm_head(h).float()
                return logits.gather(-1,target[:,None])[:,0]-torch.logsumexp(logits,dim=-1)
            h, target = hidden[pos-1], ids[0,pos]
            lp = checkpoint(project,h,target,use_reentrant=False) if torch.is_grad_enabled() else project(h,target)
            results.append(lp)
        lp = torch.cat(results)
        if return_probe:
            return lp, self.model.lm_head(hidden[positions[-1]-1]).float().detach().cpu()
        return lp


def stats(values):
    x = torch.tensor(values,dtype=torch.float64)
    return {'count':x.numel(),'mean':x.mean().item(),'p99':torch.quantile(x,.99).item(),'max':x.max().item()} if x.numel() else None


def make_model(rank, **fsdp_options):
    if not MODEL:
        raise ValueError('Set OCCAM_STEP0_MODEL to a local step-0 model directory')
    model = AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.float32,local_files_only=True)
    model.model.visual.requires_grad_(False)
    model.gradient_checkpointing_enable({'use_reentrant':False})
    model.config.use_cache = False
    linear_modules = [m for m in model.modules() if hasattr(m,'chunk_gated_delta_rule')]
    kernels = sorted({m.chunk_gated_delta_rule.__module__ for m in linear_modules})
    if not linear_modules or any(not name.startswith('fla.') for name in kernels):
        raise RuntimeError('Training linear attention did not bind FLA: '+repr(kernels))
    kernel_calls, _ = instrument_linear_attention(linear_modules)
    # Disable dropout consistently for repeated pre/post scoring and gradients.
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):
            module.p = 0.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    policy = functools.partial(transformer_auto_wrap_policy,transformer_layer_cls={Qwen3_5DecoderLayer})
    fsdp = FSDP(ActionScorer(model),auto_wrap_policy=policy,ignored_modules=[model.model.visual],
                device_id=rank,use_orig_params=True,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16),
                **fsdp_options)
    fsdp.train()
    return fsdp, trainable, kernels, kernel_calls


def trajectory_loss(lp, prior, advantage, world, trajectory_count, action_count):
    ratio = (lp - prior).exp()
    loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).sum()
    # FSDP averages gradients across ranks; recover global trajectory mean.
    return loss * world / (trajectory_count * action_count)


def indexed(records):
    offsets = collections.Counter()
    for record in records:
        key = record['trajectory_id']
        start = offsets[key]
        count = len(action_offsets(record))
        offsets[key] += count
        yield key, start, count, record
