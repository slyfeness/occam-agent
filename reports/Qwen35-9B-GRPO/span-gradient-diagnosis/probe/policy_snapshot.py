"""Export the just-updated FSDP text policy in the original checkpoint namespace."""
import json
from pathlib import Path


PROBE_PARAMETERS = {
    'lm_head.weight': 'language_model.lm_head.weight',
    'model.language_model.embed_tokens.weight': 'language_model.model.embed_tokens.weight',
    'model.language_model.layers.0.input_layernorm.weight': 'language_model.model.layers.0.input_layernorm.weight',
}


def export_policy_snapshot(model, model_directory, directory, step, rank):
    import torch
    from safetensors.torch import save_file
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

    from .snapshot_fingerprint import probe_fingerprint

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                              FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state = model.state_dict()
    if rank != 0:
        return
    index = json.loads((Path(model_directory)/'model.safetensors.index.json').read_text())['weight_map']
    names = {name for name in index if name == 'lm_head.weight' or name.startswith('model.language_model.')}
    text = {name.removeprefix('model.'): tensor for name, tensor in state.items()
            if name.removeprefix('model.') in names}
    if not names or set(text) != names:
        raise RuntimeError('Exported text parameters do not match the frozen checkpoint namespace')
    weights = {name: value.to(torch.bfloat16).contiguous() for name, value in text.items()}
    fingerprint = probe_fingerprint((alias, weights[name]) for name, alias in PROBE_PARAMETERS.items())
    directory = Path(directory)
    path = directory/f'step-{step}.safetensors'
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_suffix('.partial')
    save_file(weights, str(pending), metadata={'vllm_probe_names': json.dumps(list(PROBE_PARAMETERS.values()))})
    pending.rename(path)
    with path.with_suffix('.json').open('x') as handle:
        json.dump({'snapshot': path.name, 'fingerprint': fingerprint, 'source_tensors': len(weights)}, handle)
