"""Export the just-updated FSDP text policy in the original checkpoint namespace."""
import json
from pathlib import Path


PROBE_PARAMETERS = {
    'lm_head.weight': 'language_model.lm_head.weight',
    'model.language_model.embed_tokens.weight': 'language_model.model.embed_tokens.weight',
    'model.language_model.layers.0.input_layernorm.weight': 'language_model.model.layers.0.input_layernorm.weight',
}


def export_policy_snapshot(model, model_directory, directory, step, rank, *, verify_disk=False):
    import torch
    from safetensors.torch import save_file
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

    from .resident_rollout import probe_fingerprint

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
    metadata = {'vllm_probe_names': json.dumps(list(PROBE_PARAMETERS.values()))}
    attempts = []
    disk_sha = None
    if verify_disk:
        from .snapshot_archive import verify_archive, write_archive

        expected = {name: memoryview(tensor.view(torch.uint8).numpy()).cast('B')
                    for name, tensor in weights.items()}
        shapes = {name: list(tensor.shape) for name, tensor in weights.items()}
        for attempt in (1, 2):
            pending = path.with_suffix(f'.attempt-{attempt}.partial')
            if pending.exists():
                raise FileExistsError(pending)
            try:
                write_archive(pending, expected, shapes, metadata)
                disk_sha = verify_archive(pending, expected)
            except (OSError, ValueError) as error:
                attempts.append({'attempt': attempt, 'error': type(error).__name__, 'detail': str(error)})
                path.with_suffix('.export-errors.json').write_text(json.dumps(attempts))
                if attempt == 2:
                    raise
                continue
            pending.rename(path)
            break
    else:
        pending = path.with_suffix('.partial')
        save_file(weights, str(pending), metadata=metadata)
        pending.rename(path)
    with path.with_suffix('.json').open('x') as handle:
        receipt = {'snapshot': path.name, 'fingerprint': fingerprint, 'source_tensors': len(weights)}
        if verify_disk:
            receipt.update(file_sha256=disk_sha, verified_byte_count=path.stat().st_size,
                           retries=len(attempts))
        json.dump(receipt, handle)
