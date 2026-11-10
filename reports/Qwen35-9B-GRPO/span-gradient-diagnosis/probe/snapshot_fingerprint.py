"""Deterministic fingerprint for the resident policy probe parameters."""
import hashlib


def probe_fingerprint(samples):
    import torch

    digest = hashlib.sha256()
    for name, parameter in samples:
        tensor = parameter.detach().reshape(-1)
        if tensor.numel() < 1024:
            raise ValueError(f'Unexpectedly small policy probe parameter: {name}')
        # Probe both ends and the midpoint without transferring a full tensor to host.
        digest.update(name.encode())
        for start in (0, tensor.numel()//2, tensor.numel()-256):
            digest.update(tensor[start:start+256].cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
