"""Rank-local checkpoint I/O for the fixed FSDP topology of this smoke."""
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardedStateDictConfig, StateDictType


def sharded_state(model):
    # Named parameter shards support use_orig_params=True on both save and load.
    return FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                ShardedStateDictConfig(offload_to_cpu=True))


def save_rank(model, optimizer, path, rank, world, step):
    temporary = path.with_suffix(".partial")
    with sharded_state(model):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                    "step": step, "rank": rank, "world_size": world}, temporary)
    temporary.rename(path)


def load_rank(model, optimizer, path, rank, world, step):
    state = torch.load(path, weights_only=False, map_location="cpu")
    if (state["rank"], state["world_size"], state["step"]) != (rank, world, step):
        raise RuntimeError("Checkpoint rank, topology or step mismatch")
    with sharded_state(model):
        model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
