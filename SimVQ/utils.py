from typing import Optional
import os
import random
import numpy as np
import torch


def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    r"""Function that sets the seed for pseudo-random number generators in: torch, numpy, and Python's random module.
    In addition, sets the following environment variables:

    - ``PL_GLOBAL_SEED``: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - ``PL_SEED_WORKERS``: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If ``None``, it will read the seed from ``PL_GLOBAL_SEED`` env variable. If ``None`` and the
            ``PL_GLOBAL_SEED`` env variable is not set, then the seed defaults to 0.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~lightning.fabric.utilities.seed.pl_worker_init_function`.

    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


def to_device(batch, device):
    """
    Recursively transfers data to specified device, supporting nested structures
    Args:
        batch: Input data (tensor, dict, list, tuple)
        device: Target device
    Returns:
        Same structure as input, but with all tensors moved to specified device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    else:
        return batch