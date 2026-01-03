"""
Reproducibility utilities for seeding random number generators.

Seeds Python random, NumPy, PyTorch CPU/GPU, and sets cuDNN determinism flags.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42) -> int:
    """
    Seed all random number generators for reproducibility.
    
    Seeds:
    - Python's random module
    - NumPy
    - PyTorch CPU and GPU (all devices)
    - Sets cuDNN determinism flags
    
    Args:
        seed: The seed value to use. Default is 42.
    
    Returns:
        The seed that was used.
    
    Note:
        Setting torch.backends.cudnn.deterministic = True can reduce
        training speed by 10-20% on some operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # cuDNN determinism - trades some speed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker initialization function for reproducibility.
    
    Each DataLoader worker has its own random state. This function ensures
    workers are seeded based on the initial torch seed + worker_id.
    
    Usage:
        DataLoader(..., worker_init_fn=worker_init_fn)
    
    Args:
        worker_id: The worker ID (0 to num_workers-1).
    """
    # Use the initial seed from PyTorch plus the worker ID
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Create a PyTorch Generator with an optional seed.
    
    Useful for reproducible shuffling in DataLoader.
    
    Args:
        seed: Optional seed for the generator.
    
    Returns:
        A torch.Generator instance.
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return g


__all__ = ["seed_everything", "worker_init_fn", "get_generator"]

