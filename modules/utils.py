"""Small helpers shared by every module: seeding and device selection."""

from __future__ import annotations

import os
import random

import numpy as np

from modules.config import SEED


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer_gpu: bool = True):
    """CUDA on Colab, Apple MPS locally, CPU otherwise."""
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
