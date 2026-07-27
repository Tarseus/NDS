"""Reproducibility helpers shared by training and validation."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Reset Python, NumPy, CPU Torch, and CUDA Torch RNGs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
