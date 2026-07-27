import random

import numpy as np
import torch

from src.reproducibility import seed_everything


def test_seed_everything_replays_all_cpu_rngs():
    seed_everything(1234)
    first = (random.random(), np.random.rand(), torch.rand(4))

    seed_everything(1234)
    second = (random.random(), np.random.rand(), torch.rand(4))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
