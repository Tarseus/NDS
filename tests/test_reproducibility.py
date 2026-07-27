import random
from pathlib import Path

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


def test_seed_everything_can_enable_deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        seed_everything(1234, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous)


def test_cvrp_random_device_is_routed_through_seedable_generator():
    cpp_dir = Path(__file__).parents[1] / "src" / "cpp" / "cvrp"
    operations = (cpp_dir / "Operations.cpp").read_text()
    utilities = (cpp_dir / "Utils.cpp").read_text()

    assert "random_device" not in operations
    assert "void setRandomSeed(unsigned int seed)" in utilities
    assert "randomGenerator.seed(seed)" in utilities
