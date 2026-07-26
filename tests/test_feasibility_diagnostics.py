import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "src" / "feasibility_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("feasibility_diagnostics", MODULE_PATH)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def test_expected_subset_best_matches_bruteforce():
    rewards = torch.tensor([[0.0, 1.0, 2.0, 4.0]])
    expected_pairs = torch.tensor([1.0, 2.0, 4.0, 2.0, 4.0, 4.0]).mean()
    actual = diagnostics.expected_subset_best(rewards, 2)
    assert actual.item() == pytest.approx(expected_pairs.item())


def test_compression_curve_is_monotone_and_full_retention_is_one():
    rewards = torch.tensor([[0.0, 1.0, 3.0, 2.0], [1.0, 1.0, 1.0, 1.0]])
    metrics = diagnostics.compression_curve(rewards, [1, 2])
    assert metrics["diag_gain_k1"] <= metrics["diag_gain_k2"]
    assert metrics["diag_gain_k2"] <= metrics["diag_gain_full"]
    assert metrics["diag_retention_k4"] == pytest.approx(1.0)


def test_repeated_specialists_have_positive_bws_and_multiple_winners():
    costs1 = torch.tensor(
        [
            [1.0, 4.0, 4.0, 4.0],
            [4.0, 1.0, 4.0, 4.0],
            [4.0, 4.0, 1.0, 4.0],
            [4.0, 4.0, 4.0, 1.0],
        ]
    )
    costs2 = costs1.clone()
    incumbent = torch.full((4,), 5.0)
    metrics = diagnostics.repeated_tournament_metrics(
        costs1, costs2, incumbent, [4]
    )
    assert metrics["probe_k4_same_agreement"] == pytest.approx(1.0)
    assert metrics["probe_k4_shuffled_agreement"] == pytest.approx(0.0)
    assert metrics["probe_k4_bws"] == pytest.approx(4.0 / 3.0)
    assert metrics["probe_k4_useful_repeat_frac"] == pytest.approx(1.0)
    assert metrics["probe_k4_effective_winners"] == pytest.approx(4.0)


def test_constant_generalist_has_zero_bws():
    costs1 = torch.tensor([[1.0, 2.0, 3.0]]).repeat(8, 1)
    costs2 = costs1.clone()
    incumbent = torch.full((8,), 4.0)
    metrics = diagnostics.repeated_tournament_metrics(
        costs1, costs2, incumbent, [3]
    )
    assert metrics["probe_k3_same_agreement"] == pytest.approx(1.0)
    assert metrics["probe_k3_shuffled_agreement"] == pytest.approx(1.0)
    assert metrics["probe_k3_bws"] == pytest.approx(0.0)
    assert metrics["probe_k3_effective_winners"] == pytest.approx(1.0)
