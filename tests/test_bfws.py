import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "src" / "bfws.py"
SPEC = importlib.util.spec_from_file_location("bfws", MODULE_PATH)
bfws = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bfws)


def _functional(rewards, winners, permutation):
    batch_size, _, num_codes = rewards.shape
    first = winners[:, 0]
    second = winners[:, 1]
    j_value = torch.stack(
        [
            rewards[:, 0].gather(1, first[:, None]).squeeze(1),
            rewards[:, 1].gather(1, second[:, None]).squeeze(1),
        ],
        dim=1,
    ).mean()
    scale = num_codes / (num_codes - 1)
    value = scale * (
        (first == second).float().mean()
        - (first == second[permutation]).float().mean()
    )
    return j_value, value


def test_random_derangement_has_no_fixed_points():
    for batch_size in (2, 3, 16):
        permutation = bfws.random_derangement(batch_size, torch.device("cpu"))
        assert sorted(permutation.tolist()) == list(range(batch_size))
        assert torch.all(permutation != torch.arange(batch_size))


def test_constant_generalist_has_zero_bfws():
    winners = torch.zeros(8, 2, dtype=torch.long)
    permutation = torch.roll(torch.arange(8), 1)
    value, same, mismatched = bfws.bfws_value(winners, permutation, 4)
    assert value.item() == pytest.approx(0.0)
    assert same.item() == pytest.approx(1.0)
    assert mismatched.item() == pytest.approx(1.0)


def test_exact_credits_equal_brute_force_functional_differences():
    rewards = torch.tensor(
        [
            [[0.1, 0.4, -0.2], [0.3, 0.2, -0.1]],
            [[0.7, 0.1, 0.2], [-0.2, 0.5, 0.4]],
            [[0.2, -0.1, 0.6], [0.8, 0.1, 0.3]],
        ],
        dtype=torch.float32,
    )
    winners = torch.tensor([[1, 0], [0, 1], [2, 0]])
    runners = torch.tensor([[0, 1], [2, 2], [0, 2]])
    permutation = torch.tensor([1, 2, 0])
    credits = bfws.exact_functional_credits(
        rewards, winners, runners, permutation
    )
    full_j, full_v = _functional(rewards, winners, permutation)

    for batch in range(rewards.size(0)):
        for replica in range(2):
            for code in range(rewards.size(2)):
                if code != winners[batch, replica]:
                    expected_j = 0.0
                    expected_v = 0.0
                else:
                    removed_winners = winners.clone()
                    removed_winners[batch, replica] = runners[batch, replica]
                    removed_j, removed_v = _functional(
                        rewards, removed_winners, permutation
                    )
                    expected_j = (full_j - removed_j).item()
                    expected_v = (full_v - removed_v).item()
                assert credits.reward[batch, replica, code].item() == pytest.approx(
                    expected_j, abs=1e-7
                )
                assert credits.specialization[
                    batch, replica, code
                ].item() == pytest.approx(expected_v, abs=1e-7)


def test_fixed_code_panel_is_repeated_in_replica_order():
    sampler_path = Path(__file__).parents[1] / "src" / "seed_sampler.py"
    spec = importlib.util.spec_from_file_location("seed_sampler", sampler_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sampler = module.SeedVectorSampler(4, torch.device("cpu"))
    indices = sampler.fixed_indices(3, seed=17)
    codes = sampler.repeat_fixed(indices, batch_size=2, replicas=2)
    assert codes.shape == (2, 6, 4)
    assert torch.equal(codes[0, :3], codes[0, 3:])
    assert torch.equal(codes[0], codes[1])
    assert torch.unique(indices).numel() == 3


def test_resampled_panel_is_shared_and_repeated_in_replica_order():
    sampler_path = Path(__file__).parents[1] / "src" / "seed_sampler.py"
    spec = importlib.util.spec_from_file_location("seed_sampler", sampler_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sampler = module.SeedVectorSampler(4, torch.device("cpu"))
    torch.manual_seed(17)
    first = sampler.sample_shared_panel(2, num_codes=3, replicas=2)
    second = sampler.sample_shared_panel(2, num_codes=3, replicas=2)
    assert first.shape == (2, 6, 4)
    assert torch.equal(first[0, :3], first[0, 3:])
    assert torch.equal(first[0], first[1])
    assert torch.unique(first[0, :3], dim=0).size(0) == 3
    assert not torch.equal(first, second)


def test_k3_score_credit_matches_exact_enumerated_gradient():
    batch_size, replicas, num_codes = 2, 2, 3
    shape = (batch_size, replicas, num_codes)
    logits = torch.linspace(-0.8, 0.9, 12, requires_grad=True).reshape(shape)
    base = torch.tensor(
        [
            [[0.11, 0.07, 0.02], [0.04, 0.13, 0.08]],
            [[0.09, 0.03, 0.15], [0.12, 0.06, 0.01]],
        ]
    )
    action_gain = torch.tensor(
        [
            [[0.19, 0.23, 0.17], [0.16, 0.21, 0.25]],
            [[0.22, 0.18, 0.20], [0.24, 0.15, 0.26]],
        ]
    )
    permutation = torch.tensor([1, 0])
    dual = 0.37
    probabilities = logits.sigmoid()
    expected_objective = logits.new_zeros(())
    score_surrogate = logits.new_zeros(())

    for bits in itertools.product((0.0, 1.0), repeat=12):
        actions = torch.tensor(bits).reshape(shape)
        event_probability = torch.where(
            actions.bool(), probabilities, 1.0 - probabilities
        ).prod()
        rewards = base + actions * action_gain
        ranking = rewards.argsort(dim=2, descending=True)
        winners = ranking[:, :, 0]
        runners = ranking[:, :, 1]
        value, _, _ = bfws.bfws_value(winners, permutation, num_codes)
        objective = rewards.max(dim=2).values.mean() + dual * value
        expected_objective = expected_objective + event_probability * objective

        credits = bfws.exact_functional_credits(
            rewards, winners, runners, permutation
        )
        total_credit = credits.reward + dual * credits.specialization
        event_log_probability = torch.where(
            actions.bool(),
            probabilities.log(),
            (1.0 - probabilities).log(),
        )
        score_surrogate = score_surrogate + event_probability.detach() * (
            total_credit.detach() * event_log_probability
        ).sum()

    exact_gradient = torch.autograd.grad(
        expected_objective, logits, retain_graph=True
    )[0]
    score_gradient = torch.autograd.grad(score_surrogate, logits)[0]
    torch.testing.assert_close(score_gradient, exact_gradient, atol=2e-6, rtol=2e-5)
