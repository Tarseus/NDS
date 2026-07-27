"""Brier--Fisher Winner Specialization utilities.

The functions in this module operate only on tournament winners and cached
runner-up labels. They do not differentiate through destroy/repair.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class BFWSCredits:
    """Exact leave-one-out score-function credits for one training batch."""

    reward: torch.Tensor
    specialization: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


def random_derangement(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample a uniform random derangement by rejection."""
    if batch_size < 2:
        raise ValueError("BFWS requires batch_size >= 2 for mismatched states")

    identity = torch.arange(batch_size, device=device)
    while True:
        permutation = torch.randperm(batch_size, device=device)
        if torch.all(permutation != identity):
            return permutation


def bfws_value(
    winners: torch.Tensor, derangement: torch.Tensor, num_codes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BFWS and its same-state and mismatched agreement terms."""
    if winners.ndim != 2 or winners.size(1) != 2:
        raise ValueError("winners must have shape (batch, 2)")
    if derangement.shape != (winners.size(0),):
        raise ValueError("derangement must have shape (batch,)")
    if num_codes < 2:
        raise ValueError("BFWS requires at least two codes")

    first = winners[:, 0]
    second = winners[:, 1]
    same = (first == second).float().mean()
    mismatched = (first == second[derangement]).float().mean()
    scale = num_codes / (num_codes - 1)
    return scale * (same - mismatched), same, mismatched


def exact_functional_credits(
    rewards: torch.Tensor,
    winners: torch.Tensor,
    runners_up: torch.Tensor,
    derangement: torch.Tensor,
) -> BFWSCredits:
    """Compute exact ``functional - leave-one-out functional`` credits.

    ``rewards`` has shape ``(B, 2, K)`` and contains signed relative
    improvements. Only tournament winners receive non-zero credit. Deleting a
    winner changes its label to the cached runner-up; no repair replay is
    required.
    """
    if rewards.ndim != 3 or rewards.size(1) != 2:
        raise ValueError("rewards must have shape (batch, 2, num_codes)")
    batch_size, replicas, num_codes = rewards.shape
    if winners.shape != (batch_size, replicas):
        raise ValueError("winners shape must match reward batch/replica dimensions")
    if runners_up.shape != winners.shape:
        raise ValueError("runners_up shape must match winners")
    if derangement.shape != (batch_size,):
        raise ValueError("derangement must have shape (batch,)")
    if num_codes < 2:
        raise ValueError("BFWS requires at least two codes")

    device = rewards.device
    batch_index = torch.arange(batch_size, device=device)
    replica_index = torch.arange(replicas, device=device)

    reward_credit = torch.zeros_like(rewards)
    winner_reward = rewards[
        batch_index[:, None], replica_index[None, :], winners
    ]
    runner_reward = rewards[
        batch_index[:, None], replica_index[None, :], runners_up
    ]
    reward_margin = (winner_reward - runner_reward) / (batch_size * replicas)
    reward_credit.scatter_(2, winners.unsqueeze(2), reward_margin.unsqueeze(2))

    first = winners[:, 0]
    second = winners[:, 1]
    first_runner = runners_up[:, 0]
    second_runner = runners_up[:, 1]
    same_old = (first == second).float()

    mismatch_old_first = (first == second[derangement]).float()
    same_new_first = (first_runner == second).float()
    mismatch_new_first = (first_runner == second[derangement]).float()
    first_delta = (
        same_old
        - same_new_first
        - mismatch_old_first
        + mismatch_new_first
    )

    inverse = torch.empty_like(derangement)
    inverse[derangement] = batch_index
    mismatch_old_second = (first[inverse] == second).float()
    same_new_second = (first == second_runner).float()
    mismatch_new_second = (first[inverse] == second_runner).float()
    second_delta = (
        same_old
        - same_new_second
        - mismatch_old_second
        + mismatch_new_second
    )

    specialization_credit = torch.zeros_like(rewards)
    bfws_scale = num_codes / ((num_codes - 1) * batch_size)
    specialization_credit[:, 0].scatter_(
        1, first.unsqueeze(1), (bfws_scale * first_delta).unsqueeze(1)
    )
    specialization_credit[:, 1].scatter_(
        1, second.unsqueeze(1), (bfws_scale * second_delta).unsqueeze(1)
    )

    value, same, mismatched = bfws_value(winners, derangement, num_codes)
    winner_frequency = torch.bincount(
        winners.reshape(-1), minlength=num_codes
    ).float()
    winner_frequency = winner_frequency / winner_frequency.sum()
    effective_codes = winner_frequency.square().sum().reciprocal()
    best_reward = rewards.max(dim=2).values

    diagnostics = {
        "j_k": best_reward.mean(),
        "bfws": value,
        "same_agreement": same,
        "mismatched_agreement": mismatched,
        "effective_codes": effective_codes,
        "joint_failure": (best_reward <= 0).float().mean(),
        "winner_margin": (winner_reward - runner_reward).mean(),
        "reward_credit_l1": reward_credit.abs().sum(),
        "bfws_credit_l1": specialization_credit.abs().sum(),
        "winner_frequency": winner_frequency,
    }
    return BFWSCredits(
        reward=reward_credit,
        specialization=specialization_credit,
        diagnostics=diagnostics,
    )
