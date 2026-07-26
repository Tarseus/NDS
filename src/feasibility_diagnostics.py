"""Side-effect-free diagnostics for compact strategy portfolios.

The functions in this module operate only on rewards/costs that have already
been produced by NDS.  They never contribute to the training loss.
"""

import math
from typing import Dict, Iterable

import torch


def _subset_max_weights(
    rollout_size: int, subset_size: int, device: torch.device
) -> torch.Tensor:
    """Probability weights for the maximum of a uniform subset without replacement."""
    if not 1 <= subset_size <= rollout_size:
        raise ValueError(
            f"subset_size must be in [1, {rollout_size}], got {subset_size}"
        )

    denominator = math.comb(rollout_size, subset_size)
    weights = [
        (
            math.comb(rank, subset_size - 1) / denominator
            if rank >= subset_size - 1
            else 0.0
        )
        for rank in range(rollout_size)
    ]
    return torch.tensor(weights, dtype=torch.float64, device=device)


def expected_subset_best(
    rewards: torch.Tensor, subset_size: int
) -> torch.Tensor:
    """Exact expected best reward of a uniform subset of rollout columns.

    Args:
        rewards: Non-negative rewards with shape ``(batch, rollout)``.
        subset_size: Number of rollout columns sampled without replacement.

    Returns:
        Scalar mean over states.
    """
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (batch, rollout)")
    rollout_size = rewards.size(1)
    weights = _subset_max_weights(rollout_size, subset_size, rewards.device)
    sorted_rewards = rewards.detach().to(torch.float64).sort(dim=1).values
    return (sorted_rewards * weights).sum(dim=1).mean()


def compression_curve(
    rewards: torch.Tensor, subset_sizes: Iterable[int]
) -> Dict[str, float]:
    """Compute exact random-subset best-reward gains and retention ratios."""
    rollout_size = rewards.size(1)
    sizes = sorted({int(size) for size in subset_sizes if int(size) <= rollout_size})
    if rollout_size not in sizes:
        sizes.append(rollout_size)

    gains = {
        size: float(expected_subset_best(rewards, size).item()) for size in sizes
    }
    full_gain = gains[rollout_size]
    metrics: Dict[str, float] = {"diag_gain_full": full_gain}
    for size in sizes:
        metrics[f"diag_gain_k{size}"] = gains[size]
        metrics[f"diag_retention_k{size}"] = (
            gains[size] / full_gain if full_gain > 0 else 1.0
        )

    top_two = torch.topk(rewards.detach().float(), k=2, dim=1).values
    positive = top_two[:, 0] > 0
    relative_margin = torch.zeros_like(top_two[:, 0])
    relative_margin[positive] = (
        (top_two[positive, 0] - top_two[positive, 1])
        / top_two[positive, 0].clamp_min(1e-12)
    )
    metrics["diag_positive_best_frac"] = float(positive.float().mean().item())
    metrics["diag_relative_margin"] = float(relative_margin.mean().item())
    return metrics


def _random_tie_argmin(costs: torch.Tensor) -> torch.Tensor:
    """Argmin with uniform random tie-breaking under the caller's RNG context."""
    minimum = costs.min(dim=1, keepdim=True).values
    ties = costs == minimum
    tie_keys = torch.rand(costs.shape, device=costs.device)
    tie_keys = tie_keys.masked_fill(~ties, float("inf"))
    return tie_keys.argmin(dim=1)


def repeated_tournament_metrics(
    costs_replica_1: torch.Tensor,
    costs_replica_2: torch.Tensor,
    incumbent_costs: torch.Tensor,
    prefix_sizes: Iterable[int],
    improvement_eps: float = 1e-8,
) -> Dict[str, float]:
    """Measure reward, repeated winner specialization, margin, and noise.

    Each replica must evaluate the same ordered code panel on the same states.
    Cross-state agreement uses a deterministic cyclic derangement of the batch.
    """
    if costs_replica_1.shape != costs_replica_2.shape:
        raise ValueError("the two cost replicas must have the same shape")
    if costs_replica_1.ndim != 2:
        raise ValueError("cost replicas must have shape (batch, panel)")
    if incumbent_costs.ndim != 1 or incumbent_costs.size(0) != costs_replica_1.size(0):
        raise ValueError("incumbent_costs must have shape (batch,)")
    if costs_replica_1.size(0) < 2:
        raise ValueError("at least two independent states are required")

    panel_size = costs_replica_1.size(1)
    sizes = sorted(
        {int(size) for size in prefix_sizes if 2 <= int(size) <= panel_size}
    )
    metrics: Dict[str, float] = {}
    incumbent = incumbent_costs.float()

    for size in sizes:
        c1 = costs_replica_1[:, :size].float()
        c2 = costs_replica_2[:, :size].float()
        winner1 = _random_tie_argmin(c1)
        winner2 = _random_tie_argmin(c2)

        same = winner1 == winner2
        shuffled = winner1 == winner2.roll(1)
        same_agreement = same.float().mean()
        shuffled_agreement = shuffled.float().mean()
        bws = (size / (size - 1.0)) * (same_agreement - shuffled_agreement)

        top2_1 = torch.topk(c1, k=2, dim=1, largest=False).values
        top2_2 = torch.topk(c2, k=2, dim=1, largest=False).values
        margin1 = top2_1[:, 1] - top2_1[:, 0]
        margin2 = top2_2[:, 1] - top2_2[:, 0]
        improvement1 = (incumbent - top2_1[:, 0]).clamp_min(0)
        improvement2 = (incumbent - top2_2[:, 0]).clamp_min(0)
        useful = (
            same
            & (improvement1 > improvement_eps)
            & (improvement2 > improvement_eps)
        )

        stable_margin = torch.minimum(margin1, margin2)
        stable_margin = torch.where(useful, stable_margin, torch.zeros_like(stable_margin))
        replica_noise = (c1 - c2).abs().median()
        median_margin = torch.cat((margin1, margin2)).median()
        margin_snr = median_margin / replica_noise.clamp_min(1e-12)

        winners = torch.cat((winner1, winner2))
        counts = torch.bincount(winners, minlength=size).float()
        probabilities = counts / counts.sum().clamp_min(1)
        nonzero = probabilities > 0
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        effective_winners = entropy.exp()

        prefix = f"probe_k{size}"
        metrics[f"{prefix}_best_gain"] = float(
            torch.cat((improvement1, improvement2)).mean().item()
        )
        metrics[f"{prefix}_same_agreement"] = float(same_agreement.item())
        metrics[f"{prefix}_shuffled_agreement"] = float(shuffled_agreement.item())
        metrics[f"{prefix}_bws"] = float(bws.item())
        metrics[f"{prefix}_useful_repeat_frac"] = float(useful.float().mean().item())
        metrics[f"{prefix}_stable_margin"] = float(stable_margin.mean().item())
        metrics[f"{prefix}_replica_noise"] = float(replica_noise.item())
        metrics[f"{prefix}_margin_snr"] = float(margin_snr.item())
        metrics[f"{prefix}_effective_winners"] = float(effective_winners.item())

    return metrics
