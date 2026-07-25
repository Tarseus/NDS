"""Quality-weighted Fisher-volume objective for diverse NDS rollouts."""

from typing import Dict, Tuple

import torch


def sample_rollout_subset(
    batch_size: int,
    rollout_size: int,
    subset_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample an independent, uniform subset of rollout indices per instance."""
    if not 1 <= subset_size <= rollout_size:
        raise ValueError(
            f"subset_size must be in [1, {rollout_size}], got {subset_size}"
        )
    random_scores = torch.rand(batch_size, rollout_size, device=device)
    return random_scores.topk(subset_size, dim=1, largest=False).indices


def gather_rollouts(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather the rollout dimension (dimension 1) using per-instance indices."""
    if values.ndim < 2 or indices.ndim != 2:
        raise ValueError("values and indices must have rollout and batch dimensions")
    index = indices
    for _ in range(values.ndim - 2):
        index = index.unsqueeze(-1)
    index = index.expand(*indices.shape, *values.shape[2:])
    return values.gather(1, index)


def route_sqrt_features(
    customer_probs: torch.Tensor,
    tour_index: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate customer probabilities into current-route categories.

    Visited customers share the category of their route. Each unvisited customer
    receives its own category. The latter is required for PCVRP, where selecting
    an unvisited customer can be meaningful and route-only aggregation would lose
    probability mass.
    """
    if customer_probs.ndim != 3:
        raise ValueError("customer_probs must have shape (batch, subset, customers)")
    if tour_index.ndim != 2:
        raise ValueError("tour_index must have shape (batch, customers)")

    batch_size, subset_size, num_customers = customer_probs.shape
    if tour_index.shape != (batch_size, num_customers):
        raise ValueError(
            "tour_index shape must match customer_probs batch/customer dimensions"
        )
    visited = tour_index >= 0
    num_tours = tour_index.clamp_min(-1).amax(dim=1) + 1
    unvisited_rank = (~visited).long().cumsum(dim=1) - 1
    category_index = torch.where(
        visited,
        tour_index,
        num_tours[:, None] + unvisited_rank,
    )

    route_probs = customer_probs.new_zeros(
        batch_size, subset_size, num_customers
    )
    route_probs.scatter_add_(
        2,
        category_index[:, None, :].expand(-1, subset_size, -1),
        customer_probs,
    )

    # Padding categories are exactly zero. Clamp before sqrt to avoid an infinite
    # derivative at zero, then renormalize to keep every feature on the unit sphere.
    features = route_probs.clamp_min(eps).sqrt()
    features = features / features.norm(dim=2, keepdim=True).clamp_min(eps)
    return features, route_probs


def fisher_volume_loss(
    features: torch.Tensor,
    quality: torch.Tensor,
    trajectory_log_prob: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return the unbiased Fisher-volume surrogate and detached diagnostics.

    For J = E[D], the stochastic-computation-graph gradient is

        grad D + sum_i (D - D_-i) grad log pi(tau_i).

    Both terms are divided by the same subset size here. Dividing only the
    score-function term would change their relative weight and would no longer
    be the gradient of the stated objective.
    """
    if features.ndim != 3:
        raise ValueError("features must have shape (batch, subset, categories)")
    if quality.shape != features.shape[:2]:
        raise ValueError("quality shape must match feature batch/subset dimensions")
    if trajectory_log_prob.shape != quality.shape:
        raise ValueError("trajectory_log_prob shape must match quality")

    batch_size, subset_size, _ = features.shape
    quality = quality.detach().clamp_min(0.0)
    weighted_features = quality.sqrt().unsqueeze(2) * features

    # Cholesky in FP32 is deliberately outside autocast for numerical stability.
    weighted_features_fp32 = weighted_features.float()
    gram = weighted_features_fp32 @ weighted_features_fp32.transpose(1, 2)
    identity = torch.eye(
        subset_size, dtype=torch.float32, device=features.device
    ).expand(batch_size, -1, -1)
    kernel = identity + gram
    cholesky = torch.linalg.cholesky(kernel)

    logdet = 2.0 * torch.log(
        torch.diagonal(cholesky, dim1=1, dim2=2)
    ).sum(dim=1)
    inverse_diagonal = torch.diagonal(
        torch.cholesky_inverse(cholesky), dim1=1, dim2=2
    )
    marginal = -torch.log(inverse_diagonal.clamp_min(eps))

    score_term = (
        marginal.detach() * trajectory_log_prob.float()
    ).sum(dim=1)
    loss = -((logdet + score_term) / subset_size).mean()

    similarity = features.float() @ features.float().transpose(1, 2)
    if subset_size > 1:
        off_diagonal = (
            similarity.sum(dim=(1, 2))
            - torch.diagonal(similarity, dim1=1, dim2=2).sum(dim=1)
        ) / (subset_size * (subset_size - 1))
    else:
        off_diagonal = similarity.new_zeros(batch_size)

    diagnostics = {
        "logdet": logdet.detach().mean(),
        "marginal": marginal.detach().mean(),
        "similarity": off_diagonal.detach().mean(),
        "positive_quality_frac": (quality > 0).float().mean(),
    }
    return loss, diagnostics
