"""Utilities for the same-size cross-distribution latent-code pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch


CVRP100_DISTRIBUTIONS: Dict[str, dict] = {
    "uniform": {
        "use_X_generator": False,
        "rootPos": 2,
        "custPos": 1,
        "demandType": 2,
        "avgRouteSize": 3,
    },
    "x_uniform_center": {
        "use_X_generator": True,
        "rootPos": 2,
        "custPos": 1,
        "demandType": 2,
        "avgRouteSize": 3,
    },
    "x_cluster_center": {
        "use_X_generator": True,
        "rootPos": 2,
        "custPos": 2,
        "demandType": 2,
        "avgRouteSize": 3,
    },
    "x_mixed_center": {
        "use_X_generator": True,
        "rootPos": 2,
        "custPos": 3,
        "demandType": 2,
        "avgRouteSize": 3,
    },
    "x_cluster_corner_quad": {
        "use_X_generator": True,
        "rootPos": 3,
        "custPos": 2,
        "demandType": 6,
        "avgRouteSize": 3,
    },
    "x_mixed_random_fewlarge": {
        "use_X_generator": True,
        "rootPos": 1,
        "custPos": 3,
        "demandType": 7,
        "avgRouteSize": 2,
    },
}


FEATURE_NAMES = (
    "depot_x",
    "depot_y",
    "customer_x_mean",
    "customer_y_mean",
    "customer_x_std",
    "customer_y_std",
    "customer_x_span",
    "customer_y_span",
    "radius_mean",
    "radius_std",
    "radius_q25",
    "radius_q75",
    "nearest_neighbor_mean",
    "nearest_neighbor_std",
    "demand_mean",
    "demand_std",
    "demand_q25",
    "demand_q75",
    "demand_max",
    "capacity",
    "demand_capacity_ratio",
)


@dataclass(frozen=True)
class BootstrapInterval:
    mean: float
    low: float
    high: float


def fixed_code_indices(pool_size: int, num_codes: int, seed: int) -> np.ndarray:
    """Return a deterministic panel of distinct code indices."""
    if not 1 <= num_codes <= pool_size:
        raise ValueError("num_codes must lie in [1, pool_size]")
    return np.random.default_rng(seed).permutation(pool_size)[:num_codes]


def build_code_major_panel(
    code_pool: torch.Tensor,
    code_indices: Sequence[int] | np.ndarray | torch.Tensor,
    batch_size: int,
    replicas: int,
) -> torch.Tensor:
    """Create ``[code0 x R, code1 x R, ...]`` for every instance."""
    if batch_size < 1 or replicas < 1:
        raise ValueError("batch_size and replicas must be positive")
    indices = torch.as_tensor(
        code_indices, dtype=torch.long, device=code_pool.device
    )
    panel = code_pool[indices].repeat_interleave(replicas, dim=0)
    return panel.unsqueeze(0).expand(batch_size, -1, -1)


def reshape_code_replicas(
    values: np.ndarray, num_codes: int, replicas: int
) -> np.ndarray:
    """Reshape a batch × rollout array using the code-major panel layout."""
    values = np.asarray(values)
    expected = num_codes * replicas
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(
            f"expected shape (batch, {expected}), received {values.shape}"
        )
    return values.reshape(values.shape[0], num_codes, replicas)


def extract_instance_features(
    depot_xy: torch.Tensor,
    node_xy: torch.Tensor,
    node_demand: torch.Tensor,
    capacity: torch.Tensor,
) -> np.ndarray:
    """Compute low-cost, permutation-invariant CVRP instance features."""
    depot_xy = depot_xy.detach().float().cpu()
    node_xy = node_xy.detach().float().cpu()
    node_demand = node_demand.detach().float().cpu()
    capacity = capacity.detach().float().cpu().reshape(-1)

    if depot_xy.ndim != 3 or depot_xy.shape[1:] != (1, 2):
        raise ValueError("depot_xy must have shape (batch, 1, 2)")
    if node_xy.ndim != 3 or node_xy.shape[2] != 2:
        raise ValueError("node_xy must have shape (batch, nodes, 2)")
    if node_demand.shape != node_xy.shape[:2]:
        raise ValueError("node_demand shape must match node_xy[:2]")
    if capacity.shape[0] != node_xy.shape[0]:
        raise ValueError("capacity batch dimension must match node_xy")

    coordinate_mean = node_xy.mean(dim=1)
    coordinate_std = node_xy.std(dim=1, unbiased=False)
    coordinate_span = node_xy.amax(dim=1) - node_xy.amin(dim=1)

    radius = torch.linalg.vector_norm(node_xy - depot_xy, dim=-1)
    radius_stats = torch.stack(
        (
            radius.mean(dim=1),
            radius.std(dim=1, unbiased=False),
            torch.quantile(radius, 0.25, dim=1),
            torch.quantile(radius, 0.75, dim=1),
        ),
        dim=1,
    )

    pair_distance = torch.cdist(node_xy, node_xy)
    diagonal = torch.eye(
        node_xy.shape[1], dtype=torch.bool
    ).unsqueeze(0)
    pair_distance = pair_distance.masked_fill(diagonal, float("inf"))
    nearest = pair_distance.amin(dim=2)
    nearest_stats = torch.stack(
        (
            nearest.mean(dim=1),
            nearest.std(dim=1, unbiased=False),
        ),
        dim=1,
    )

    demand_stats = torch.stack(
        (
            node_demand.mean(dim=1),
            node_demand.std(dim=1, unbiased=False),
            torch.quantile(node_demand, 0.25, dim=1),
            torch.quantile(node_demand, 0.75, dim=1),
            node_demand.amax(dim=1),
        ),
        dim=1,
    )
    demand_capacity_ratio = node_demand.sum(dim=1) / capacity.clamp_min(1)

    features = torch.cat(
        (
            depot_xy[:, 0],
            coordinate_mean,
            coordinate_std,
            coordinate_span,
            radius_stats,
            nearest_stats,
            demand_stats,
            capacity[:, None],
            demand_capacity_ratio[:, None],
        ),
        dim=1,
    )
    if features.shape[1] != len(FEATURE_NAMES):
        raise AssertionError("feature names and feature tensor disagree")
    return features.numpy()


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    begin = 0
    while begin < values.size:
        end = begin + 1
        while end < values.size and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end - 1)
        begin = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Spearman correlation and return zero for constant inputs."""
    left_rank = _rankdata(np.asarray(left))
    right_rank = _rankdata(np.asarray(right))
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_rank, right_rank) / denominator)


def split_half_reliability(utilities: np.ndarray) -> np.ndarray:
    """Per-instance code-rank correlation across two replica halves."""
    utilities = np.asarray(utilities)
    if utilities.ndim != 3 or utilities.shape[2] < 2:
        raise ValueError("utilities must have shape (instances, codes, replicas>=2)")
    midpoint = utilities.shape[2] // 2
    if midpoint == 0 or midpoint == utilities.shape[2]:
        raise ValueError("replicas cannot be divided into two nonempty halves")
    left = utilities[:, :, :midpoint].mean(axis=2)
    right = utilities[:, :, midpoint:].mean(axis=2)
    return np.asarray(
        [spearman_correlation(a, b) for a, b in zip(left, right)],
        dtype=np.float64,
    )


def bootstrap_mean_interval(
    values: np.ndarray,
    seed: int,
    samples: int = 5000,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Nonparametric bootstrap interval for a mean."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty array")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(samples, values.size))
    means = values[draws].mean(axis=1)
    alpha = 0.5 * (1.0 - confidence)
    return BootstrapInterval(
        mean=float(values.mean()),
        low=float(np.quantile(means, alpha)),
        high=float(np.quantile(means, 1.0 - alpha)),
    )


def choose_global_code(calibration_utility: np.ndarray) -> int:
    """Choose the code with maximum mean calibration utility."""
    calibration_utility = np.asarray(calibration_utility)
    if calibration_utility.ndim != 2:
        raise ValueError("calibration_utility must have shape (instances, codes)")
    return int(calibration_utility.mean(axis=0).argmax())


def choose_multisource_mean_code(
    utility: np.ndarray,
    held_out_distribution: int,
    calibration_instances: int,
) -> int:
    """Choose a code using equal-weight means from all non-target sources."""
    utility = np.asarray(utility, dtype=np.float64)
    if utility.ndim != 3:
        raise ValueError("utility must have shape (distributions, instances, codes)")
    distributions, instances, _ = utility.shape
    if not 0 <= held_out_distribution < distributions:
        raise ValueError("held_out_distribution is outside the utility tensor")
    if distributions < 2:
        raise ValueError("at least two distributions are required")
    if not 1 <= calibration_instances <= instances:
        raise ValueError("calibration_instances must lie within each distribution")
    source_mask = np.arange(distributions) != held_out_distribution
    source_means = utility[source_mask, :calibration_instances].mean(axis=1)
    return int(source_means.mean(axis=0).argmax())


def multisource_knn_codes(
    source_features: np.ndarray,
    source_utility: np.ndarray,
    target_features: np.ndarray,
    neighbours: int = 5,
) -> np.ndarray:
    """Select a code from the mean utility of feature-nearest source rows."""
    source_features = np.asarray(source_features, dtype=np.float64)
    source_utility = np.asarray(source_utility, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("features must be matrices")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("source and target features must share a dimension")
    if source_utility.ndim != 2:
        raise ValueError("source_utility must be a matrix")
    if source_utility.shape[0] != source_features.shape[0]:
        raise ValueError("source utilities and features must align")
    if not 1 <= neighbours <= source_features.shape[0]:
        raise ValueError("neighbours must lie within the source memory size")

    mean = source_features.mean(axis=0)
    scale = source_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    source_scaled = (source_features - mean) / scale
    target_scaled = (target_features - mean) / scale
    squared_distance = (
        (target_scaled[:, None, :] - source_scaled[None, :, :]) ** 2
    ).sum(axis=2)
    nearest = np.argpartition(
        squared_distance, neighbours - 1, axis=1
    )[:, :neighbours]
    neighbour_utility = source_utility[nearest].mean(axis=1)
    return neighbour_utility.argmax(axis=1)


def nearest_source_codes(
    source_features: np.ndarray,
    source_utility: np.ndarray,
    target_features: np.ndarray,
) -> np.ndarray:
    """Retrieve the nearest source instance and reuse its best code."""
    source_features = np.asarray(source_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    source_utility = np.asarray(source_utility, dtype=np.float64)
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("features must be matrices")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("source and target features must share a dimension")
    if source_utility.shape[0] != source_features.shape[0]:
        raise ValueError("source utilities and features must align")

    mean = source_features.mean(axis=0)
    scale = source_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    source_scaled = (source_features - mean) / scale
    target_scaled = (target_features - mean) / scale
    squared_distance = (
        (target_scaled[:, None, :] - source_scaled[None, :, :]) ** 2
    ).sum(axis=2)
    neighbour = squared_distance.argmin(axis=1)
    source_best = source_utility.argmax(axis=1)
    return source_best[neighbour]


def selected_utility(
    utility: np.ndarray, code_indices: Iterable[int] | np.ndarray
) -> np.ndarray:
    """Gather one selected code utility per instance."""
    utility = np.asarray(utility)
    code_indices = np.asarray(code_indices, dtype=np.int64)
    if utility.ndim != 2 or code_indices.shape != (utility.shape[0],):
        raise ValueError("one code index is required per utility row")
    return utility[np.arange(utility.shape[0]), code_indices]


def distribution_slices(
    names: Sequence[str], instances_per_distribution: int
) -> Mapping[str, slice]:
    """Map concatenated distribution names to contiguous slices."""
    return {
        name: slice(
            index * instances_per_distribution,
            (index + 1) * instances_per_distribution,
        )
        for index, name in enumerate(names)
    }
