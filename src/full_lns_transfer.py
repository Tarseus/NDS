"""Pure helpers for the fixed-budget full-LNS transfer experiment."""

from __future__ import annotations

import numpy as np


def geometric_temperatures(
    iterations: int, start: float, final: float
) -> np.ndarray:
    """Return an inclusive geometric temperature schedule."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if start <= 0 or final <= 0:
        raise ValueError("temperatures must be positive")
    if iterations == 1:
        return np.asarray([start], dtype=np.float64)
    return np.geomspace(start, final, num=iterations, dtype=np.float64)


def random_fixed_assignments(
    code_indices: np.ndarray,
    distributions: int,
    instances: int,
    seed: int,
) -> np.ndarray:
    """Assign one reproducible candidate code to every target instance."""
    code_indices = np.asarray(code_indices, dtype=np.int64)
    if code_indices.ndim != 1 or code_indices.size == 0:
        raise ValueError("code_indices must be a nonempty vector")
    if distributions < 1 or instances < 1:
        raise ValueError("distributions and instances must be positive")
    rng = np.random.default_rng(seed)
    return rng.choice(
        code_indices,
        size=(distributions, instances),
        replace=True,
    )


def normalized_improvement_curve(
    checkpoint_costs: np.ndarray, initial_cost: np.ndarray
) -> np.ndarray:
    """Convert best-so-far costs into fractional improvement curves."""
    checkpoint_costs = np.asarray(checkpoint_costs, dtype=np.float64)
    initial_cost = np.asarray(initial_cost, dtype=np.float64)
    if checkpoint_costs.ndim != 2:
        raise ValueError("checkpoint_costs must have shape (instances, checkpoints)")
    if initial_cost.shape != (checkpoint_costs.shape[0],):
        raise ValueError("initial_cost must align with checkpoint_costs")
    return (initial_cost[:, None] - checkpoint_costs) / np.maximum(
        np.abs(initial_cost[:, None]), 1e-12
    )


def normalized_anytime_auc(
    checkpoint_costs: np.ndarray,
    checkpoints: np.ndarray,
    initial_cost: np.ndarray,
) -> np.ndarray:
    """Area under normalized improvement over a log-scaled iteration axis."""
    checkpoints = np.asarray(checkpoints, dtype=np.float64)
    if checkpoints.ndim != 1 or checkpoints.size < 2:
        raise ValueError("checkpoints must contain at least two values")
    if np.any(np.diff(checkpoints) <= 0) or checkpoints[0] < 0:
        raise ValueError("checkpoints must be strictly increasing and nonnegative")
    curve = normalized_improvement_curve(checkpoint_costs, initial_cost)
    if curve.shape[1] != checkpoints.size:
        raise ValueError("checkpoint axis and costs disagree")
    axis = np.log1p(checkpoints)
    axis /= axis[-1]
    return np.trapezoid(curve, axis, axis=1)
