"""Helpers for transporting a distribution over NDS binary latent codes."""

from __future__ import annotations

import numpy as np


def binary_code_features(code_pool: np.ndarray) -> np.ndarray:
    """Return intercept plus centered binary coordinates for every code."""
    pool = np.asarray(code_pool, dtype=np.float64)
    if pool.ndim != 2 or not np.isin(pool, [0.0, 1.0]).all():
        raise ValueError("code_pool must be a binary matrix")
    return np.concatenate(
        (np.ones((pool.shape[0], 1)), 2.0 * pool - 1.0), axis=1
    )


def fit_transport_logits(
    code_pool: np.ndarray,
    observed_indices: np.ndarray,
    observed_utility: np.ndarray,
    ridge: float = 1.0,
) -> np.ndarray:
    """Fit an additive ridge model on observed codes and score the full pool."""
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    features = binary_code_features(code_pool)
    indices = np.asarray(observed_indices, dtype=np.int64).reshape(-1)
    utility = np.asarray(observed_utility, dtype=np.float64).reshape(-1)
    if indices.shape != utility.shape or indices.size < 2:
        raise ValueError("observed indices and utility must be aligned")
    if indices.min() < 0 or indices.max() >= features.shape[0]:
        raise ValueError("observed code index is outside the pool")
    design = features[indices]
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty, design.T @ utility
    )
    logits = features @ coefficients
    scale = logits.std()
    if scale < 1e-12:
        return np.zeros_like(logits)
    return np.clip((logits - logits.mean()) / scale, -4.0, 4.0)


def mixed_transport_probabilities(
    logits: np.ndarray,
    exploration: float = 0.5,
    strength: float = 1.0,
) -> np.ndarray:
    """Convert prior logits to a safe mixture with uniform exploration."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if logits.size < 2 or not np.isfinite(logits).all():
        raise ValueError("logits must be a finite vector with at least two entries")
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("exploration must lie in [0, 1]")
    if strength < 0:
        raise ValueError("strength must be non-negative")
    scaled = strength * logits
    scaled -= scaled.max()
    learned = np.exp(scaled)
    learned /= learned.sum()
    uniform = np.full(logits.size, 1.0 / logits.size)
    return (1.0 - exploration) * learned + exploration * uniform


def sample_code_indices(
    probabilities: np.ndarray,
    batch_size: int,
    rounds: int,
    rollout_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample a distinct code set for every instance and every round."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if batch_size < 1 or rounds < 1 or rollout_size < 1:
        raise ValueError("batch_size, rounds, and rollout_size must be positive")
    if rollout_size > probabilities.size:
        raise ValueError("rollout_size cannot exceed the code pool")
    probabilities = probabilities / probabilities.sum()
    result = np.empty((batch_size, rounds, rollout_size), dtype=np.int64)
    for batch_index in range(batch_size):
        for round_index in range(rounds):
            result[batch_index, round_index] = rng.choice(
                probabilities.size,
                size=rollout_size,
                replace=False,
                p=probabilities,
            )
    return result


def paired_best_improvement(
    candidate_costs: np.ndarray,
    incumbent_costs: np.ndarray,
) -> np.ndarray:
    """Return normalized best improvement along the final candidate axis."""
    candidate_costs = np.asarray(candidate_costs, dtype=np.float64)
    incumbent_costs = np.asarray(incumbent_costs, dtype=np.float64).reshape(-1)
    if candidate_costs.ndim < 2 or candidate_costs.shape[0] != incumbent_costs.size:
        raise ValueError("candidate and incumbent batch axes must align")
    best = candidate_costs.min(axis=-1)
    shape = (incumbent_costs.size,) + (1,) * (best.ndim - 1)
    base = incumbent_costs.reshape(shape)
    return np.maximum(base - best, 0.0) / np.maximum(np.abs(base), 1e-12)
