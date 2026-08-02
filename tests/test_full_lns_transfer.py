import numpy as np

from src.full_lns_transfer import (
    geometric_temperatures,
    normalized_anytime_auc,
    normalized_improvement_curve,
    random_fixed_assignments,
)


def test_geometric_temperature_schedule_is_inclusive():
    schedule = geometric_temperatures(3, 1.0, 0.01)
    np.testing.assert_allclose(schedule, [1.0, 0.1, 0.01])


def test_random_fixed_assignments_are_reproducible_members():
    codes = np.asarray([3, 7, 11])
    first = random_fixed_assignments(codes, 2, 5, 19)
    second = random_fixed_assignments(codes, 2, 5, 19)
    np.testing.assert_array_equal(first, second)
    assert np.isin(first, codes).all()


def test_normalized_curve_and_auc_reward_early_improvement():
    initial = np.asarray([100.0, 100.0])
    checkpoints = np.asarray([0, 1, 10])
    costs = np.asarray([[100.0, 90.0, 80.0], [100.0, 100.0, 80.0]])
    curve = normalized_improvement_curve(costs, initial)
    np.testing.assert_allclose(curve[:, -1], [0.2, 0.2])
    auc = normalized_anytime_auc(costs, checkpoints, initial)
    assert auc[0] > auc[1]
