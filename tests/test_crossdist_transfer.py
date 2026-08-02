import numpy as np
import torch

from src.crossdist_transfer import (
    FEATURE_NAMES,
    build_code_major_panel,
    choose_multisource_mean_code,
    extract_instance_features,
    nearest_source_codes,
    multisource_knn_codes,
    reshape_code_replicas,
    spearman_correlation,
    split_half_reliability,
)


def test_code_major_panel_and_reshape_agree():
    pool = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    panel = build_code_major_panel(pool, [2, 0], batch_size=2, replicas=3)
    expected = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    torch.testing.assert_close(panel[0], expected)
    values = np.arange(12).reshape(2, 6)
    reshaped = reshape_code_replicas(values, num_codes=2, replicas=3)
    np.testing.assert_array_equal(reshaped[0, 0], [0, 1, 2])
    np.testing.assert_array_equal(reshaped[0, 1], [3, 4, 5])


def test_instance_features_are_finite_and_named():
    depot = torch.tensor([[[0.5, 0.5]], [[0.0, 0.0]]])
    nodes = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0], [0.5, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        ]
    )
    demand = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    capacity = torch.tensor([[10.0], [20.0]])
    features = extract_instance_features(depot, nodes, demand, capacity)
    assert features.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(features).all()


def test_spearman_and_split_half_reliability():
    assert np.isclose(spearman_correlation([1, 2, 3], [3, 2, 1]), -1.0)
    utilities = np.asarray(
        [
            [
                [1.0, 1.1, 0.9, 1.0],
                [2.0, 2.1, 1.9, 2.0],
                [3.0, 3.1, 2.9, 3.0],
            ]
        ]
    )
    np.testing.assert_allclose(split_half_reliability(utilities), [1.0])


def test_nearest_source_reuses_source_best_code():
    source_features = np.asarray([[0.0], [10.0]])
    source_utility = np.asarray([[4.0, 1.0], [1.0, 5.0]])
    target_features = np.asarray([[0.1], [9.9]])
    np.testing.assert_array_equal(
        nearest_source_codes(source_features, source_utility, target_features),
        [0, 1],
    )


def test_multisource_mean_code_excludes_target_distribution():
    utility = np.asarray(
        [
            [[9.0, 0.0], [9.0, 0.0]],
            [[0.0, 4.0], [0.0, 4.0]],
            [[0.0, 6.0], [0.0, 6.0]],
        ]
    )
    assert choose_multisource_mean_code(utility, 0, 2) == 1
    assert choose_multisource_mean_code(utility, 2, 2) == 0


def test_multisource_knn_averages_neighbour_utilities():
    source_features = np.asarray([[0.0], [0.2], [10.0]])
    source_utility = np.asarray([[5.0, 0.0], [0.0, 9.0], [8.0, 0.0]])
    target_features = np.asarray([[0.1], [9.9]])
    np.testing.assert_array_equal(
        multisource_knn_codes(
            source_features,
            source_utility,
            target_features,
            neighbours=2,
        ),
        [1, 1],
    )
