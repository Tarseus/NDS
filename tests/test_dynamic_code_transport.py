import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "dynamic_code_transport",
    Path(__file__).parents[1] / "src" / "dynamic_code_transport.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

binary_code_features = _MODULE.binary_code_features
fit_transport_logits = _MODULE.fit_transport_logits
mixed_transport_probabilities = _MODULE.mixed_transport_probabilities
paired_best_improvement = _MODULE.paired_best_improvement
sample_code_indices = _MODULE.sample_code_indices


def test_transport_prior_scores_full_binary_pool():
    pool = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    assert binary_code_features(pool).shape == (4, 3)
    logits = fit_transport_logits(pool, np.arange(4), pool[:, 0], ridge=0.1)
    assert logits[2] > logits[0]
    assert logits[3] > logits[1]


def test_mixture_preserves_uniform_exploration_and_normalization():
    probabilities = mixed_transport_probabilities(
        np.asarray([-2.0, 0.0, 2.0]), exploration=0.4, strength=1.0
    )
    np.testing.assert_allclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0.4 / 3.0)
    assert probabilities[2] > probabilities[1] > probabilities[0]


def test_sampling_resamples_without_replacement():
    indices = sample_code_indices(
        np.full(8, 1 / 8), 3, 4, 5, np.random.default_rng(7)
    )
    assert indices.shape == (3, 4, 5)
    for row in indices.reshape(-1, 5):
        assert len(set(row.tolist())) == 5
    assert not np.array_equal(indices[0, 0], indices[0, 1])


def test_paired_best_improvement_keeps_leading_axes():
    costs = np.asarray([[[10.0, 8.0], [9.0, 7.0]]])
    improvement = paired_best_improvement(costs, np.asarray([10.0]))
    np.testing.assert_allclose(improvement, [[0.2, 0.3]])
