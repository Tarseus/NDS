import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "src" / "fisher_dpp.py"
SPEC = importlib.util.spec_from_file_location("fisher_dpp", MODULE_PATH)
fisher_dpp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fisher_dpp)
fisher_volume_loss = fisher_dpp.fisher_volume_loss
route_sqrt_features = fisher_dpp.route_sqrt_features


def test_route_features_preserve_pcvrp_unvisited_probability_mass():
    probs = torch.tensor(
        [[[0.10, 0.20, 0.30, 0.15, 0.25]]], requires_grad=True
    )
    # Customers 0/1 share route 0, customer 2 uses route 1, and customers
    # 3/4 are unvisited. The unvisited customers must remain distinct.
    tour_index = torch.tensor([[0, 0, 1, -1, -1]])

    features, route_probs = route_sqrt_features(probs, tour_index)

    expected = torch.tensor([[[0.30, 0.30, 0.15, 0.25, 0.00]]])
    assert torch.allclose(route_probs, expected)
    assert torch.allclose(route_probs.sum(dim=2), torch.ones(1, 1))
    assert torch.allclose(features.norm(dim=2), torch.ones(1, 1))

    features.sum().backward()
    assert torch.isfinite(probs.grad).all()


def test_leave_one_out_marginals_match_direct_logdet_differences():
    features = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]]],
        requires_grad=True,
    )
    quality = torch.tensor([[1.0, 2.0, 0.5]])
    trajectory_log_prob = torch.zeros(1, 3, requires_grad=True)

    loss, diagnostics = fisher_volume_loss(
        features, quality, trajectory_log_prob
    )

    weighted = quality.sqrt().unsqueeze(2) * features.detach()
    kernel = torch.eye(3).unsqueeze(0) + weighted @ weighted.transpose(1, 2)
    full_logdet = torch.linalg.slogdet(kernel).logabsdet
    direct_marginals = []
    for i in range(3):
        keep = torch.tensor([j for j in range(3) if j != i])
        reduced = kernel[:, keep][:, :, keep]
        direct_marginals.append(
            (full_logdet - torch.linalg.slogdet(reduced).logabsdet).item()
        )

    inverse_diagonal = torch.diagonal(torch.linalg.inv(kernel), dim1=1, dim2=2)
    identity_marginals = -torch.log(inverse_diagonal).squeeze(0)
    assert torch.allclose(
        identity_marginals,
        torch.tensor(direct_marginals),
        atol=1e-6,
    )
    assert torch.allclose(diagnostics["logdet"], full_logdet.mean())

    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(trajectory_log_prob.grad).all()


def test_duplicate_features_have_smaller_volume_than_orthogonal_features():
    duplicate = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    orthogonal = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    quality = torch.ones(1, 2)
    log_prob = torch.zeros(1, 2)

    _, duplicate_metrics = fisher_volume_loss(duplicate, quality, log_prob)
    _, orthogonal_metrics = fisher_volume_loss(orthogonal, quality, log_prob)

    assert orthogonal_metrics["logdet"] > duplicate_metrics["logdet"]
