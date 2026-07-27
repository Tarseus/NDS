import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


cppimport = types.ModuleType("cppimport")
cppimport.__path__ = []
sys.modules.setdefault("cppimport", cppimport)
sys.modules.setdefault("cppimport.import_hook", types.ModuleType("cppimport.import_hook"))

MODULE_PATH = Path(__file__).parents[1] / "src" / "instance_set.py"
SPEC = importlib.util.spec_from_file_location("instance_set", MODULE_PATH)
instance_set = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(instance_set)


class FakeSolution:
    def __init__(self, cost, label="incumbent"):
        self.totalCosts = float(cost)
        self.label = label

    def getTourList(self):
        return [self.label]


class FakeOps:
    @staticmethod
    def remove_recreate_singleImp(
        solution, selected_nodes, beta, recreate_n, insert_in_new_tours_only
    ):
        del beta, recreate_n, insert_in_new_tours_only
        improvements = np.asarray(selected_nodes)[:, 0].astype(float)
        costs = solution.totalCosts - improvements
        winner = int(np.argmin(costs))
        return FakeSolution(costs[winner], f"proposal-{improvements[winner]:g}"), costs

    @staticmethod
    def remove_recreate_singleImp_priority(
        solution,
        selected_nodes,
        priorities,
        beta,
        recreate_n,
        insert_in_new_tours_only,
    ):
        del beta, recreate_n, insert_in_new_tours_only
        improvements = np.asarray(selected_nodes)[:, 0].astype(float)
        costs = solution.totalCosts - improvements
        ranking = np.lexsort(
            (np.arange(costs.shape[0]), -np.asarray(priorities), costs)
        )
        winner = int(ranking[0])
        return FakeSolution(costs[winner], f"proposal-{winner}"), costs


def test_two_replicas_share_incumbent_and_only_first_is_committed():
    solutions = [FakeSolution(100.0)]
    costs = [100.0]
    tours = [solutions[0].getTourList()]
    selected = np.array([[[1], [3], [2], [10], [5], [6]]])
    priorities = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    data = (
        selected,
        3,
        1,
        0.0,
        0.0,
        True,
        priorities,
        np.array([0.5]),
        0,
    )

    result = instance_set._handle_remove_recreate_portfolio(
        FakeOps, solutions, costs, tours, data
    )
    candidate_costs, winners, runners, accepted = result

    np.testing.assert_allclose(candidate_costs[0], [[99, 97, 98], [90, 95, 94]])
    np.testing.assert_array_equal(winners[0], [1, 0])
    np.testing.assert_array_equal(runners[0], [2, 2])
    assert accepted == [True]
    assert solutions[0].totalCosts == 97.0
    assert costs == [97.0]


def test_action_independent_priority_breaks_exact_cost_ties():
    solutions = [FakeSolution(100.0)]
    costs = [100.0]
    tours = [solutions[0].getTourList()]
    selected = np.array([[[3], [3], [1]]])
    priorities = np.array([[[0.2, 0.9, 0.1]]])
    data = (
        selected,
        3,
        1,
        0.0,
        0.0,
        True,
        priorities,
        np.array([0.5]),
        0,
    )

    _, winners, runners, _ = instance_set._handle_remove_recreate_portfolio(
        FakeOps, solutions, costs, tours, data
    )
    np.testing.assert_array_equal(winners[0], [1])
    np.testing.assert_array_equal(runners[0], [0])
    assert solutions[0].label == "proposal-1"


def test_zero_temperature_rejects_a_worse_portfolio_winner():
    incumbent = FakeSolution(100.0)
    solutions = [incumbent]
    costs = [100.0]
    tours = [incumbent.getTourList()]
    selected = np.array([[[-1], [-3], [-2]]])
    data = (
        selected,
        3,
        1,
        0.0,
        0.0,
        True,
        np.array([[[0.1, 0.2, 0.3]]]),
        np.array([0.5]),
        0,
    )

    _, winners, _, accepted = instance_set._handle_remove_recreate_portfolio(
        FakeOps, solutions, costs, tours, data
    )
    np.testing.assert_array_equal(winners[0], [0])
    assert accepted == [False]
    assert solutions[0] is incumbent
    assert costs == [100.0]
