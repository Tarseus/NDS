"""Analyze raw metrics from the cross-distribution latent-code pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.crossdist_transfer import (
    bootstrap_mean_interval,
    choose_global_code,
    nearest_source_codes,
    selected_utility,
    split_half_reliability,
)


def _interval_dict(interval) -> dict:
    return {
        "mean": interval.mean,
        "low": interval.low,
        "high": interval.high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--calibration-instances", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    raw = np.load(args.run_dir / "raw_metrics.npz", allow_pickle=False)
    names = [str(name) for name in raw["distribution_names"]]
    features = raw["features"]
    utilities = raw["utility"]
    utility_mean = utilities.mean(axis=3)
    instances = utilities.shape[1]
    calibration = args.calibration_instances
    if not 1 <= calibration < instances:
        raise ValueError("calibration-instances must lie within each distribution")

    metrics = {
        "reliability": {},
        "uniform_source_transfer": {},
        "distribution_code_transfer": {},
    }
    markdown = [
        "# Cross-Distribution Latent-Code Pilot",
        "",
        "## Split-half code-rank reliability",
        "",
        "| Distribution | Mean Spearman | 95% CI | Median |",
        "|---|---:|---:|---:|",
    ]
    for index, name in enumerate(names):
        correlations = split_half_reliability(utilities[index])
        interval = bootstrap_mean_interval(
            correlations,
            args.seed + index,
            samples=args.bootstrap_samples,
        )
        metrics["reliability"][name] = {
            **_interval_dict(interval),
            "median": float(np.median(correlations)),
        }
        markdown.append(
            f"| {name} | {interval.mean:.3f} | "
            f"[{interval.low:.3f}, {interval.high:.3f}] | "
            f"{np.median(correlations):.3f} |"
        )

    source_index = names.index("uniform")
    source_features = features[source_index, :calibration]
    source_utility = utility_mean[source_index, :calibration]
    global_code = choose_global_code(source_utility)

    markdown.extend(
        [
            "",
            "## Transfer from uniform calibration memory",
            "",
            "| Target | Random reward | Global gain | kNN gain | "
            "kNN relative gain | kNN 95% CI | kNN negative transfer | "
            "Target-calibration gain | Oracle gain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target_index, name in enumerate(names):
        target_features = features[target_index, calibration:]
        target_utility = utility_mean[target_index, calibration:]
        target_calibration_utility = utility_mean[target_index, :calibration]
        random_reward = target_utility.mean(axis=1)
        global_reward = target_utility[:, global_code]
        retrieved_codes = nearest_source_codes(
            source_features, source_utility, target_features
        )
        knn_reward = selected_utility(target_utility, retrieved_codes)
        global_difference = global_reward - random_reward
        knn_difference = knn_reward - random_reward
        interval = bootstrap_mean_interval(
            knn_difference,
            args.seed + 100 + target_index,
            samples=args.bootstrap_samples,
        )
        relative_gain = float(
            knn_difference.mean() / max(abs(random_reward.mean()), 1e-12)
        )
        negative_transfer = float((knn_difference < 0).mean())
        target_calibration_code = choose_global_code(target_calibration_utility)
        target_calibration_difference = (
            target_utility[:, target_calibration_code] - random_reward
        )
        oracle_difference = target_utility.max(axis=1) - random_reward
        metrics["uniform_source_transfer"][name] = {
            "random_reward": float(random_reward.mean()),
            "global_code": int(global_code),
            "global_gain": float(global_difference.mean()),
            "knn_gain": _interval_dict(interval),
            "knn_relative_gain": relative_gain,
            "knn_negative_transfer_rate": negative_transfer,
            "target_calibration_code": int(target_calibration_code),
            "target_calibration_gain": float(target_calibration_difference.mean()),
            "per_instance_oracle_gain": float(oracle_difference.mean()),
        }
        markdown.append(
            f"| {name} | {random_reward.mean():.6f} | "
            f"{global_difference.mean():+.6f} | {knn_difference.mean():+.6f} | "
            f"{relative_gain:+.1%} | [{interval.low:+.6f}, {interval.high:+.6f}] | "
            f"{negative_transfer:.1%} | "
            f"{target_calibration_difference.mean():+.6f} | "
            f"{oracle_difference.mean():+.6f} |"
        )

    transfer_matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    for source_index, source_name in enumerate(names):
        code = choose_global_code(
            utility_mean[source_index, :calibration]
        )
        metrics["distribution_code_transfer"][source_name] = {
            "selected_code": int(code),
            "targets": {},
        }
        for target_index, target_name in enumerate(names):
            target = utility_mean[target_index, calibration:]
            difference = target[:, code] - target.mean(axis=1)
            transfer_matrix[source_index, target_index] = difference.mean()
            metrics["distribution_code_transfer"][source_name]["targets"][
                target_name
            ] = float(difference.mean())

    markdown.extend(
        [
            "",
            "## Distribution-best code transfer gain over random",
            "",
            "| Source \\ Target | " + " | ".join(names) + " |",
            "|---|" + "---:|" * len(names),
        ]
    )
    for row, source_name in zip(transfer_matrix, names):
        markdown.append(
            f"| {source_name} | "
            + " | ".join(f"{value:+.6f}" for value in row)
            + " |"
        )

    metrics["success_criteria"] = {
        "reliability_distributions_passing": int(
            sum(
                item["median"] >= 0.20 and item["low"] > 0
                for item in metrics["reliability"].values()
            )
        ),
        "knn_ood_distributions_passing": int(
            sum(
                item["knn_relative_gain"] >= 0.10
                and item["knn_gain"]["low"] > 0
                for name, item in metrics["uniform_source_transfer"].items()
                if name != "uniform"
            )
        ),
    }
    metrics["success_criteria"]["proceed_to_full_search"] = bool(
        metrics["success_criteria"]["reliability_distributions_passing"] >= 3
        and metrics["success_criteria"]["knn_ood_distributions_passing"] >= 3
    )
    markdown.extend(
        [
            "",
            "## Locked decision rule",
            "",
            f"- Reliability distributions passing: "
            f"{metrics['success_criteria']['reliability_distributions_passing']}/6.",
            f"- kNN OOD distributions passing: "
            f"{metrics['success_criteria']['knn_ood_distributions_passing']}/5.",
            f"- Proceed to full-search experiment: "
            f"**{metrics['success_criteria']['proceed_to_full_search']}**.",
            "",
        ]
    )

    (args.run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (args.run_dir / "summary.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
