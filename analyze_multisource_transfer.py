"""Confirm leave-one-distribution-out multi-source latent-code transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.crossdist_transfer import (
    bootstrap_mean_interval,
    choose_global_code,
    choose_multisource_mean_code,
    multisource_knn_codes,
    selected_utility,
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
    parser.add_argument("--neighbours", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    raw = np.load(args.run_dir / "raw_metrics.npz", allow_pickle=False)
    names = [str(name) for name in raw["distribution_names"]]
    code_indices = raw["code_indices"]
    features = raw["features"]
    utility = raw["utility"].mean(axis=3)
    distributions, instances, _ = utility.shape
    calibration = args.calibration_instances
    if distributions != len(names):
        raise ValueError("distribution names and utility tensor disagree")
    if not 1 <= calibration <= instances:
        raise ValueError("calibration-instances must lie within each distribution")

    metrics = {"targets": {}}
    markdown = [
        "# Multi-Source Leave-One-Distribution-Out Confirmation",
        "",
        "| Held-out target | Random reward | Mean code | Mean gain | Relative | "
        "95% CI | Negative transfer | Uniform gain | 5-NN gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target_index, name in enumerate(names):
        source_indices = [
            index for index in range(distributions) if index != target_index
        ]
        target_utility = utility[target_index]
        random_reward = target_utility.mean(axis=1)
        mean_code = choose_multisource_mean_code(
            utility, target_index, calibration
        )
        mean_difference = target_utility[:, mean_code] - random_reward
        mean_interval = bootstrap_mean_interval(
            mean_difference,
            args.seed + target_index,
            samples=args.bootstrap_samples,
        )
        relative_gain = float(
            mean_difference.mean() / max(abs(random_reward.mean()), 1e-12)
        )

        source_features = np.concatenate(
            [features[index, :calibration] for index in source_indices]
        )
        source_utility = np.concatenate(
            [utility[index, :calibration] for index in source_indices]
        )
        knn_codes = multisource_knn_codes(
            source_features,
            source_utility,
            features[target_index],
            neighbours=args.neighbours,
        )
        knn_difference = (
            selected_utility(target_utility, knn_codes) - random_reward
        )

        uniform_gain = None
        if name != "uniform":
            uniform_index = names.index("uniform")
            uniform_code = choose_global_code(
                utility[uniform_index, :calibration]
            )
            uniform_gain = float(
                (target_utility[:, uniform_code] - random_reward).mean()
            )

        metrics["targets"][name] = {
            "random_reward": float(random_reward.mean()),
            "mean_code_position": int(mean_code),
            "mean_code_index": int(code_indices[mean_code]),
            "mean_gain": _interval_dict(mean_interval),
            "mean_relative_gain": relative_gain,
            "mean_negative_transfer_rate": float((mean_difference < 0).mean()),
            "uniform_global_gain": uniform_gain,
            "multisource_knn_gain": float(knn_difference.mean()),
        }
        uniform_text = "n/a" if uniform_gain is None else f"{uniform_gain:+.6f}"
        markdown.append(
            f"| {name} | {random_reward.mean():.6f} | "
            f"{int(code_indices[mean_code])} | {mean_difference.mean():+.6f} | "
            f"{relative_gain:+.1%} | "
            f"[{mean_interval.low:+.6f}, {mean_interval.high:+.6f}] | "
            f"{(mean_difference < 0).mean():.1%} | {uniform_text} | "
            f"{knn_difference.mean():+.6f} |"
        )

    passing = sum(
        item["mean_relative_gain"] >= 0.10 and item["mean_gain"]["low"] > 0
        for item in metrics["targets"].values()
    )
    significantly_negative = sum(
        item["mean_gain"]["high"] < 0
        for item in metrics["targets"].values()
    )
    confirmed = passing >= 4 and significantly_negative == 0
    metrics["success_criteria"] = {
        "targets_passing_relative_and_ci_rule": int(passing),
        "targets_significantly_negative": int(significantly_negative),
        "h1b_confirmed": bool(confirmed),
        "proceed_to_full_search_protocol": bool(confirmed),
    }
    markdown.extend(
        [
            "",
            "## Locked decision rule",
            "",
            f"- Targets passing relative-gain and CI rule: {passing}/6.",
            f"- Targets significantly negative: {significantly_negative}/6.",
            f"- H1b confirmed: **{confirmed}**.",
            f"- Register full-LNS protocol: **{confirmed}**.",
            "",
        ]
    )
    (args.run_dir / "multisource_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (args.run_dir / "multisource_summary.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
