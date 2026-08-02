"""Analyze the pre-registered multi-source top-8 full-LNS experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.crossdist_transfer import bootstrap_mean_interval
from src.full_lns_transfer import normalized_anytime_auc


def _interval(values: np.ndarray, seed: int, samples: int) -> dict:
    result = bootstrap_mean_interval(values, seed, samples=samples)
    return {"mean": result.mean, "low": result.low, "high": result.high}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    raw = np.load(args.run_dir / "raw_full_lns.npz", allow_pickle=False)
    names = [str(value) for value in raw["distribution_names"]]
    strategies = [str(value) for value in raw["strategy_names"]]
    checkpoints = raw["checkpoints"]
    costs = raw["best_cost"]
    random_index = strategies.index("random_fixed")
    top8_index = strategies.index("multisource_top8")
    panel_index = strategies.index("random_panel")

    metrics = {"targets": {}}
    markdown = [
        "# Multi-Source Top-8 Full-LNS Transfer",
        "",
        "| Target | Top-8 final improve | vs panel final | 95% CI | "
        "vs panel AUC | 95% CI | vs random final |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    final_pass = 0
    auc_pass = 0
    significant_harm = 0
    for index, name in enumerate(names):
        initial = costs[index, random_index, :, 0]
        if not np.allclose(costs[index, :, :, 0], initial[None, :], atol=1e-8):
            raise RuntimeError("strategy starting costs differ")
        random_final = costs[index, random_index, :, -1]
        top8_final = costs[index, top8_index, :, -1]
        panel_final = costs[index, panel_index, :, -1]
        denominator = np.maximum(np.abs(initial), 1e-12)
        top8_improvement = (initial - top8_final) / denominator
        panel_advantage = (panel_final - top8_final) / denominator
        random_advantage = (random_final - top8_final) / denominator
        panel_auc = normalized_anytime_auc(
            costs[index, panel_index], checkpoints, initial
        )
        top8_auc = normalized_anytime_auc(
            costs[index, top8_index], checkpoints, initial
        )
        auc_advantage = top8_auc - panel_auc
        final_interval = _interval(
            panel_advantage, args.seed + index, args.bootstrap_samples
        )
        auc_interval = _interval(
            auc_advantage, args.seed + 100 + index, args.bootstrap_samples
        )
        if final_interval["low"] > 0:
            final_pass += 1
        if auc_interval["low"] > 0:
            auc_pass += 1
        if final_interval["high"] < 0 or auc_interval["high"] < 0:
            significant_harm += 1
        metrics["targets"][name] = {
            "top8_final_normalized_improvement": float(top8_improvement.mean()),
            "top8_vs_panel_final": final_interval,
            "top8_vs_panel_anytime_auc": auc_interval,
            "top8_vs_random_final_mean": float(random_advantage.mean()),
        }
        markdown.append(
            f"| {name} | {top8_improvement.mean():+.2%} | "
            f"{final_interval['mean']:+.2%} | "
            f"[{final_interval['low']:+.2%}, {final_interval['high']:+.2%}] | "
            f"{auc_interval['mean']:+.4f} | "
            f"[{auc_interval['low']:+.4f}, {auc_interval['high']:+.4f}] | "
            f"{random_advantage.mean():+.2%} |"
        )

    h4_passed = final_pass >= 4 and auc_pass >= 4 and significant_harm == 0
    metrics["success_criteria"] = {
        "final_distributions_passing": final_pass,
        "anytime_auc_distributions_passing": auc_pass,
        "distributions_with_significant_primary_harm": significant_harm,
        "h4_passed": bool(h4_passed),
    }
    markdown.extend(
        [
            "",
            "## Locked decision rule",
            "",
            f"- Final-cost distributions passing: {final_pass}/6.",
            f"- Anytime-AUC distributions passing: {auc_pass}/6.",
            f"- Distributions with significant primary harm: {significant_harm}/6.",
            f"- H4 passed: **{h4_passed}**.",
            "",
        ]
    )
    (args.run_dir / "portfolio_lns_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (args.run_dir / "portfolio_lns_summary.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
