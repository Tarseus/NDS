"""Summarize paired dynamic-code transport results with bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _bootstrap(values, rng, samples):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    draws = rng.integers(0, values.size, size=(samples, values.size))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    raw = np.load(args.run_dir / "raw_dynamic_transport.npz")
    names = [str(value) for value in raw["distribution_names"]]
    improvement = np.asarray(raw["best_improvement"], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    rows = []
    for index, name in enumerate(names):
        uniform = improvement[index, :, 0].mean(axis=1)
        transport = improvement[index, :, 1].mean(axis=1)
        mean, low, high = _bootstrap(
            transport - uniform, rng, args.bootstrap_samples
        )
        rows.append({
            "distribution": name,
            "uniform_mean": float(uniform.mean()),
            "transport_mean": float(transport.mean()),
            "paired_delta_mean": mean,
            "paired_delta_ci_low": low,
            "paired_delta_ci_high": high,
            "positive": bool(low > 0),
            "negative": bool(high < 0),
        })
    summary = {
        "rows": rows,
        "positive_distributions": sum(row["positive"] for row in rows),
        "negative_distributions": sum(row["negative"] for row in rows),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.run_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
