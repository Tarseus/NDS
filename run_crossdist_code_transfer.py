"""Measure latent-code reliability and cross-distribution transfer in CVRP100."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.crossdist_transfer import (
    CVRP100_DISTRIBUTIONS,
    FEATURE_NAMES,
    build_code_major_panel,
    extract_instance_features,
    fixed_code_indices,
    reshape_code_replicas,
)
from src.env import Env
from src.model import Model
from src.reproducibility import seed_everything
from src.seed_sampler import SeedVectorSampler


def _rollout(model: Model, env: Env) -> tuple[np.ndarray, np.ndarray]:
    state = env.reset()
    first_step_probabilities = None
    done = False
    while not done:
        selected, _, all_probabilities = model(state)
        if first_step_probabilities is None:
            first_step_probabilities = all_probabilities.detach().float().cpu().numpy()
        state, done = env.step(selected)
    selected_nodes = env.selected_node_list.detach().cpu().numpy()
    return selected_nodes, first_step_probabilities


def _evaluate_distribution(
    name: str,
    generator_params: dict,
    checkpoint: dict,
    model: Model,
    sampler: SeedVectorSampler,
    code_indices: np.ndarray,
    instances: int,
    replicas: int,
    seed: int,
    device: torch.device,
    num_processes: int,
) -> dict:
    seed_everything(seed)
    env_params = dict(checkpoint["env_params"])
    env_params["generator_params"] = generator_params
    env_params["random_seed"] = seed
    rollout_size = len(code_indices) * replicas
    env = Env(num_processes=num_processes, **env_params)
    env.init_instances(instances, rollout_size, device)

    z = build_code_major_panel(
        sampler.pool, code_indices, env.batch_size, replicas
    )
    old_cost = np.asarray(env.instanceSet.costs, dtype=np.float64)
    with torch.inference_mode():
        model.pre_forward(env.get_model_input(device), z)
        selected_nodes, first_step_probability = _rollout(model, env)

    new_cost = np.asarray(
        env.instanceSet.remove_recreate(
            selected_nodes,
            int(env_params["recreate_n"]),
            "singleImp",
            beta=float(env_params["beta"]),
            insert_in_new_tours_only=bool(
                env_params["insert_in_new_tours_only"]
            ),
        ),
        dtype=np.float64,
    )
    normalized_reward = np.maximum(old_cost[:, None] - new_cost, 0.0)
    normalized_reward /= np.maximum(np.abs(old_cost[:, None]), 1e-12)

    features = extract_instance_features(
        env.problem_feat.depot_xy,
        env.problem_feat.node_xy,
        torch.as_tensor(
            env.problem_data.depot_node_demand[:, 1:], dtype=torch.float32
        ),
        torch.as_tensor(env.problem_data.capacity, dtype=torch.float32),
    )
    result = {
        "name": name,
        "features": features,
        "old_cost": old_cost,
        "utility": reshape_code_replicas(
            normalized_reward, len(code_indices), replicas
        ),
        "first_step_probability": first_step_probability.reshape(
            instances, len(code_indices), replicas, -1
        ).mean(axis=2),
    }
    del env
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/cvrp_100/checkpoint-2000.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=64)
    parser.add_argument("--num-codes", type=int, default=32)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--code-seed", type=int, default=20260803)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.instances < 2:
        raise ValueError("instances must be at least two")
    if args.replicas < 2:
        raise ValueError("replicas must be at least two for split-half reliability")
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model_params = dict(checkpoint["model_params"])
    model_params["eval_type"] = "softmax"
    model = Model(**model_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    sampler = SeedVectorSampler(int(model_params["z_dim"]), device)
    code_indices = fixed_code_indices(
        sampler.pool.shape[0], args.num_codes, args.code_seed
    )

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    for index, (name, parameters) in enumerate(CVRP100_DISTRIBUTIONS.items()):
        print(f"[{index + 1}/{len(CVRP100_DISTRIBUTIONS)}] {name}", flush=True)
        results.append(
            _evaluate_distribution(
                name=name,
                generator_params=parameters,
                checkpoint=checkpoint,
                model=model,
                sampler=sampler,
                code_indices=code_indices,
                instances=args.instances,
                replicas=args.replicas,
                seed=args.seed + 1000 * index,
                device=device,
                num_processes=args.num_processes,
            )
        )

    np.savez_compressed(
        args.output / "raw_metrics.npz",
        distribution_names=np.asarray([result["name"] for result in results]),
        code_indices=code_indices,
        feature_names=np.asarray(FEATURE_NAMES),
        features=np.stack([result["features"] for result in results]),
        old_cost=np.stack([result["old_cost"] for result in results]),
        utility=np.stack([result["utility"] for result in results]),
        first_step_probability=np.stack(
            [result["first_step_probability"] for result in results]
        ),
    )
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "instances_per_distribution": args.instances,
        "num_codes": args.num_codes,
        "replicas": args.replicas,
        "seed": args.seed,
        "code_seed": args.code_seed,
        "code_indices": code_indices.tolist(),
        "distributions": CVRP100_DISTRIBUTIONS,
        "wall_time_seconds": time.perf_counter() - started,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
