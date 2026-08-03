"""Paired frozen-checkpoint pilot for randomly resampled code transport."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.dynamic_code_transport import (
    fit_transport_logits,
    mixed_transport_probabilities,
    paired_best_improvement,
    sample_code_indices,
)
from src.env import Env
from src.model import Model
from src.seed_sampler import SeedVectorSampler


CVRP100_DISTRIBUTIONS = {
    "uniform": {"use_X_generator": False, "rootPos": 2, "custPos": 1, "demandType": 2, "avgRouteSize": 3},
    "x_uniform_center": {"use_X_generator": True, "rootPos": 2, "custPos": 1, "demandType": 2, "avgRouteSize": 3},
    "x_cluster_center": {"use_X_generator": True, "rootPos": 2, "custPos": 2, "demandType": 2, "avgRouteSize": 3},
    "x_mixed_center": {"use_X_generator": True, "rootPos": 2, "custPos": 3, "demandType": 2, "avgRouteSize": 3},
    "x_cluster_corner_quad": {"use_X_generator": True, "rootPos": 3, "custPos": 2, "demandType": 6, "avgRouteSize": 3},
    "x_mixed_random_fewlarge": {"use_X_generator": True, "rootPos": 1, "custPos": 3, "demandType": 7, "avgRouteSize": 2},
}


def _rollout(model: Model, env: Env) -> np.ndarray:
    state = env.reset()
    done = False
    while not done:
        selected, _, _ = model(state)
        state, done = env.step(selected)
    return env.selected_node_list.detach().cpu().numpy()


def _source_prior(memory, target_name, code_pool, calibration_instances, ridge):
    names = [str(value) for value in memory["distribution_names"]]
    if target_name not in names:
        raise ValueError(f"target distribution {target_name!r} is absent from memory")
    utility = np.asarray(memory["utility"], dtype=np.float64)
    code_indices = np.asarray(memory["code_indices"], dtype=np.int64)
    if utility.ndim != 4 or utility.shape[0] != len(names):
        raise ValueError("memory utility shape is invalid")
    if utility.shape[2] != code_indices.size:
        raise ValueError("memory code and utility axes disagree")
    if not 1 <= calibration_instances <= utility.shape[1]:
        raise ValueError("invalid calibration instance count")
    source_mask = np.asarray([name != target_name for name in names])
    observed = utility[source_mask, :calibration_instances].mean(axis=(0, 1, 3))
    return fit_transport_logits(code_pool, code_indices, observed, ridge)


def _evaluate_distribution(
    generator_params,
    checkpoint,
    model,
    sampler,
    prior_probabilities,
    instances,
    rounds,
    rollout_size,
    seed,
    device,
    num_processes,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    env_params = dict(checkpoint["env_params"])
    env_params["generator_params"] = generator_params
    total_rollouts = 2 * rounds * rollout_size
    env = Env(num_processes=num_processes, **env_params)
    env.init_instances(instances, total_rollouts, device)
    incumbent = np.asarray(env.instanceSet.costs, dtype=np.float64)

    uniform = np.full(sampler.pool.shape[0], 1.0 / sampler.pool.shape[0])
    uniform_indices = sample_code_indices(
        uniform, instances, rounds, rollout_size, rng
    )
    transport_indices = sample_code_indices(
        prior_probabilities, instances, rounds, rollout_size, rng
    )
    indices = np.stack((uniform_indices, transport_indices), axis=1)
    flat_indices = torch.as_tensor(
        indices.reshape(instances, total_rollouts),
        dtype=torch.long,
        device=device,
    )
    z = sampler.pool[flat_indices]

    started = time.perf_counter()
    with torch.inference_mode():
        model.pre_forward(env.get_model_input(device), z)
        selected_nodes = _rollout(model, env)
    candidate = np.asarray(
        env.instanceSet.remove_recreate(
            selected_nodes,
            int(env_params["recreate_n"]),
            "singleImp",
            beta=float(env_params["beta"]),
            insert_in_new_tours_only=bool(env_params["insert_in_new_tours_only"]),
        ),
        dtype=np.float64,
    ).reshape(instances, 2, rounds, rollout_size)
    return {
        "incumbent_cost": incumbent,
        "candidate_cost": candidate,
        "best_improvement": paired_best_improvement(candidate, incumbent),
        "sampled_code_indices": indices,
        "wall_time_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/cvrp_100/checkpoint-2000.pt"),
    )
    parser.add_argument("--source-memory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--rollout-size", type=int, default=32)
    parser.add_argument("--calibration-instances", type=int, default=16)
    parser.add_argument("--exploration", type=float, default=0.5)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    use_cpu = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu") if use_cpu else torch.device("cuda", 0)
    if device.type == "cuda":
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
    code_pool = sampler.pool.detach().cpu().numpy()

    args.output.mkdir(parents=True, exist_ok=True)
    arrays = []
    prior_logits = []
    memory = np.load(args.source_memory)
    names = list(CVRP100_DISTRIBUTIONS)
    for index, (name, parameters) in enumerate(CVRP100_DISTRIBUTIONS.items()):
        logits = _source_prior(
            memory, name, code_pool, args.calibration_instances, args.ridge
        )
        probabilities = mixed_transport_probabilities(
            logits, args.exploration, args.prior_strength
        )
        prior_logits.append(logits)
        print(f"[{index + 1}/{len(names)}] {name}", flush=True)
        arrays.append(
            _evaluate_distribution(
                parameters, checkpoint, model, sampler, probabilities,
                args.instances, args.rounds, args.rollout_size,
                args.seed + 1000 * index, device, args.num_processes,
            )
        )

    np.savez_compressed(
        args.output / "raw_dynamic_transport.npz",
        distribution_names=np.asarray(names),
        strategy_names=np.asarray(["uniform_resample", "transport_resample"]),
        incumbent_cost=np.stack([item["incumbent_cost"] for item in arrays]),
        candidate_cost=np.stack([item["candidate_cost"] for item in arrays]),
        best_improvement=np.stack([item["best_improvement"] for item in arrays]),
        sampled_code_indices=np.stack([item["sampled_code_indices"] for item in arrays]),
        prior_logits=np.stack(prior_logits),
    )
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "source_memory": str(args.source_memory),
        "device": str(device),
        "instances_per_distribution": args.instances,
        "rounds": args.rounds,
        "rollout_size_per_strategy": args.rollout_size,
        "calibration_instances_per_source_distribution": args.calibration_instances,
        "exploration": args.exploration,
        "prior_strength": args.prior_strength,
        "ridge": args.ridge,
        "seed": args.seed,
        "target_codes_are_resampled": True,
        "fixed_target_codes": False,
        "wall_time_seconds": [item["wall_time_seconds"] for item in arrays],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
