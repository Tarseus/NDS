"""Run equal-rollout iterative LNS strategies on fresh CVRP100 instances."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.crossdist_transfer import CVRP100_DISTRIBUTIONS, fixed_code_indices
from src.env import Env
from src.full_lns_transfer import (
    geometric_temperatures,
    random_fixed_assignments,
)
from src.model import Model
from src.reproducibility import seed_everything
from src.seed_sampler import SeedVectorSampler


STRATEGIES = (
    "random_fixed",
    "multisource_fixed",
    "random_panel",
    "uniform_fixed",
)
MULTISOURCE_CODES = {
    "uniform": 959,
    "x_uniform_center": 175,
    "x_cluster_center": 175,
    "x_mixed_center": 175,
    "x_cluster_corner_quad": 959,
    "x_mixed_random_fewlarge": 175,
}
UNIFORM_CODE = 910


def _rollout(model: Model, env: Env, softmax_temp: float) -> np.ndarray:
    state = env.reset()
    done = False
    while not done:
        selected, _, _ = model(state, softmax_temp)
        state, done = env.step(selected)
    return env.selected_node_list.detach().cpu().numpy()


def _strategy_codes(
    strategy: str,
    sampler: SeedVectorSampler,
    panel_indices: np.ndarray,
    random_instance_codes: np.ndarray,
    distribution_name: str,
    batch_size: int,
    rollout_size: int,
) -> torch.Tensor:
    if strategy == "random_fixed":
        index = torch.as_tensor(
            random_instance_codes, dtype=torch.long, device=sampler.device
        )
        return sampler.pool[index][:, None, :].expand(
            batch_size, rollout_size, sampler.z_dim
        )
    if strategy == "multisource_fixed":
        index = torch.as_tensor(
            [MULTISOURCE_CODES[distribution_name]],
            dtype=torch.long,
            device=sampler.device,
        )
        return sampler.repeat_fixed(index, batch_size, replicas=rollout_size)
    if strategy == "random_panel":
        index = torch.as_tensor(
            panel_indices, dtype=torch.long, device=sampler.device
        )
        return sampler.repeat_fixed(index, batch_size, replicas=1)
    if strategy == "uniform_fixed":
        index = torch.as_tensor(
            [UNIFORM_CODE], dtype=torch.long, device=sampler.device
        )
        return sampler.repeat_fixed(index, batch_size, replicas=rollout_size)
    raise ValueError(f"unknown strategy: {strategy}")


def _run_strategy(
    strategy: str,
    distribution_name: str,
    generator_params: dict,
    checkpoint: dict,
    model: Model,
    sampler: SeedVectorSampler,
    panel_indices: np.ndarray,
    random_instance_codes: np.ndarray,
    instances: int,
    iterations: int,
    checkpoints: np.ndarray,
    rollout_size: int,
    instance_seed: int,
    device: torch.device,
    num_processes: int,
) -> tuple[np.ndarray, float]:
    seed_everything(instance_seed)
    env_params = dict(checkpoint["env_params"])
    env_params["generator_params"] = generator_params
    env_params["random_seed"] = instance_seed
    env = Env(num_processes=num_processes, **env_params)
    env.init_instances(instances, rollout_size, device)
    initial = np.asarray(env.instanceSet.costs, dtype=np.float64)
    best = initial.copy()
    trace = np.empty((instances, len(checkpoints)), dtype=np.float64)
    trace[:, 0] = initial
    checkpoint_position = {int(value): index for index, value in enumerate(checkpoints)}
    temperatures = geometric_temperatures(iterations, 0.1, 0.001)
    z = _strategy_codes(
        strategy,
        sampler,
        panel_indices,
        random_instance_codes,
        distribution_name,
        instances,
        rollout_size,
    )

    started = time.perf_counter()
    with torch.inference_mode():
        for iteration, temperature in enumerate(temperatures, start=1):
            model.pre_forward(env.get_model_input(device), z)
            selected_nodes = _rollout(model, env, 1.0)
            env.instanceSet.remove_recreate(
                selected_nodes,
                int(env_params["recreate_n"]),
                "allImp",
                T=float(temperature),
                beta=float(env_params["beta"]),
                insert_in_new_tours_only=bool(
                    env_params["insert_in_new_tours_only"]
                ),
            )
            best = np.minimum(
                best, np.asarray(env.instanceSet.costs, dtype=np.float64)
            )
            if iteration in checkpoint_position:
                trace[:, checkpoint_position[iteration]] = best
    elapsed = time.perf_counter() - started
    del env
    gc.collect()
    return trace, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/cvrp_100/checkpoint-2000.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rollout-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--random-code-seed", type=int, default=20260807)
    parser.add_argument("--code-seed", type=int, default=20260803)
    parser.add_argument("--num-processes", type=int, default=8)
    args = parser.parse_args()

    if args.rollout_size != 32:
        raise ValueError("the locked protocol requires rollout-size 32")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
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
    panel_indices = fixed_code_indices(
        sampler.pool.shape[0], args.rollout_size, args.code_seed
    )
    names = list(CVRP100_DISTRIBUTIONS)
    assignments = random_fixed_assignments(
        panel_indices,
        len(names),
        args.instances,
        args.random_code_seed,
    )
    checkpoints = np.asarray([0, 1, 2, 5, 10, 20, 50, 100, 200])
    if args.iterations != int(checkpoints[-1]):
        raise ValueError("the locked protocol requires 200 iterations")

    all_traces = np.empty(
        (len(names), len(STRATEGIES), args.instances, len(checkpoints)),
        dtype=np.float64,
    )
    wall_times = np.empty((len(names), len(STRATEGIES)), dtype=np.float64)
    for distribution_index, (name, parameters) in enumerate(
        CVRP100_DISTRIBUTIONS.items()
    ):
        reference_initial = None
        for strategy_index, strategy in enumerate(STRATEGIES):
            print(
                f"[{distribution_index + 1}/6][{strategy_index + 1}/4] "
                f"{name} {strategy}",
                flush=True,
            )
            trace, elapsed = _run_strategy(
                strategy=strategy,
                distribution_name=name,
                generator_params=parameters,
                checkpoint=checkpoint,
                model=model,
                sampler=sampler,
                panel_indices=panel_indices,
                random_instance_codes=assignments[distribution_index],
                instances=args.instances,
                iterations=args.iterations,
                checkpoints=checkpoints,
                rollout_size=args.rollout_size,
                instance_seed=args.seed + distribution_index,
                device=device,
                num_processes=args.num_processes,
            )
            if reference_initial is None:
                reference_initial = trace[:, 0].copy()
            elif not np.allclose(reference_initial, trace[:, 0], rtol=0, atol=1e-8):
                raise RuntimeError("strategy starting costs are not identical")
            all_traces[distribution_index, strategy_index] = trace
            wall_times[distribution_index, strategy_index] = elapsed

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "raw_full_lns.npz",
        distribution_names=np.asarray(names),
        strategy_names=np.asarray(STRATEGIES),
        checkpoints=checkpoints,
        best_cost=all_traces,
        wall_time_seconds=wall_times,
        code_indices=panel_indices,
        random_fixed_codes=assignments,
    )
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "instances_per_distribution": args.instances,
        "iterations": args.iterations,
        "rollout_size": args.rollout_size,
        "instance_seed": args.seed,
        "random_code_seed": args.random_code_seed,
        "code_seed": args.code_seed,
        "checkpoints": checkpoints.tolist(),
        "strategies": list(STRATEGIES),
        "multisource_codes": MULTISOURCE_CODES,
        "uniform_code": UNIFORM_CODE,
        "wall_time_seconds": wall_times.tolist(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
