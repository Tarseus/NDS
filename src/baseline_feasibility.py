"""Runtime instrumentation for a reward-only NDS baseline.

This module is installed only by ``train_baseline_feasibility.py``.  It keeps
the original policy loss and incumbent update, while adding:

* exact random-subset compression curves from the existing rollout rewards;
* low-frequency, non-committing repeated tournaments for fixed code panels;
* encoder/decoder/repair timing on diagnostic steps;
* epoch summaries plus per-probe CSV records.

The shadow probe restores PyTorch RNG state and never enters the loss.  The C++
repair implementation uses process-global random engines, so a probe advances
those engines; it does not change their distribution, but diagnostic runs are
not bitwise replay-equivalent to an uninstrumented run.
"""

import csv
import math
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .env import StepState
from .feasibility_diagnostics import (
    compression_curve,
    repeated_tournament_metrics,
)


_INSTALLED = False


def _diagnostic_worker(
    problem: str,
    input_queue,
    result_queue,
    starting_solution_params: dict,
) -> None:
    """Instance worker with an additional non-committing evaluation command."""
    from . import instance_set as instance_set_module

    nds_ops = instance_set_module._load_cpp_operations(problem)
    instances = []
    solutions = []
    solution_costs = []
    tours = []

    try:
        while True:
            mode, data = input_queue.get()
            if mode == "new_instance":
                instances, solutions, solution_costs, tours = (
                    instance_set_module._handle_new_instances(
                        nds_ops, problem, data, starting_solution_params
                    )
                )
                result_queue.put([solution_costs, tours])
            elif mode == "remove_recreate":
                candidate_costs = instance_set_module._handle_remove_recreate(
                    nds_ops, solutions, solution_costs, tours, data
                )
                result_queue.put([candidate_costs, solution_costs, tours])
            elif mode == "evaluate_remove_recreate":
                selected_nodes, recreate_n, beta, insert_new_only = data
                candidate_costs = []
                for index, solution in enumerate(solutions):
                    _, costs = nds_ops.remove_recreate_singleImp(
                        solution,
                        selected_nodes[index],
                        beta,
                        recreate_n,
                        insert_new_only,
                    )
                    candidate_costs.append(costs)
                result_queue.put(candidate_costs)
    except Exception as error:
        print(f"Diagnostic worker exception occurred: {error}", flush=True)


def _evaluate_remove_recreate_mp(
    self,
    selected_nodes: np.ndarray,
    recreate_n: int,
    beta: float,
    insert_in_new_tours_only: bool,
) -> List:
    """Evaluate candidates in workers without changing their incumbents."""
    instances_per_process = math.ceil(self.batch_size / self.num_processes)
    for index, (_, input_queue, _) in enumerate(self.processes):
        begin = index * instances_per_process
        end = begin + instances_per_process
        input_queue.put(
            [
                "evaluate_remove_recreate",
                [
                    selected_nodes[begin:end],
                    recreate_n,
                    beta,
                    insert_in_new_tours_only,
                ],
            ]
        )

    candidate_costs = []
    for _, _, output_queue in self.processes:
        candidate_costs.extend(output_queue.get())
    return candidate_costs


def _evaluate_remove_recreate_sp(
    self,
    selected_nodes: np.ndarray,
    recreate_n: int,
    beta: float,
    insert_in_new_tours_only: bool,
) -> List:
    """Evaluate candidates locally without changing the current solutions."""
    candidate_costs = []
    for index, solution in enumerate(self._solutions):
        _, costs = self.NDSOps.remove_recreate_singleImp(
            solution,
            selected_nodes[index],
            beta,
            recreate_n,
            insert_in_new_tours_only,
        )
        candidate_costs.append(costs)
    return candidate_costs


def _evaluate_remove_recreate(
    self,
    selected_nodes: np.ndarray,
    recreate_n: int,
    beta: float = 0.0,
    insert_in_new_tours_only: bool = True,
) -> List:
    """Return singleImp candidate costs without committing a best solution."""
    if self.use_multiprocessing:
        return _evaluate_remove_recreate_mp(
            self,
            selected_nodes,
            recreate_n,
            beta,
            insert_in_new_tours_only,
        )
    return _evaluate_remove_recreate_sp(
        self,
        selected_nodes,
        recreate_n,
        beta,
        insert_in_new_tours_only,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _setup_diagnostics(self) -> None:
    params = dict(self.trainer_params.get("feasibility_diagnostics", {}))
    self.diag_enabled = bool(params.get("enabled", True))
    self.diag_probe_enabled = bool(params.get("probe_enabled", True))
    self.diag_subset_sizes = tuple(
        int(size) for size in params.get("subset_sizes", [8, 16, 32, 64])
    )
    self.diag_panel_size = int(params.get("panel_size", 64))
    self.diag_probe_interval = int(params.get("probe_interval", 97))
    self.diag_probe_seed = int(params.get("probe_seed", 20260726))
    self.diag_improvement_eps = float(params.get("improvement_eps", 1e-8))
    if self.diag_probe_interval < 1:
        raise ValueError("feasibility probe_interval must be positive")
    if self.diag_panel_size > self.seed_sampler.pool.size(0):
        raise ValueError("feasibility panel_size exceeds the binary code pool")
    if 2 * self.diag_panel_size < 2:
        raise ValueError("feasibility panel_size must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(self.diag_probe_seed)
    permutation = torch.randperm(self.seed_sampler.pool.size(0), generator=generator)
    self.diag_code_order = permutation.to(self.device)
    self.diag_train_calls = 0
    self.diag_probe_calls = 0
    self.diag_epoch = self.start_epoch
    self.diag_probe_csv = os.path.join(
        self.results_dir, "feasibility_probe_batches.csv"
    )
    self.diag_epoch_csv = os.path.join(self.results_dir, "feasibility_epochs.csv")


def _make_probe_codes(self, batch_size: int) -> Tuple[torch.Tensor, int]:
    pool_size = self.diag_code_order.numel()
    panel_count = max(1, pool_size // self.diag_panel_size)
    panel_id = self.diag_probe_calls % panel_count
    begin = panel_id * self.diag_panel_size
    indices = self.diag_code_order[begin : begin + self.diag_panel_size]
    if indices.numel() < self.diag_panel_size:
        missing = self.diag_panel_size - indices.numel()
        indices = torch.cat((indices, self.diag_code_order[:missing]))

    panel = self.seed_sampler.pool[indices]
    panel = panel.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat((panel, panel), dim=1), panel_id


def _probe_rollout(self, z_probe: torch.Tensor) -> np.ndarray:
    batch_size, rollout_size, _ = z_probe.shape
    problem_size = self.env.problem_size
    batch_index = torch.arange(batch_size, device=self.device)[:, None].expand(
        batch_size, rollout_size
    )
    rollout_index = torch.arange(rollout_size, device=self.device)[None, :].expand(
        batch_size, rollout_size
    )
    mask = torch.zeros(
        batch_size, rollout_size, problem_size + 1, device=self.device
    )
    mask[:, :, 0] = float("-inf")
    state = StepState(
        BATCH_IDX=batch_index,
        ROLLOUT_IDX=rollout_index,
        selected_count=0,
        current_node=None,
        ninf_mask=mask,
    )
    selected_list = []

    self.model.decoder.set_kv(self.model.encoded_nodes, z_probe)
    for _ in range(self.env.num_nodes_to_remove):
        with torch.amp.autocast(device_type=self.device.type):
            selected, _, _ = self.model(state)
        selected_list.append(selected)
        state.selected_count += 1
        state.current_node = selected
        state.ninf_mask[batch_index, rollout_index, selected] = float("-inf")

    return torch.stack(selected_list, dim=2).cpu().numpy()


def _run_shadow_probe(self, batch_size: int) -> Dict[str, float]:
    z_probe, panel_id = _make_probe_codes(self, batch_size)
    old_costs = torch.tensor(
        self.env.instanceSet.costs, device=self.device, dtype=torch.float32
    )
    fork_devices = [self.device.index] if self.device.type == "cuda" else []

    _synchronize(self.device)
    decoder_start = time.perf_counter()
    with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(self.diag_probe_seed + self.diag_probe_calls)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(
                self.diag_probe_seed + self.diag_probe_calls
            )
        selected_nodes = _probe_rollout(self, z_probe)
    _synchronize(self.device)
    decoder_ms = 1000.0 * (time.perf_counter() - decoder_start)

    repair_start = time.perf_counter()
    candidate_costs = self.env.instanceSet.evaluate_remove_recreate(
        selected_nodes,
        self.env_params["recreate_n"],
        beta=self.env_params["beta"],
        insert_in_new_tours_only=self.env_params["insert_in_new_tours_only"],
    )
    repair_ms = 1000.0 * (time.perf_counter() - repair_start)

    all_costs = torch.tensor(
        np.asarray(candidate_costs), device=self.device, dtype=torch.float32
    )
    first = all_costs[:, : self.diag_panel_size]
    second = all_costs[:, self.diag_panel_size :]
    metrics = repeated_tournament_metrics(
        first,
        second,
        old_costs,
        self.diag_subset_sizes,
        improvement_eps=self.diag_improvement_eps,
    )
    metrics.update(
        {
            "probe_panel_id": float(panel_id),
            "probe_decoder_ms": decoder_ms,
            "probe_repair_ms": repair_ms,
        }
    )
    return metrics


def _write_csv_row(path: str, row: Dict[str, Any]) -> None:
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _diagnostic_train_one_batch(self, batch_size: int) -> Dict[str, float]:
    """Original reward-only update plus side-effect-free diagnostics."""
    self.model.train()
    self.diag_train_calls += 1
    do_probe = (
        self.diag_enabled
        and self.diag_probe_enabled
        and self.diag_train_calls % self.diag_probe_interval == 0
    )

    state = self.env.reset()
    reset_state = self.env.get_model_input(self.device)
    z = self.seed_sampler.sample(batch_size, self.rollout_size)

    if do_probe:
        _synchronize(self.device)
        encoder_start = time.perf_counter()
    with torch.amp.autocast(device_type=self.device.type):
        self.model.pre_forward(reset_state, z)
    if do_probe:
        _synchronize(self.device)
        encoder_ms = 1000.0 * (time.perf_counter() - encoder_start)
        rollout_start = time.perf_counter()

    step_probs, first_step_probs = self._perform_rollout(state)
    selected_nodes = self.env.selected_node_list.cpu().numpy()

    if do_probe:
        _synchronize(self.device)
        rollout_ms = 1000.0 * (time.perf_counter() - rollout_start)
        probe_metrics = _run_shadow_probe(self, batch_size)
        baseline_repair_start = time.perf_counter()
    else:
        probe_metrics = {}

    reward = self._compute_reward(selected_nodes)
    if do_probe:
        baseline_repair_ms = 1000.0 * (
            time.perf_counter() - baseline_repair_start
        )

    dpp_diagnostics = {
        "logdet": reward.new_zeros(()),
        "marginal": reward.new_zeros(()),
        "similarity": reward.new_zeros(()),
        "gradient_scale": reward.new_zeros(()),
    }
    if self.dpp_enabled:
        loss, dpp_diagnostics = self._compute_dpp_policy_loss(
            reward,
            step_probs,
            first_step_probs,
            reset_state.tour_index,
        )
    else:
        loss = self._compute_policy_loss(reward, step_probs, batch_size)

    self.scaler.scale(loss).backward()
    max_reward, _ = reward.max(dim=1)
    score_mean = max_reward.float().mean()
    improved = (max_reward > 1e-5).sum().item()

    metrics = {
        "score": score_mean.item(),
        "loss": loss.item(),
        "reward": reward.mean().item(),
        "improved_frac": improved / batch_size,
        "positive_rollout_frac": (reward > 1e-5).float().mean().item(),
        "dpp_logdet": dpp_diagnostics["logdet"].item(),
        "dpp_marginal": dpp_diagnostics["marginal"].item(),
        "dpp_similarity": dpp_diagnostics["similarity"].item(),
        "dpp_gradient_scale": dpp_diagnostics["gradient_scale"].item(),
    }
    if self.diag_enabled:
        metrics.update(compression_curve(reward, self.diag_subset_sizes))
    if do_probe:
        metrics.update(probe_metrics)
        metrics.update(
            {
                "diag_encoder_ms": encoder_ms,
                "diag_rollout_ms": rollout_ms,
                "diag_baseline_repair_ms": baseline_repair_ms,
            }
        )
        probe_row: Dict[str, Any] = {
            "epoch": self.diag_epoch,
            "train_call": self.diag_train_calls,
            "instance_iteration": (
                (self.diag_train_calls - 1)
                % int(self.env_params["iterations_per_instance"])
            ),
        }
        probe_row.update(probe_metrics)
        probe_row.update(
            {
                "diag_encoder_ms": encoder_ms,
                "diag_rollout_ms": rollout_ms,
                "diag_baseline_repair_ms": baseline_repair_ms,
            }
        )
        _write_csv_row(self.diag_probe_csv, probe_row)
        self.diag_probe_calls += 1
    return metrics


def install() -> None:
    """Install instrumentation into the classes used by the baseline entrypoint."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import instance_set as instance_set_module
    from . import trainer as trainer_module

    instance_set_module.worker = _diagnostic_worker
    instance_set_module.InstanceSet.evaluate_remove_recreate = (
        _evaluate_remove_recreate
    )

    trainer_class = trainer_module.Trainer
    original_init = trainer_class.__init__
    original_train_epoch = trainer_class._train_one_epoch
    original_update_metrics = trainer_class._update_metrics
    original_log_epoch = trainer_class._log_epoch_summary
    original_log_wandb = trainer_class._log_to_wandb

    def diagnostic_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _setup_diagnostics(self)

    def diagnostic_train_epoch(self, epoch: int):
        self.diag_epoch = epoch
        return original_train_epoch(self, epoch)

    def diagnostic_update_metrics(self, metrics, batch_metrics):
        for name in batch_metrics:
            if name not in metrics:
                metrics[name] = trainer_module.AverageMeter()
        return original_update_metrics(self, metrics, batch_metrics)

    def diagnostic_log_epoch(self, epoch, processed, total, metrics, final_costs):
        original_log_epoch(self, epoch, processed, total, metrics, final_costs)
        if not self.diag_enabled:
            return

        row = {"epoch": epoch}
        for name, meter in metrics.items():
            if name.startswith(("diag_", "probe_")) and meter.count > 0:
                row[name] = meter.avg
        full_gain = row.get("diag_gain_full", 0.0)
        for size in self.diag_subset_sizes:
            gain = row.get(f"diag_gain_k{size}")
            if gain is not None:
                row[f"diag_epoch_retention_k{size}"] = (
                    gain / full_gain if full_gain > 0 else 1.0
                )
        _write_csv_row(self.diag_epoch_csv, row)

        key_size = min(self.diag_panel_size, max(self.diag_subset_sizes))
        self.logger.info(
            "Feasibility | Ret@%d: %.4f | Margin: %.4f | "
            "BWS@%d: %.4f | UsefulRepeat: %.4f | K_eff: %.2f | "
            "Enc/DecProbe/RepairProbe ms: %.1f/%.1f/%.1f",
            key_size,
            row.get(f"diag_epoch_retention_k{key_size}", float("nan")),
            row.get("diag_relative_margin", float("nan")),
            key_size,
            row.get(f"probe_k{key_size}_bws", float("nan")),
            row.get(f"probe_k{key_size}_useful_repeat_frac", float("nan")),
            row.get(f"probe_k{key_size}_effective_winners", float("nan")),
            row.get("diag_encoder_ms", float("nan")),
            row.get("probe_decoder_ms", float("nan")),
            row.get("probe_repair_ms", float("nan")),
        )

    def diagnostic_log_wandb(self, epoch, metrics, final_costs, duration):
        original_log_wandb(self, epoch, metrics, final_costs, duration)
        if not self.diag_enabled:
            return
        data = {
            f"feasibility/{name}": meter.avg
            for name, meter in metrics.items()
            if name.startswith(("diag_", "probe_")) and meter.count > 0
        }
        if data:
            trainer_module.wandb.log(step=epoch, data=data)

    trainer_class.__init__ = diagnostic_init
    trainer_class._train_one_epoch = diagnostic_train_epoch
    trainer_class._train_one_batch = _diagnostic_train_one_batch
    trainer_class._update_metrics = diagnostic_update_metrics
    trainer_class._log_epoch_summary = diagnostic_log_epoch
    trainer_class._log_to_wandb = diagnostic_log_wandb
    _INSTALLED = True
