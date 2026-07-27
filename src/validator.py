"""Validation helpers for evaluating trained models."""

import os
from logging import getLogger
from typing import Tuple, Dict, Any

import numpy as np
import torch
import wandb

from .env import Env
from .logging_utils import (
    get_result_folder,
    TimeEstimator,
    AverageMeter,
)
from .seed_sampler import SeedVectorSampler
from .bfws import bfws_value, random_derangement
from .reproducibility import seed_everything


class Validator:
    """Runs periodic validation and aggregates performance metrics."""

    def __init__(
        self,
        device: torch.device,
        env_params: Dict[str, Any],
        trainer_params: Dict[str, Any],
        model_params: Dict[str, Any],
        logger_params: Dict[str, Any],
        fixed_code_indices: torch.Tensor = None,
    ):
        """Initialize validator with configuration parameters."""
        self.env_params = env_params
        self.trainer_params = trainer_params
        self.device = device

        # Setup logging
        self.logger = getLogger(name="validator")
        self.result_folder = get_result_folder()
        self.time_estimator = TimeEstimator()

        # Setup environment
        self.env = Env(**self.env_params)

        # Seed vector sampler
        self.seed_sampler = SeedVectorSampler(model_params["z_dim"], device)
        bfws_params = self.trainer_params.get("bfws", {})
        self.bfws_enabled = bool(bfws_params.get("enabled", False))
        self.bfws_num_codes = int(bfws_params.get("num_codes", 8))
        self.bfws_resample_codes_each_step = bool(
            bfws_params.get("resample_codes_each_step", False)
        )
        self.valid_seed = self.trainer_params.get("valid_seed")
        if self.bfws_enabled:
            if fixed_code_indices is None:
                fixed_code_indices = self.seed_sampler.fixed_indices(
                    self.bfws_num_codes,
                    int(bfws_params.get("code_seed", 20260726)),
                )
            self.bfws_code_indices = fixed_code_indices.to(device)
        else:
            self.bfws_code_indices = None

        # Experiment tracking
        self.use_wandb = logger_params["wandb"]["enable"]

    def run(self, model, frozen_model, training_epoch: int) -> float:
        """Run full validation and return augmented score."""
        if self.valid_seed is not None:
            seed_everything(int(self.valid_seed))
        self.time_estimator.reset()

        # Initialize metrics
        metrics = {
            "score": AverageMeter(),
            "aug_score": AverageMeter(),
            "diversity_overlap": AverageMeter(),
            "unique_rollouts": AverageMeter(),
            "portfolio_j_k": AverageMeter(),
            "bfws": AverageMeter(),
            "bfws_same": AverageMeter(),
            "bfws_mismatched": AverageMeter(),
            "bfws_effective_codes": AverageMeter(),
            "bfws_joint_failure": AverageMeter(),
            "portfolio_accepted_frac": AverageMeter(),
        }

        # Load validation dataset if specified
        self._load_validation_data()

        # Determine augmentation factor
        aug_factor = self._get_augmentation_factor()

        # Run validation
        self.logger.info("=" * 80)
        self.logger.info("Validation")
        self.logger.info("=" * 80)
        self._validate_all_episodes(model, frozen_model, aug_factor, metrics)

        # Log results
        self._log_validation_results(metrics)

        # Additional greedy evaluation
        greedy_diversity = self._evaluate_greedy_diversity(
            model, frozen_model, aug_factor
        )

        # Log to W&B if enabled
        if self.use_wandb:
            self._log_to_wandb(training_epoch, metrics, greedy_diversity)

        return metrics["aug_score"].avg

    def _load_validation_data(self) -> None:
        """Load validation dataset if specified in config."""
        if not self.trainer_params["valid_data_load"]["enable"]:
            return

        filename = self.trainer_params["valid_data_load"]["filename"]
        extension = os.path.splitext(filename)[1]

        if extension == ".pkl":
            self.env.load_problem_dataset_pkl(
                filename, self.trainer_params["valid_episodes"]
            )
        elif extension == ".pt":
            self.env.load_problem_dataset_pt(filename, self.device)
        else:
            raise ValueError(f"Unsupported dataset format: {extension}")

    def _get_augmentation_factor(self) -> int:
        """Get augmentation factor from config."""
        if self.trainer_params["valid_augmentation_enable"]:
            return self.trainer_params["valid_aug_factor"]
        return 1

    def _validate_all_episodes(
        self, model, frozen_model, aug_factor: int, metrics: Dict[str, AverageMeter]
    ) -> np.ndarray:
        """Run validation across all episodes and return cost logs."""
        validate_num_episode = self.trainer_params["valid_episodes"]
        episode = 0

        while episode < validate_num_episode:
            remaining = validate_num_episode - episode
            batch_size = min(self.trainer_params["valid_batch_size"], remaining)

            # Validate one batch
            (
                score,
                aug_score,
                logs_episode,
                diversity,
                portfolio_stats,
            ) = self._validate_one_batch(
                model,
                frozen_model,
                batch_size,
                self.trainer_params["valid_iterations"],
                aug_factor=aug_factor,
            )

            # Update metrics
            metrics["score"].update(score, batch_size)
            metrics["aug_score"].update(aug_score, batch_size)
            metrics["diversity_overlap"].update(diversity[0], batch_size)
            metrics["unique_rollouts"].update(diversity[1], batch_size)
            for name, value in portfolio_stats.items():
                metrics[name].update(value, batch_size)

            episode += batch_size

            # Log progress
            self._log_episode_progress(episode, validate_num_episode, metrics)

    def _validate_one_batch(
        self,
        model,
        frozen_model,
        batch_size: int,
        nb_iterations: int,
        aug_factor: int = 1,
    ) -> Tuple:
        """Validate one batch and return scores, diversity and portfolio stats."""
        rollout_size = (
            2 * self.bfws_num_codes
            if self.bfws_enabled
            else self.trainer_params["valid_rollout_size"]
        )
        z_dim = model.model_params["z_dim"]
        recreate_n = self.env_params["recreate_n"]
        beta = self.env_params["beta"]
        insert_in_new_tours_only = self.env_params["insert_in_new_tours_only"]
        aug_batch_size = batch_size * aug_factor

        logs = np.zeros((batch_size, nb_iterations))
        portfolio_history = []

        model.eval()
        with torch.no_grad():
            self.env.init_instances(batch_size, rollout_size, self.device, aug_factor)

            for i in range(nb_iterations):
                if self.bfws_enabled:
                    tie_priorities = torch.rand(
                        aug_batch_size,
                        2,
                        self.bfws_num_codes,
                        device=self.device,
                    )
                    acceptance_uniforms = torch.rand(
                        aug_batch_size, device=self.device
                    )
                    derangement = (
                        random_derangement(aug_batch_size, self.device)
                        if aug_batch_size > 1
                        else torch.zeros(1, dtype=torch.long, device=self.device)
                    )

                # Reset and get state
                state = self.env.reset()
                reset_state = self.env.get_model_input(self.device)

                # Sample latent vectors
                if self.bfws_enabled:
                    if self.bfws_resample_codes_each_step:
                        z = self.seed_sampler.sample_shared_panel(
                            aug_batch_size,
                            self.bfws_num_codes,
                            replicas=2,
                        )
                    else:
                        z = self.seed_sampler.repeat_fixed(
                            self.bfws_code_indices,
                            aug_batch_size,
                            replicas=2,
                        )
                else:
                    z = self.seed_sampler.sample(aug_batch_size, rollout_size)

                # Forward pass
                with torch.amp.autocast(device_type=self.device.type):
                    model.pre_forward(reset_state, z)

                # Rollout
                done = False
                while not done:
                    with torch.amp.autocast(device_type=self.device.type):
                        selected, _, _ = model(state)
                    state, done = self.env.step(selected)

                # Apply repair
                selected_nodes = self.env.selected_node_list.cpu().numpy()
                if self.bfws_enabled:
                    old_costs = torch.as_tensor(
                        self.env.instanceSet.costs,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    if torch.any(old_costs <= 0):
                        raise ValueError(
                            "BFWS relative rewards require positive incumbent costs"
                        )
                    portfolio = self.env.instanceSet.remove_recreate_portfolio(
                        selected_nodes,
                        self.bfws_num_codes,
                        recreate_n,
                        T=0,
                        beta=beta,
                        insert_in_new_tours_only=insert_in_new_tours_only,
                        tie_priorities=tie_priorities.cpu().numpy(),
                        acceptance_uniforms=acceptance_uniforms.cpu().numpy(),
                    )
                    candidate_costs = torch.as_tensor(
                        portfolio.candidate_costs,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    rewards = (
                        old_costs[:, None, None] - candidate_costs
                    ) / old_costs[:, None, None]
                    winners = torch.as_tensor(
                        portfolio.winners,
                        dtype=torch.long,
                        device=self.device,
                    )
                    value, same, mismatched = bfws_value(
                        winners, derangement, self.bfws_num_codes
                    )
                    frequency = torch.bincount(
                        winners.reshape(-1), minlength=self.bfws_num_codes
                    ).float()
                    frequency = frequency / frequency.sum()
                    best_reward = rewards.max(dim=2).values
                    portfolio_history.append(
                        {
                            "portfolio_j_k": best_reward.mean().item(),
                            "bfws": value.item(),
                            "bfws_same": same.item(),
                            "bfws_mismatched": mismatched.item(),
                            "bfws_effective_codes": frequency.square()
                            .sum()
                            .reciprocal()
                            .item(),
                            "bfws_joint_failure": (best_reward <= 0)
                            .float()
                            .mean()
                            .item(),
                            "portfolio_accepted_frac": float(
                                portfolio.accepted.mean()
                            ),
                        }
                    )
                else:
                    self.env.instanceSet.remove_recreate(
                        selected_nodes,
                        recreate_n,
                        "allImp",
                        T=0,
                        beta=beta,
                        insert_in_new_tours_only=insert_in_new_tours_only,
                    )

                # Log costs (best across augmentations)
                costs = np.array(self.env.instanceSet.costs)
                logs[:, i] = costs.reshape(aug_factor, -1).min(axis=0)

            # Compute diversity metrics
            diversity_nodes = (
                selected_nodes[:, : self.bfws_num_codes]
                if self.bfws_enabled
                else selected_nodes
            )
            diversity = self._calculate_diversity(diversity_nodes)

            # Compute final scores
            aug_costs = np.array(self.env.instanceSet.costs).reshape(aug_factor, -1)
            no_aug_score = np.mean(aug_costs[0])
            aug_score = np.mean(aug_costs.min(axis=0))

            if portfolio_history:
                portfolio_stats = {
                    name: float(np.mean([row[name] for row in portfolio_history]))
                    for name in portfolio_history[0]
                }
            else:
                portfolio_stats = {
                    "portfolio_j_k": 0.0,
                    "bfws": 0.0,
                    "bfws_same": 0.0,
                    "bfws_mismatched": 0.0,
                    "bfws_effective_codes": 0.0,
                    "bfws_joint_failure": 0.0,
                    "portfolio_accepted_frac": 0.0,
                }

            return no_aug_score, aug_score, logs, diversity, portfolio_stats

    def _calculate_diversity(self, selected_nodes: np.ndarray) -> Tuple[float, float]:
        """Calculate diversity metrics: overlap score and unique rollout ratio."""
        batch_size, rollout_size, num_nodes = selected_nodes.shape

        overlap_scores = []
        unique_ratios = []

        for i in range(batch_size):
            # Metric 1: Node overlap across rollouts
            total_overlap = 0
            for j in range(rollout_size):
                for node in selected_nodes[i, j]:
                    # Count how many other rollouts contain this node
                    overlap = (selected_nodes[i] == node).any(axis=1).sum() - 1
                    total_overlap += overlap

            # Normalize by total possible overlaps
            max_overlap = (rollout_size - 1) * num_nodes * rollout_size
            overlap_score = total_overlap / max_overlap if max_overlap > 0 else 0
            overlap_scores.append(overlap_score)

            # Metric 2: Ratio of unique rollouts
            sorted_selections = np.sort(selected_nodes[i], axis=1)
            unique_selections = np.unique(sorted_selections, axis=0)
            unique_ratio = unique_selections.shape[0] / rollout_size
            unique_ratios.append(unique_ratio)

        return np.mean(overlap_scores), np.mean(unique_ratios)

    def _evaluate_greedy_diversity(
        self, model, frozen_model, aug_factor: int
    ) -> Tuple[float, float]:
        """Evaluate diversity with greedy (argmax) decoding."""
        original_eval_type = model.model_params["eval_type"]
        model.model_params["eval_type"] = "argmax"
        self.env.problem.saved_index = 0

        _, _, _, greedy_diversity, _ = self._validate_one_batch(
            model,
            frozen_model,
            self.trainer_params["valid_batch_size"],
            self.trainer_params["valid_iterations"],
            aug_factor=aug_factor,
        )

        model.model_params["eval_type"] = original_eval_type
        return greedy_diversity

    def _log_episode_progress(
        self, episode: int, total: int, metrics: Dict[str, AverageMeter]
    ) -> None:
        """Log progress for current episode."""
        elapsed, remaining = self.time_estimator.get_est_string(episode, total)
        self.logger.info(
            f"Episode {episode:3d}/{total:3d}  |  "
            f"Elapsed: {elapsed}  |  Remain: {remaining}  |  "
            f"Score: {metrics['score'].avg:7.2f}  |  Aug Score: {metrics['aug_score'].avg:7.2f}"
        )

    def _log_validation_results(self, metrics: Dict[str, AverageMeter]) -> None:
        """Log final validation results."""
        self.logger.info("=" * 80)
        self.logger.info("Validation Complete")
        self.logger.info("=" * 80)
        self.logger.info(f"No-Aug Score:    {metrics['score'].avg:7.3f}")
        self.logger.info(f"Aug Score:       {metrics['aug_score'].avg:7.3f}")
        self.logger.info(f"Diversity Score: {metrics['diversity_overlap'].avg:7.4f}")
        self.logger.info(f"Unique Rollouts: {metrics['unique_rollouts'].avg:7.4f}")
        if self.bfws_enabled:
            self.logger.info(f"Portfolio J_K:   {metrics['portfolio_j_k'].avg:7.5f}")
            self.logger.info(f"BFWS:            {metrics['bfws'].avg:7.4f}")
            self.logger.info(
                f"Winner Agreement: {metrics['bfws_same'].avg:7.4f} same / "
                f"{metrics['bfws_mismatched'].avg:7.4f} mismatched"
            )
            self.logger.info(
                f"Effective Codes:  {metrics['bfws_effective_codes'].avg:7.3f}"
            )
        self.logger.info("=" * 80)

    def _log_to_wandb(
        self,
        training_epoch: int,
        metrics: Dict[str, AverageMeter],
        greedy_diversity: Tuple[float, float],
    ) -> None:
        """Log validation metrics to Weights & Biases."""
        wandb.log(
            step=training_epoch,
            data={
                "val/no_aug_score": metrics["score"].avg,
                "val/aug_score": metrics["aug_score"].avg,
                "val/diversity_score": metrics["diversity_overlap"].avg,
                "val/unique_rollouts": metrics["unique_rollouts"].avg,
                "val/greedy_unique_rollout": greedy_diversity[1],
                "val/portfolio_j_k": metrics["portfolio_j_k"].avg,
                "val/bfws": metrics["bfws"].avg,
                "val/bfws_same": metrics["bfws_same"].avg,
                "val/bfws_mismatched": metrics["bfws_mismatched"].avg,
                "val/bfws_effective_codes": metrics[
                    "bfws_effective_codes"
                ].avg,
                "val/bfws_joint_failure": metrics[
                    "bfws_joint_failure"
                ].avg,
            },
        )
