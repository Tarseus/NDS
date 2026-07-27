"""Evaluate fixed versus per-step-resampled latent code panels."""

import argparse
import logging
from pathlib import Path
from typing import List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src import create_logger
from src.model import Model
from src.reproducibility import seed_everything
from src.validator import Validator


def main(cfg: DictConfig) -> None:
    seed = int(cfg.trainer_params.get("valid_seed", 0))
    seed_everything(seed, deterministic=True)
    OmegaConf.update(cfg, "env_params.random_seed", seed, force_add=True)
    OmegaConf.update(
        cfg, "trainer_params.valid_greedy_diversity_enable", False, force_add=True
    )
    OmegaConf.update(cfg, "logger_params.wandb.enable", False, force_add=True)
    create_logger(cfg.logger_params)

    use_cuda = bool(cfg.trainer_params["use_cuda"])
    if use_cuda:
        cuda_device_num = int(cfg.trainer_params["cuda_device_num"])
        torch.cuda.set_device(cuda_device_num)
        device = torch.device("cuda", cuda_device_num)
    else:
        device = torch.device("cpu")

    model_load = cfg.trainer_params["model_load"]
    checkpoint_path = Path(
        "{path}/checkpoint-{epoch}.pt".format(**model_load)
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model_params = checkpoint.get("model_params", cfg.model_params)
    model = Model(**model_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger = logging.getLogger("root")
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Loaded baseline checkpoint from %s", checkpoint_path)
    validator = Validator(
        device=device,
        env_params=cfg.env_params,
        trainer_params=cfg.trainer_params,
        model_params=model_params,
        logger_params=cfg.logger_params,
    )
    validator.run(model, model, int(model_load.get("epoch", 0)))


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Config relative to configs/train")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    config_file = Path("configs") / "train" / args.config
    with hydra.initialize(
        config_path=str(config_file.parent), version_base=None
    ):
        config = hydra.compose(
            config_name=config_file.name, overrides=args.overrides
        )
    main(config)
