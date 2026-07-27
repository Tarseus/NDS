"""Train a fixed-code NDS portfolio with optional BFWS constraint."""

import argparse
import logging
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from src import Trainer, create_logger
from src.reproducibility import seed_everything


def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("root")
    if not bool(cfg.trainer_params.get("bfws", {}).get("enabled", False)):
        raise ValueError("train_bfws.py requires trainer_params.bfws.enabled=true")
    if bool(cfg.trainer_params.get("dpp_objective", {}).get("enabled", False)):
        raise ValueError("BFWS cannot be combined with the legacy DPP objective")

    seed_everything(int(cfg.trainer_params.get("seed", 0)))
    create_logger(cfg.logger_params)
    logger.info("Starting fixed-code PortfolioStep training with BFWS")
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))
    trainer = Trainer(
        env_params=cfg.env_params,
        model_params=cfg.model_params,
        optimizer_params=cfg.optimizer_params,
        trainer_params=cfg.trainer_params,
        logger_params=cfg.logger_params,
    )
    trainer.run()


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BFWS-NDS")
    parser.add_argument("config", type=str, help="Config relative to configs/train")
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    config_file = Path("configs") / "train" / args.config
    with hydra.initialize(config_path=str(config_file.parent), version_base=None):
        config = hydra.compose(config_name=config_file.name, overrides=args.overrides)
    main(config)
