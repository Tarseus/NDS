"""Reward-only NDS training entrypoint with feasibility instrumentation."""

import argparse
import logging
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from src import Trainer, create_logger
from src.baseline_feasibility import install


install()


def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("root")
    logger.debug("Resolved configuration as YAML:\n%s", OmegaConf.to_yaml(cfg))

    if bool(cfg.trainer_params.get("dpp_objective", {}).get("enabled", False)):
        raise ValueError(
            "Baseline feasibility entrypoint requires "
            "trainer_params.dpp_objective.enabled=false"
        )

    create_logger(cfg.logger_params)
    logger.info(
        "Starting reward-only baseline with passive compression and "
        "non-committing repeated-tournament diagnostics"
    )
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
    parser = argparse.ArgumentParser(
        description="Train reward-only NDS with feasibility diagnostics"
    )
    parser.add_argument(
        "config", type=str, help="Config file relative to configs/train"
    )
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
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
