"""Hydra entry point for LOB-Forge training pipeline."""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the LOB-Forge training pipeline.

    This is a placeholder that prints the resolved config.
    Future phases will add real training logic for the predictor,
    generator, and executor components.
    """
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
