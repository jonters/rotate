'''Main entry point for running MARL algorithms.'''
import os
import sys

# Ensure the project root (parent of marl/) is on sys.path so that local
# packages like `envs` are found even after Hydra changes the working directory.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import hydra
from omegaconf import OmegaConf

from common.wandb_visualizations import Logger
from ippo import run_ippo


@hydra.main(version_base=None, config_path="configs", config_name="base_config_marl")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))
    wandb_logger = Logger(config)

    if config.algorithm["ALG"] == "ippo":
        run_ippo(config, wandb_logger)
    else:
        raise NotImplementedError(f"Algorithm {config['ALG']} not implemented.")
        
    wandb_logger.close()

if __name__ == "__main__":
    main()