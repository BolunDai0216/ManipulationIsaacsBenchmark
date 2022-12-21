from pdb import set_trace

import hydra
from omegaconf import DictConfig, OmegaConf

from franka_base_env import FrankaBaseEnv


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    env = FrankaBaseEnv(cfg)


if __name__ == "__main__":
    main()
