from pdb import set_trace

import hydra
from omegaconf import DictConfig, OmegaConf

from franka_base_env import FrankaBaseEnv


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    env = FrankaBaseEnv(cfg)

    while not env.gym.query_viewer_has_closed(env.viewer):
        env.step(1)


if __name__ == "__main__":
    main()
