import os
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path='config.yaml')
def my_app(cfg):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.training.epoch)

if __name__ == "__main__":
    my_app()