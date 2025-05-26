from tqdm import tqdm
import os
from tqdm import tqdm, trange
from functools import partialmethod
import torch.nn as nn
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
from common.utils import log_hyperparameters, PROJECT_ROOT
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
import omegaconf
import hydra
from time import sleep

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run(cfg: DictConfig) -> None:
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    print('Instantiate SHAFT')
    shaft = hydra.utils.instantiate(cfg.SHAFT.SHAFT_Density)
    shaft.sample_with_threshold_and_type(number_to_sample=1000, save_folder='/sample_ternary_db/', bscorethreshold=0.55, reward_threshold=0.7, type='ternary')

@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="sample")
def main(cfg: omegaconf.DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()
