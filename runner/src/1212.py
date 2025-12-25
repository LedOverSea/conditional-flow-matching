import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "README.md"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_pylogger(__name__)

def train(cfg: DictConfig) -> Tuple[dict, dict]:
    # 测试数据模块
    log.info(f"config: <{cfg}>")
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"data: {datamodule.data}")
    log.info(f"dim: {datamodule.dim}")
    log.info(f"label: {datamodule.labels}")
    log.info(f"system: {datamodule.system}")
    log.info(f"data.shape: {datamodule.data.shape}")

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train_1212.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)


if __name__ == "__main__":
    main()
