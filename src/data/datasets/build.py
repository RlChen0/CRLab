from yacs.config import CfgNode
from torch import nn
from src.tools.registry import Registry
from torch.utils.data import Dataset
DATASET_REGISTRY = Registry()


def build_dataset(dataset_cfg: CfgNode, **kwargs) -> Dataset:
    dataset_module = DATASET_REGISTRY[dataset_cfg.NAME](dataset_cfg, **kwargs)
    return dataset_module
