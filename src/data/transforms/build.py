from yacs.config import CfgNode
from torch import nn
from src.tools.registry import Registry

TRANSFORM_REGISTRY = Registry()


def build_transform(transform_cfg: CfgNode, **kwargs) -> nn.Module:
    transform_module = TRANSFORM_REGISTRY[transform_cfg.NAME](transform_cfg, **kwargs)
    return transform_module
