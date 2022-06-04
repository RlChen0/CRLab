from torch import nn
from yacs.config import CfgNode
from src.tools.registry import Registry

META_ARCH_REGISTRY = Registry()


def build_model(model_cfg: CfgNode) -> nn.Module:
    """
    build model
    :param model_cfg: model config blob
    :return: model
    """
    meta_arch = model_cfg.META_ARCHITECTURE
    model = META_ARCH_REGISTRY[meta_arch.NAME](model_cfg)

    return model
