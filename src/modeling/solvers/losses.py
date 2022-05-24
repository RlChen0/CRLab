from torch import nn
from src.modeling.solvers.build import LOSS_REGISTRY
from yacs.config import CfgNode


@LOSS_REGISTRY.register('CrossEntropyLoss')
def build_cross_entropy_loss(loss_cfg: CfgNode):
    return nn.CrossEntropyLoss()