import torch
from torch import nn
from yacs.config import CfgNode
from src.modeling.backbone.build import build_backbone
from src.modeling.head.build import build_head

from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register('baseline', build_model)
def build_model(model_cfg: CfgNode) -> nn.Module:
    """
    Builds the baseline model using the CFGNode object,
    :param model_cfg: YAML based YACS configuration node.
    :return: return torch neural network module
    """
    # instantiate and return the BaselineModel using the configuration node
    return Model(model_cfg)





class Model(nn.Module):

    def __init__(self, model_cfg: CfgNode):
        super(Model, self).__init__()
        self.backbone = build_backbone(model_cfg.BACKBONE)
        self.head = build_head(model_cfg.HEAD)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
