from torch import nn
from yacs.config import CfgNode
from .build import HEAD_REGISTRY
from src.tools.helper import cnl_init


@HEAD_REGISTRY.register('linear')
def build_simple_pred_head(head_cfg: CfgNode) -> nn.Module:
    return SimplePredictionHead(head_cfg)


class SimplePredictionHead(nn.Module):

    def __init__(self, head_cfg: CfgNode):
        super(SimplePredictionHead, self).__init__()

        input_dims = head_cfg.INPUT_DIMS
        output_dims = head_cfg.OUTPUT_DIMS
        self.head = nn.Linear(input_dims, output_dims)
        self.head.apply(cnl_init)

    def forward(self, x):
        return self.head(x)