from yacs.config import CfgNode
from torch import nn
from src.tools.registry import Registry
import torch
from torch.optim.optimizer import Optimizer
LOSS_REGISTRY = Registry()


def build_loss(loss_cfg: CfgNode, **kwargs) -> nn.Module:
    loss_module = LOSS_REGISTRY[loss_cfg.NAME](loss_cfg, **kwargs)
    return loss_module


METRIC_REGISTRY = Registry()


def build_metric(metric_cfg: CfgNode, **kwargs) -> nn.Module:
    metric_module = METRIC_REGISTRY[metric_cfg.NAME](metric_cfg, **kwargs)
    return metric_module


def build_optimizer(model: torch.nn.Module, opti_cfg: CfgNode) -> Optimizer:
    parameters = model.parameters()
    opti_type = opti_cfg.NAME
    if opti_type == 'Adam' or 'adam':
        adam_cfg = opti_cfg.ADAM
        optimizer = torch.optim.Adam(parameters, lr=adam_cfg.TARGET_LR)
    elif opti_type == 'SGD' or 'sgd':
        sgd_cfg = opti_cfg.SGD
        optimizer = torch.optim.SGD(parameters, lr=sgd_cfg.TARGET_LR, momentum=sgd_cfg.MOMENTUM)
    else:
        raise Exception('invalid optimizer, available choices adam/sgd')
    return optimizer
