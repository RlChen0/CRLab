import torchvision.transforms as t
from .build import TRANSFORM_REGISTRY
from yacs.config import CfgNode


@TRANSFORM_REGISTRY.register('HorizonVerticalFlip')
def build_horizon_vertical(transform_cfg: CfgNode, mode='train'):
    if mode == 'train':
        transform = t.Compose([
            t.RandomHorizontalFlip(p=transform_cfg.HORIZONTAL_FLIP_PROB),
            t.RandomVerticalFlip(p=transform_cfg.VERTICAL_FLIP_PROB)
        ])
    else:
        transform = None
    return transform
