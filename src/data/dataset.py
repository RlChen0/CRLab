from torch.utils import data
from .transforms.build import build_transform
from .datasets.build import build_dataset
from yacs.config import CfgNode
from torch import nn


def build_dataloader(dataset_cfg: CfgNode, mode='train'):
    if mode == 'train' or 'valid':
        batch_size = dataset_cfg.TRAIN_BATCH_SIZE
        shuffle = True

    elif mode == 'test':
        batch_size = dataset_cfg.TEST_BATCH_SIZE
        shuffle = False

    transforms = build_transform(dataset_cfg.AUGMENTATION, mode=mode)
    dataset = build_dataset(dataset_cfg, mode=mode, transform=transforms)
    num_workers = dataset_cfg.LOADER_WORKERS

    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader
