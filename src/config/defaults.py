import os
import warnings

from dotenv import find_dotenv, load_dotenv
from yacs.config import CfgNode as CN
from pathlib import Path

###################################################
# Config definitions
###################################################
_C = CN()
cfg = _C
###################################################
# Dataset
###################################################
_C.DATASET = CN()
_C.DATASET.IMG_SIZE = (224, 224)
_C.DATASET.DIR = 'F:/CRLab/data/classify-leaves/'
_C.DATASET.NAME = 'CfImageDataset'
_C.DATASET.TRAIN_DATA_ANNOTATION = 'F:/CRLab/data/classify-leave/train.csv'
_C.DATASET.TRAIN_BATCH_SIZE = 32
_C.DATASET.VALID_RATE = 0.1
_C.DATASET.TSET_DATA_ANNOTATION = 'F:/CRLab/data/classify-leave/test.csv'
_C.DATASET.TEST_BATCH_SIZE = 64

_C.DATASET.AUGMENTATION = CN()
_C.DATASET.AUGMENTATION.NAME = 'HorizonVerticalFlip'
_C.DATASET.AUGMENTATION.HORIZONTAL_FLIP_PROB = 0.5
_C.DATASET.AUGMENTATION.VERTICAL_FLIP_PROB = 0.5

_C.DATASET.LOADER_WORKERS = 1
###################################################
# MODELING
###################################################
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = 'baseline'

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'resnet18'

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = 'simple_head'
_C.MODEL.HEAD.INPUT_DIMS = 512
_C.MODEL.HEAD.OUTPUT_DIMS = 176

###################################################
# Solver
###################################################
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = CN()
_C.SOLVER.OPTIMIZER.NAME = "Adam"
_C.SOLVER.OPTIMIZER.TARGET_LR = 0.001
_C.SOLVER.OPTIMIZER.MOMENTUM = 0.9

_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.MIN_EPOCHS = 10
_C.SOLVER.PATIENCE = 20


_C.SOLVER.WARMUP_STEPS = 500
_C.SOLVER.WARMUP_METHOD = 'Linear'

_C.SOLVER.LOSS = CN()
_C.SOLVER.LOSS.NAME = 'CrossEntropyLoss'

_C.SOLVER.METRIC = CN()
_C.SOLVER.METRIC.NAME = 'Accuracy'
_C.SOLVER.METRIC.NUM_TARGETS = 10
_C.SOLVER.METRIC.SUMMARIZE = True
####################################################
_C.OUTPUT_DIR = ""


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return _C.clone()


def combine_cfgs(path_cfg: Path = None):
    """
    An internal facing routine that at combined CFG in the order provided.
    :param: path_cfg: path to path_cfg_data files
    :return: cfg_base incorporating the override.
    """
    if path_cfg is not None:
        path_cfg = Path(path_cfg)

    cfg_base = get_cfg_defaults()

    if path_cfg is not None and path_cfg.exists():
        cfg_base.merge_from_file(path_cfg.absolute())

    return cfg_base

