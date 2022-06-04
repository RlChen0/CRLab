import os
import warnings


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
_C.DATASET.TRAIN_DATA_ANNOTATION = 'F:/CRLab/data/classify-leaves/train.csv'
_C.DATASET.TRAIN_BATCH_SIZE = 32
_C.DATASET.VALID_RATE = 0.1
_C.DATASET.TSET_DATA_ANNOTATION = 'F:/CRLab/data/classify-leaves/test.csv'
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
_C.MODEL.META_ARCHITECTURE = CN()
_C.MODEL.META_ARCHITECTURE.NAME = 'baseline'

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'resnet18'

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = 'linear'
_C.MODEL.HEAD.INPUT_DIMS = 512
_C.MODEL.HEAD.OUTPUT_DIMS = 176

###################################################
# Solver
###################################################
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = CN()
_C.SOLVER.OPTIMIZER.NAME = "Adam"

_C.SOLVER.OPTIMIZER.ADAM = CN()
_C.SOLVER.OPTIMIZER.ADAM.TARGET_LR = 0.001

_C.SOLVER.OPTIMIZER.SGD = CN()
_C.SOLVER.OPTIMIZER.SGD.TARGET_LR = 0.001
_C.SOLVER.OPTIMIZER.SGD.MOMENTUM = 0.9

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
_C.SOLVER.CHECK_EPOCH = 1
####################################################
_C.OUTPUT_DIR = ''
_C.RESUME_PATH = ''

def get_cfg_defaults():

    return _C.clone()


def combine_cfgs(cfg_path):

    cfg_base = get_cfg_defaults()

    if cfg_path is not None and os.path.exists(cfg_path):
        cfg_base.merge_from_file(cfg_path)

    return cfg_base
