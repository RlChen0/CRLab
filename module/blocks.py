import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# constants
SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896


