import torch
from torch import nn, einsum
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class Attention(nn.Module):
    def __init__(self, dim, pool_size=2):
        super(Attention, self).__init__()
        self.pool_size = pool_size
        self.pool_fn = 