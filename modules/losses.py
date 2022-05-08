import torch
from torch import nn


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


class PoissonLoss(nn.Module):
    def __init__(self,):
        super(PoissonLoss, self).__init__()

    def forward(self, pred, target):
        return (pred - target * log(pred)).mean()