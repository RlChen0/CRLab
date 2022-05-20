import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import module.layers as layers
import module.utils as utils


class ConvNAC(nn.Module):
    """Convolution block with the order of normalization, activation, convolution."""
    def __init__(self, dim_in: int, dim_out=None, kernel_size=1):
        super(ConvNAC, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(dim_in),
            layers.GELU(),
            nn.Conv1d(dim_in, utils.default(dim_out, dim_in), kernel_size=kernel_size, padding=kernel_size//2)
        )

    def forward(self, x):
        return self.net(x)


class ConvTower(nn.Module):
    """convolution tower"""
    def __init__(self, dim_in, dim_out, kernel_size=1, repeat=1, divisible_by=1):
        super(ConvTower, self).__init__()
        dim_list = utils.exponential_linspace_int(dim_in, dim_out, repeat, divisible_by)
        dim_list = [dim_in, *dim_list]
        conv_block = []

        for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
            conv_block.append(nn.Sequential(
                ConvNAC(dim_in, dim_out, kernel_size=kernel_size),
                layers.Residual(ConvNAC(dim_out, dim_out, 1)),
                nn.MaxPool1d(2)
            ))

        self.conv_tower = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_tower(x)


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out=None, use_1x1_conv=False, strides=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=strides) if use_1x1_conv else nn.Identity()

        self.bn1 = nn.BatchNorm2d(channel_out)
        self.bn2 = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        inputs = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + self.conv3(inputs))


