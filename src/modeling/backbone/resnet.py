import torch
from torch import nn
import torch.nn.functional as F
from src.tools.helper import base_linspace_int
from src.tools.helper import cnl_init
from .build import BACKBONE_REGISTRY
from yacs.config import CfgNode
import numpy as np


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


def resnet_block(channel_in, channel_out, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(ResBlock(channel_in, channel_out, use_1x1_conv=True, strides=2))
        else:
            blk.append(ResBlock(channel_out, channel_out))
    return blk


class ResNet(nn.Module):
    def __init__(self, start, repeat):
        super(ResNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, start, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(start), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        channels_list = base_linspace_int(start, 2, repeat)
        blk = [nn.Sequential(*resnet_block(start, start, 2, first_block=True)), ]
        for c_i, c_out in zip(channels_list[:-1], channels_list[1:]):
            blk.append(nn.Sequential(*resnet_block(c_i, c_out, 2)))

        self.trunk = nn.Sequential(*blk,
                                   nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Flatten())

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        return x


@BACKBONE_REGISTRY.register('resnet18')
def build_resnet18(backbone_cfg: CfgNode) -> nn.Module:
    model = ResNet(64, 4)
    model.apply(cnl_init)
    return model
