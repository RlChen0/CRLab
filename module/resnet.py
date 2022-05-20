import torch
from torch import nn
import torch.nn.functional as F
from module import blocks as block
import module.blocks as block
import module.utils as utils


def resnet_block(channel_in, channel_out, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(block.ResBlock(channel_in, channel_out, use_1x1_conv=True, strides=2))
        else:
            blk.append(block.ResBlock(channel_out, channel_out))
    return blk


class ResNet(nn.Module):
    def __init__(self, start, num_class, repeat):
        super(ResNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, start, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(start), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        channels_list = utils.base_linspace_int(start, 2, repeat)
        blk = [nn.Sequential(*resnet_block(start, start, 2, first_block=True)), ]
        for c_i, c_out in zip(channels_list[:-1], channels_list[1:]):
            blk.append(nn.Sequential(*resnet_block(c_i, c_out, 2)))

        self.trunk = nn.Sequential(*blk,
                                   nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Flatten())
        self.head = nn.Linear(channels_list[-1], num_class)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        x = self.head(x)
        return x
