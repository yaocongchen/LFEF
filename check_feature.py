import torch.nn as nn
import torchvision 
import os
# import time


class check_feature(nn.Module):
    def __init__(self, in_chs):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, 1, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True)

    def forward(self, input):
        if input.shape[1] != 1:
            x = self.conv(input)
        else:
            x = input
        x = self.upsample(x)

        return x