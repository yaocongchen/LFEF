###########################################################################
# Created by: Yao-Cong,Chen
# Email: yaocongchen@outlook.com
# Copyright (c) 2024 Yao-Cong,Chen. All rights reserved.
###########################################################################
import torch
import torch.nn as nn
from models import base_blocks

class DSEM(nn.Module):
    """
    The down-sampling block in the SRDEM
    """

    def __init__(self, nIn, nOut, dilation_rate=3, reduction=16):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            dilation_rate: the rate of dilated convolution
            reduction: the reduction ratio in the F_glo
        """
        super().__init__()
        self.conv1x1 = base_blocks.ConvINReLU(nIn, nOut, 3, 2)  #  size/2, channel: nIn--->nOut
        
        self.F_loc = base_blocks.ChannelWiseDilatedConv(nOut, nOut, 3, 1, 1)  # local feature
        self.F_sur = base_blocks.ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)

        self.in_relu =base_blocks.INReLU(2 * nOut)
        self.reduce =base_blocks.Conv(2 * nOut, nOut, 1, 1)  # reduce dimension: 2*nOut--->nOut

        self.F_glo = base_blocks.FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)  #  the joint feature

        joi_feat = self.in_relu(joi_feat)
        joi_feat = self.reduce(joi_feat)  # channel= nOut

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
        
        return output


class DEM(nn.Module):
    """
    The basic block in the SRDEM
    """
    
    def __init__(self, nIn, nOut, dilation_rate=3, reduction=16, add=True):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels,
            dilation_rate: the rate of dilated convolution
            reduction: the reduction ratio in the F_glo
            add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = base_blocks.ConvINReLU(
            nIn, n, 1, 1
        )  # 1x1 Conv is employed to reduce the computation
        self.F_loc = base_blocks.ChannelWiseDilatedConv(n, n, 3, 1, 1)  # local feature
        self.F_sur = base_blocks.ChannelWiseDilatedConv(
            n, n, 3, 1, dilation_rate
        )  # surrounding context

        self.sigmoid = nn.Sigmoid()

        self.in_relu = base_blocks.INReLU(2*n)
        
        self.conv3113 = base_blocks.ChannelWiseConv(2 * n, 2 * n, 3, 1)  # 3x3 Conv is employed to fuse the joint feature

        self.add = add
        self.F_glo = base_blocks.FGlo(2*n, reduction)


    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        
        joi_feat = torch.cat([loc, sur], 1)

        joi_feat = self.sigmoid(joi_feat)
        joi_feat = input * joi_feat

        joi_feat = self.in_relu(joi_feat)

        joi_feat = self.conv3113(joi_feat)

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output = input + output

        return output