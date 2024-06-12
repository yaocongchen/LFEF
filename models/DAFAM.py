###########################################################################
# Created by: Yao-Cong,Chen
# Email: yaocongchen@outlook.com
# Copyright (c) 2024 Yao-Cong,Chen. All rights reserved.
###########################################################################
import torch
import torch.nn as nn
from models import base_blocks

class conv3x1_1x3_dil(nn.Module):

    def __init__(self, chann, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True
        )

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True
        )

        self.in_norm = nn.InstanceNorm2d(chann, affine=True)

        self.conv3x1_2 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(1 * dilated, 0),
            bias=True,
            dilation=(dilated, 1),
        )

        self.conv1x3_2 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 1 * dilated),
            bias=True,
            dilation=(1, dilated),
        )
        self.relu = nn.ReLU(chann)
        self.in_norm2 = nn.InstanceNorm2d(chann, affine=True)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.relu(output)
        output = self.conv1x3_1(output)
        output = self.in_norm(output)
        output = self.relu(output)

        output = self.conv3x1_2(output)
        output = self.relu(output)
        output = self.conv1x3_2(output)
        output = self.in_norm2(output)

        return self.relu(output + input)  # +input = identity (residual connection)
    
class AuxiliaryNetwork(nn.Module):

    def __init__(self, nIn, nOut, stride=1):
        super().__init__()
        # self.ea = ExternalAttention(d_model=nIn)
        self.conv_layer1 = nn.Sequential(nn.Conv2d(nIn, 8, kernel_size=3, stride=stride, padding=1, bias=True),nn.ReLU())
        self.conv_layer2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU())
        self.conv_layer3 = nn.Sequential(nn.Conv2d(16, nOut, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU())

    def forward(self, input):
        output = self.conv_layer1(input)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)


        return output
    
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv = nn.Conv2d(in_channels*2, in_channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)

        return self.sigmoid(out)   
    
class DAFAM(nn.Module):
    """
    The proposed DAFAM model
    """

    def __init__(self):

        super().__init__()   
        self.level1_0 = base_blocks.ConvINReLU(3, 32, 3, 2)  # feature map size divided 2, 1/2
        self.level1_1 = conv3x1_1x3_dil(32, 1)
        self.level1_2 = conv3x1_1x3_dil(32, 2)

        self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)


        self.aux_net = AuxiliaryNetwork(3, 32, stride = 2)
        self.attention_module = AttentionModule(32)
        self.in_relu_stage1 = base_blocks.INReLU(32)

    def forward(self, input):
        stage1_output= self.level1_0(input)
        stage1_output = self.level1_1(stage1_output)
        stage1_output = self.level1_2(stage1_output)

        stage1_output = self.attention_module(stage1_output)

        input_inverted = 1 - input

        inverted_output = self.aux_net(input_inverted)
        inverted_output = self.attention_module(inverted_output)
        
        attention_output = stage1_output + inverted_output
        output = self.in_relu_stage1(stage1_output)

        return output