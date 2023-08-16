###########################################################################
# Created by: Tianyi Wu
# Email: wutianyi@ict.ac.cn
# Copyright (c) 2018
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

__all__ = ["Context_Guided_Network"]
# Filter out variables, functions, and classes that other programs don't need or don't want when running cmd "from CGNet import *"


class AttnTrans(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.conv7_7 = nn.Conv2d(1, 1, (7, 7), stride=1, padding=3, bias=True)
        self.myrelu = nn.ReLU()

        self.mysigmoid = nn.Sigmoid()

        self.conv1_1 = nn.Conv2d(in_chs, in_chs, (1, 1), stride=1, bias=True)
        # TODO: NO use upsample
        # self.upsamp = nn.Upsample(size=(64, 64), mode="bilinear", align_corners=True)

        # self.upsamp = nn.Upsample(size = (28,28),mode ='bilinear',align_corners = True)
        self.conv1_1f = nn.Conv2d(in_chs, out_chs, (1, 1), stride=1, bias=True)

    def forward(self, input):
        # Channel Avg
        channel_avg = torch.mean(input, dim=1)
        channel_avg = channel_avg.unsqueeze(1)
        channel_avg = self.conv7_7(channel_avg)
        channel_avg = self.myrelu(channel_avg)
        channel_avg = self.conv7_7(channel_avg)
        channel_avg = self.mysigmoid(channel_avg)
        # spatial Avg
        spatial_avg = torch.mean(input, dim=[2, 3])
        spatial_avg = spatial_avg.unsqueeze(2)
        spatial_avg = spatial_avg.unsqueeze(3)
        spatial_avg = self.conv1_1(spatial_avg)
        spatial_avg = self.myrelu(spatial_avg)
        spatial_avg = self.conv1_1(spatial_avg)
        spatial_avg = self.mysigmoid(spatial_avg)

        output = input * channel_avg
        output = output * spatial_avg
        # output = self.upsamp(output)
        output = self.conv1_1f(output)

        return output


class Detail_Branch(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.AttenTrans_1 = AttnTrans(in_chs, 16)
        self.AttenTrans_2 = AttnTrans(16, 16)
        self.AttenTrans_3 = AttnTrans(16, 32)
        # self.bn_32 = nn.BatchNorm2d(32, eps=1e-3)
        self.relu_32 = nn.ReLU(32)

        self.AttenTrans_4 = AttnTrans(32, 32)
        self.AttenTrans_5 = AttnTrans(32, 32)
        self.AttenTrans_6 = AttnTrans(32, 64)
        # self.bn_64 = nn.BatchNorm2d(64, eps=1e-3)
        self.relu_64 = nn.ReLU(64)

        self.AttenTrans_7 = AttnTrans(64, 64)
        self.AttenTrans_8 = AttnTrans(64, 64)
        self.AttenTrans_9 = AttnTrans(64, 128)
        self.bn_128 = nn.BatchNorm2d(128, eps=1e-3)
        self.relu_128 = nn.ReLU(128)

        self.AttenTrans_10 = AttnTrans(128, 128)
        self.AttenTrans_11 = AttnTrans(128, 128)
        self.AttenTrans_12 = AttnTrans(128, out_chs)
        # self.bn_256 = nn.BatchNorm2d(out_chs, eps=1e-3)
        self.relu_256 = nn.ReLU(out_chs)

        self.maxpl = nn.MaxPool2d(2, stride=2)
        self.avgpl = nn.AvgPool2d(2, stride=2)

    def forward(self, input):
        output = self.AttenTrans_1(input)
        output = self.AttenTrans_2(output)
        output = self.AttenTrans_3(output)
        output = self.bn_32(output)
        output = self.relu_32(output)
        max_pl = self.maxpl(output)
        avg_pl = self.avgpl(output)
        stack_1 = max_pl + avg_pl

        output = self.AttenTrans_4(stack_1)
        output = self.AttenTrans_5(output)
        output = self.AttenTrans_6(output)
        output = self.bn_64(output)
        output = self.relu_64(output)    
        max_pl = self.maxpl(output)
        avg_pl = self.avgpl(output)
        stack_2 = max_pl + avg_pl

        output = self.AttenTrans_7(stack_2)
        output = self.AttenTrans_8(output)
        output = self.AttenTrans_9(output)
        output = self.bn_128(output)
        output = self.relu_128(output)
        max_pl = self.maxpl(output)
        avg_pl = self.avgpl(output)
        stack_3 = max_pl + avg_pl

        output = self.AttenTrans_10(stack_3)
        output = self.AttenTrans_11(output)
        output = self.AttenTrans_12(output)
        output = self.bn_256(output)
        output = self.relu_256(output)
        max_pl = self.maxpl(output)
        avg_pl = self.avgpl(output)
        stack_4 = max_pl + avg_pl

        return stack_1, stack_2, stack_3, stack_4


class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BNPReLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class Net(nn.Module):
    """
    This class defines the proposed Context Guided Network (CGNet) in this work.
    """

    def __init__(self):
        super().__init__()

        self.db = Detail_Branch(3, 256)

        self.conv3_3_16 = nn.Conv2d(3, 16, (3, 3), stride=2, padding=1, bias=True)

        self.stage1 = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1, bias=True),
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1, bias=True),
        )
        self.context1 = nn.ModuleList()
        for i in range(0, 3):
            self.context1.append(self.stage1)
        self.bn_16 = nn.BatchNorm2d(16, eps=1e-3)
        self.relu_16 = nn.ReLU(16)

        self.conv3_3_16_32 = nn.Conv2d(16, 32, (3, 3), stride=2, padding=1, bias=True)
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1, bias=True),
            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1, bias=True),
        )
        self.context2 = nn.ModuleList()
        for i in range(0, 4):
            self.context2.append(self.stage2)
        self.bn_32 = nn.BatchNorm2d(32, eps=1e-3)
        self.relu_32 = nn.ReLU(32)

        self.conv3_3_32_64 = nn.Conv2d(32, 64, (3, 3), stride=2, padding=1, bias=True)
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1, bias=True),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1, bias=True),
        )
        self.context3 = nn.ModuleList()
        for i in range(0, 6):
            self.context3.append(self.stage3)
        self.bn_64 = nn.BatchNorm2d(64, eps=1e-3)
        self.relu_64 = nn.ReLU(64)

        self.conv3_3_64_128 = nn.Conv2d(64, 128, (3, 3), stride=2, padding=1, bias=True)
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), stride=1, padding=1, bias=True),
            nn.Conv2d(128, 128, (3, 3), stride=1, padding=1, bias=True),
        )
        self.context4 = nn.ModuleList()
        for i in range(0, 3):
            self.context4.append(self.stage4)
        self.bn_128 = nn.BatchNorm2d(128, eps=1e-3)
        self.relu_128 = nn.ReLU(128)

        self.conv3_3_128_256 = nn.Conv2d(
            128, 256, (3, 3), stride=2, padding=1, bias=True
        )

        self.upsample_256 = nn.Upsample(
            size=(256, 256), mode="bilinear", align_corners=True
        )

        self.conv_to_1 = nn.Conv2d(592, 1, (3, 3), stride=1, padding=1, bias=True)

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """

        stack_1, stack_2, stack_3, stack_4 = self.db(input)
        # stage 1
        output = self.conv3_3_16(input)

        for i, layer in enumerate(self.context1):
            output = layer(output)

        output = self.bn_16(output)
        output = self.relu_16(output)
        output_cat_stack_1 = torch.cat([output, stack_1], 1)
        upsample_1 = self.upsample_256(output_cat_stack_1)

        output = self.conv3_3_16_32(output)

        for i, layer in enumerate(self.context2):
            output = layer(output)

        output = self.bn_32(output)
        output = self.relu_32(output)
        output_cat_stack_2 = torch.cat([output, stack_2], 1)
        upsample_2 = self.upsample_256(output_cat_stack_2)
        output = self.conv3_3_32_64(output)

        for i, layer in enumerate(self.context3):
            output = layer(output)

        output = self.bn_64(output)
        output = self.relu_64(output)
        output_cat_stack_3 = torch.cat([output, stack_3], 1)
        upsample_3 = self.upsample_256(output_cat_stack_3)
        output = self.conv3_3_64_128(output)

        for i, layer in enumerate(self.context4):
            output = layer(output)

        output = self.bn_128(output)
        output = self.relu_128(output)
        output_cat_stack_4 = torch.cat([output, stack_4], 1)
        output = self.conv3_3_128_256(output)
        output = self.upsample_256(output)
        output = torch.cat([upsample_1, upsample_2, upsample_3, output], 1)

        output = self.conv_to_1(output)
        # stage 2

        return output


if __name__ == "__main__":
    model = Net()
    x = torch.randn(16, 3, 256, 256)
    output = model(x)
    print(output.shape)
    summary(model, input_data=(16, 3, 256, 256))
