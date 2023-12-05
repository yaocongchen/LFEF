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
    def __init__(self, in_chs, out_chs,dropprob):
        super().__init__()
        self.conv7_7 = nn.Conv2d(1, 1, (7, 7), stride=1, padding=3, bias=True)
        self.myrelu = nn.ReLU()

        self.mysigmoid = nn.Sigmoid()

        self.conv1_1 = nn.Conv2d(in_chs, in_chs, (1, 1), stride=1, bias=True)
        # TODO: NO use upsample
        # self.upsamp = nn.Upsample(size=(64, 64), mode="bilinear", align_corners=True)

        # self.upsamp = nn.Upsample(size = (28,28),mode ='bilinear',align_corners = True)
        self.conv1_1f = nn.Conv2d(in_chs, out_chs, (1, 1), stride=1, bias=True)
        self.dropout = nn.Dropout2d(dropprob)
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
        self.AttenTrans_1 = AttnTrans(in_chs, 32, 0.03)
        self.AttenTrans_2 = AttnTrans(32, 32, 0.03)
        self.AttenTrans_3 = AttnTrans(32, 32, 0.03)

        self.AttenTrans_4 = AttnTrans(32, 64, 0.03)
        self.AttenTrans_5 = AttnTrans(64, 64, 0.03)
        self.AttenTrans_6 = AttnTrans(64, 64, 0.03)

        self.AttenTrans_7 = AttnTrans(64, 128, 0.03)
        self.AttenTrans_8 = AttnTrans(128, 128, 0.03)
        self.AttenTrans_9 = AttnTrans(128, out_chs,0.03)

        # self.AttenTrans_10 = AttnTrans(256, 512)
        # self.AttenTrans_11 = AttnTrans(512, 512)
        # self.AttenTrans_12 = AttnTrans(512, out_chs)

        self.maxpl = nn.MaxPool2d(2, stride=2)
        self.avgpl = nn.AvgPool2d(2, stride=2)

    def forward(self, input):
        output = self.AttenTrans_1(input)
        output = self.AttenTrans_2(output)
        output = self.AttenTrans_3(output)
        max_pl = self.maxpl(output)
        avg_pl = self.avgpl(output)
        stack_1 = max_pl + avg_pl

        output = self.AttenTrans_4(stack_1)
        output = self.AttenTrans_5(output)
        output = self.AttenTrans_6(output)
        max_pl = self.maxpl(output)
        avg_pl = self.avgpl(output)
        stack_2 = max_pl + avg_pl

        output = self.AttenTrans_7(stack_2)
        output = self.AttenTrans_8(output)
        output = self.AttenTrans_9(output)
        max_pl = self.maxpl(output)
        avg_pl = self.avgpl(output)
        stack_3 = max_pl + avg_pl

        # out_3 = self.AttenTrans_10(out_2)
        # out_3 = self.AttenTrans_11(out_3)
        # out_3 = self.AttenTrans_12(out_3)

        return stack_1, stack_2, stack_3


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


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: optional stride rate for down-sampling
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

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            groups=nIn,
            bias=False,
        )

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class DilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias=False,
            dilation=d,
        )

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Sequential(
            nn.Conv2d(
                nIn,
                nIn,
                (1, kSize),
                stride=stride,
                padding=(0, padding),
                groups=nIn,
                bias=False,
                dilation=d,
            ),
            nn.Conv2d(
                nIn,
                nOut,
                (kSize, 1),
                stride=stride,
                padding=(padding, 0),
                groups=nIn,
                bias=False,
                dilation=d,
            ),
        )

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        self.conv1x1 = ConvBNPReLU(nIn, nOut, 3, 2)  #  size/2, channel: nIn--->nOut

        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)
        self.F_sur_4 = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate * 2)
        self.F_sur_8 = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate * 4)

        # self.bn = nn.BatchNorm2d(4 * nOut, eps=1e-3)
        self.act = nn.PReLU(4 * nOut)
        self.reduce = Conv(4 * nOut, nOut, 1, 1)  # reduce dimension: 2*nOut--->nOut

        self.F_glo = FGlo(nOut, reduction)

        # self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        sur_4 = self.F_sur_4(output)
        sur_8 = self.F_sur_8(output)

        joi_feat = torch.cat([loc, sur, sur_4, sur_8], 1)  #  the joint feature
        # joi_feat = torch.cat([sur_4, sur_8], 1)  #  the joint feature

        # joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)  # channel= nOut

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

        # if self.dropout.p != 0:
        #     output = self.dropout(output)

        return output


class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels,
           add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = ConvBNPReLU(
            nIn, n, 1, 1
        )  # 1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(n, n, 3, 1)  # local feature
        self.F_sur = ChannelWiseDilatedConv(
            n, n, 3, 1, dilation_rate
        )  # surrounding context
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        

        joi_feat = torch.cat([loc, sur], 1)

        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output = input + output
        return output


class InputInjection(nn.Module):
    def __init__(self, downsamplingRatio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, downsamplingRatio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True
        )

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True
        )

        # self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

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

        # self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        # output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        # output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class Net(nn.Module):
    """
    This class defines the proposed Context Guided Network (CGNet) in this work.
    """

    def __init__(self, classes=1, M=3, N=3, dropout_flag=False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()

        self.level1_0 = ConvBNPReLU(3, 32, 3, 2)  # feature map size divided 2, 1/2
        self.level1_1 = non_bottleneck_1d(32, 0.03, 1)
        self.level1_2 = non_bottleneck_1d(32, 0.03, 2)

        self.sample1 = InputInjection(1)  # down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)  # down-sample for Input Injiection, factor=4

        self.b1 = BNPReLU(32 + 3)

        # stage 2
        self.level2_0 = ContextGuidedBlock_Down(
            32 + 3, 64, dilation_rate=2, reduction=8
        )
        self.level2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.level2.append(
                ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8)
            )  # CG block
        self.bn_prelu_2 = BNPReLU(128 + 3)

        # stage 3
        self.level3_0 = ContextGuidedBlock_Down(
            128 + 3, 128, dilation_rate=4, reduction=16
        )
        self.level3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.level3.append(
                ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16)
            )  # CG block
        self.bn_prelu_3 = BNPReLU(256)

        if dropout_flag:
            print("have droput layer")
            self.classifier = nn.Sequential(
                nn.Dropout2d(0.1, False), nn.Conv2d(96, classes, 1, 1)
            )
        else:
            self.classifier = nn.Sequential(nn.Conv2d(96, classes, 1, 1))

        # init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Conv2d") != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find("ConvTranspose2d") != -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

        self.db = Detail_Branch(3, 128)
        self.conv1x1_67_32 = nn.Conv2d(67, 32, kernel_size=(1, 1), stride=1)
        self.conv1x1_195_32 = nn.Conv2d(195, 32, kernel_size=(1, 1), stride=1)
        self.conv1x1_384_32 = nn.Conv2d(384, 32, kernel_size=(1, 1), stride=1)
        self.upsample_128 = nn.Upsample(
            size=(128, 128), mode="bilinear", align_corners=True
        )
        self.upsample_256 = nn.Upsample(
            size=(256, 256), mode="bilinear", align_corners=True
        )
        # self.conv1x1_96_32 = nn.Conv2d(96, 1, kernel_size=(1, 1), stride=1)

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1

        stack_1, stack_2, stack_3 = self.db(input)

        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output0_cat_stack_1 = torch.cat([output0_cat, stack_1], 1)
        cat_conv_1 = self.conv1x1_67_32(output0_cat_stack_1)

        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))
        output1_cat_stack_2 = torch.cat([output1_cat, stack_2], 1)
        cat_conv_2 = self.conv1x1_195_32(output1_cat_stack_2)

        # stage 3
        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
        output2_cat_stack_3 = torch.cat([output2_cat, stack_3], 1)
        cat_conv_3 = self.conv1x1_384_32(output2_cat_stack_3)

        # upsample_1 = self.upsample(cat_conv_1)
        upsample_2 = self.upsample_128(cat_conv_2)
        upsample_3 = self.upsample_128(cat_conv_3)
        output = torch.cat([cat_conv_1, upsample_2, upsample_3], 1)
        output = self.upsample_256(output)

        # classifier
        output = self.classifier(output)

        # # upsample segmenation map ---> the input image size
        # out = F.upsample(
        #     classifier, input.size()[2:], mode="bilinear", align_corners=False
        # )  # Upsample score map, factor=8
        return output


if __name__ == "__main__":
    model = Net()
    x = torch.randn(16, 3, 256, 256)
    output = model(x)
    print(output.shape)
    summary(model, input_size=(16, 3, 256, 256))
