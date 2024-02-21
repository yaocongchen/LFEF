###########################################################################
# Created by: Yao-Cong,Chen
# Email: yaocongchen@outlook.com
# Copyright (c) 2024 Yao-Cong,Chen. All rights reserved.
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.nn import init

__all__ = ["Context_Guided_Network"]
# Filter out variables, functions, and classes that other programs don't need or don't want when running cmd "from CGNet import *"


def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[
        :, :c1, :, :
    ].contiguous()  # contiguous: the memory location remains unchanged
    x2 = x[:, c1:, :, :].contiguous()  # contiguous：記憶體位置不變

    return x1, x2


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape (torch)
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()  # Transpose 轉置

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ConvINReLU(nn.Module):
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
        # self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.in_norm = nn.InstanceNorm2d(nOut, affine=True)
        self.act = nn.ReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.in_norm(output)
        # output = F.layer_norm(output, output.size()[1:])
        output = self.act(output)
        return output


class INReLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        # self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.in_norm = nn.InstanceNorm2d(nOut, affine=True)
        self.act = nn.ReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.in_norm(input)
        # output = F.layer_norm(input, input.size()[1:])
        output = self.act(output)
        return output


class ConvIN(nn.Module):
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
        # self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.in_norm = nn.InstanceNorm2d(nOut, affine=True)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.in_norm(output)
        # output = F.layer_norm(output, output.size()[1:])
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
                (kSize, 1),
                stride=stride,
                padding=(padding , 0),
                groups=nIn,
                bias=False,
                dilation=d,
            ),
            nn.Conv2d(
                nIn,
                nOut,
                (1, kSize),
                stride=stride,
                padding=(0 , padding),
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

# class ExternalAttention(nn.Module):

#     def __init__(self, d_model,S=64):
#         super().__init__()
#         self.mk=nn.Linear(d_model,S,bias=False)
#         self.mv=nn.Linear(S,d_model,bias=False)
#         self.softmax=nn.Softmax(dim=1)
#         self.init_weights()


#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             # elif isinstance(m, nn.BatchNorm2d):
#             elif isinstance(m, nn.InstanceNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self, queries):
#         attn=self.mk(queries) #bs,n,S
#         attn=self.softmax(attn) #bs,n,S
#         attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
#         out=self.mv(attn) #bs,n,d_model

#         return out
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
        self.conv1x1 = ConvINReLU(nIn, nOut, 3, 2)  #  size/2, channel: nIn--->nOut

        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)
        self.F_sur_4 = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate * 2)
        self.F_sur_8 = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate * 4)

        # self.bn = nn.BatchNorm2d(4 * nOut, eps=1e-3)
        self.in_norm = nn.InstanceNorm2d(4 * nOut, affine=True)
        self.act = nn.ReLU(4 * nOut)
        self.reduce = Conv(4 * nOut, nOut, 1, 1)  # reduce dimension: 2*nOut--->nOut

        self.F_glo = FGlo(nOut, reduction)

        # self.ea = ExternalAttention(d_model=nIn)
        # self.add_conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, padding=0, bias=False)

        # self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        # self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        sur_4 = self.F_sur_4(output)
        sur_8 = self.F_sur_8(output)

        joi_feat = torch.cat([loc, sur, sur_4, sur_8], 1)  #  the joint feature
        # joi_feat = torch.cat([sur_4, sur_8], 1)  #  the joint feature

        joi_feat = self.in_norm(joi_feat)
        # joi_feat = F.layer_norm(joi_feat, joi_feat.size()[1:])
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)  # channel= nOut

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

        # b, c, w, h = input.size()
        # input_3c = input.view(b, c, w * h).permute(0, 2, 1)
        
        # ea_output = self.ea(input_3c)
        # ea_output = ea_output.permute(0, 2, 1).view(b, c, w, h)
        # ea_output = self.add_conv(ea_output)
        # ea_output = self.avg_pool(ea_output) + self.max_pool(ea_output)

        # output = output * ea_output
        
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
        n = int(nOut / 4)
        self.conv1x1 = ConvINReLU(
            nIn, n, 1, 1
        )  # 1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(n, n, 3, 1)  # local feature
        self.F_sur = ChannelWiseDilatedConv(
            n, n, 3, 1, dilation_rate
        )  # surrounding context
        self.F_sur_4 = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate * 2)
        self.F_sur_8 = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate * 4)

        self.in_relu = INReLU(4*n)
        self.add = add
        self.F_glo = FGlo(4*n, reduction)

        # self.ea = ExternalAttention(d_model=nIn)
        # self.add_conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, padding=0, bias=False)

        # self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        # self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        sur_4 = self.F_sur_4(output)
        sur_8 = self.F_sur_8(output)
        
        #joi_feat = torch.cat([loc, sur], 1)
        joi_feat = torch.cat([loc, sur, sur_4, sur_8], 1)  #  the joint feature

        joi_feat = self.in_relu(joi_feat)

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output = input + output

        # b, c, w, h = input.size()
        # input_3c = input.view(b, c, w * h).permute(0, 2, 1)

        # ea_output = self.ea(input_3c)
        # ea_output = ea_output.permute(0, 2, 1).view(b, c, w, h)
        # ea_output = self.add_conv(ea_output)
        # ea_output = self.avg_pool(ea_output) + self.max_pool(ea_output)

        # output = output * ea_output
        
        return output


class InputInjection(nn.Module):
    def __init__(self, downsamplingRatio):
        super().__init__()
        self.avg_pool = nn.ModuleList()
        for i in range(0, downsamplingRatio):
            self.avg_pool.append(nn.AvgPool2d(3, stride=2, padding=1))

        self.max_pool = nn.ModuleList()
        for i in range(0, downsamplingRatio):
            self.max_pool.append(nn.MaxPool2d(3, stride=2, padding=1))

    def forward(self, input):
        avg_pool_input = input
        for avg_pool in self.avg_pool:
            avg_pool_input = avg_pool(avg_pool_input)

        max_pool_input = input
        for max_pool in self.max_pool:
            max_pool_input = max_pool(max_pool_input)

        input = avg_pool_input + max_pool_input
        
        return input


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True
        )

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True
        )

        # self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
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
        # self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.in_norm2 = nn.InstanceNorm2d(chann, affine=True)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.relu(output)
        output = self.conv1x3_1(output)
        output = self.in_norm(output)
        # output = F.layer_norm(output, output.size()[1:])
        output = self.relu(output)

        output = self.conv3x1_2(output)
        output = self.relu(output)
        output = self.conv1x3_2(output)
        output = self.in_norm2(output)
        # output = F.layer_norm(output, output.size()[1:])

        return self.relu(output + input)  # +input = identity (residual connection)
    

class AuxiliaryNetwork(nn.Module):
    def __init__(self, nIn, nOut, stride=1):
        super().__init__()
        # self.ea = ExternalAttention(d_model=nIn)
        self.conv_layer1 = nn.Sequential(nn.Conv2d(nIn, 8, kernel_size=3, stride=stride, padding=1, bias=False), nn.ReLU())
        self.conv_layer2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False) , nn.ReLU())
        self.conv_layer3 = nn.Sequential(nn.Conv2d(16, nOut, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU())

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding = 1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding = 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # b, c, w, h = input.size()
        # input_3c = input.view(b, c, w * h).permute(0, 2, 1)

        # ea_output = self.ea(input_3c)
        # ea_output = ea_output.permute(0, 2, 1).view(b, c, w, h)
        output = self.conv_layer1(input)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = self.avg_pool(output) + self.max_pool(output)

        output = self.sigmoid(output)

        return output
    

class BrightnessAdjustment(nn.Module):
    def __init__(self):
        super().__init__()
        self.brightness = nn.Parameter(torch.tensor([1.0]))

    def forward(self, input_image):

        adjusted_image = input_image * self.brightness
        return adjusted_image  
    


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
        self.brightness_adjustment = BrightnessAdjustment()

        self.level1_0 = ConvINReLU(3, 32, 3, 2)  # feature map size divided 2, 1/2
        self.level1_1 = non_bottleneck_1d(32, 1)
        self.level1_2 = non_bottleneck_1d(32, 2)

        self.sample1 = InputInjection(1)  # down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)  # down-sample for Input Injiection, factor=4


        self.aux_net = AuxiliaryNetwork(3, 32, stride = 2)


        # stage 2
        self.level2_0 = ContextGuidedBlock_Down(
            32, 64,dilation_rate=2, reduction=8
        )
        self.level2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.level2.append(
                ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8)
            )  # CG block
        self.in_relu_2 = INReLU(128)
        # self.bn_relu_2_2 = BNReLU(128 + 3)


        # stage 3
        self.level3_0 = ContextGuidedBlock_Down(
            128, 128, dilation_rate=4, reduction=16
        )
        self.level3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.level3.append(
                ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16)
            )  # CG bloc
        self.in_relu_3 = INReLU(256)


        if dropout_flag:
            print("have droput layer")
            self.classifier = nn.Sequential(
                nn.Dropout2d(0.1, False), Conv(3, classes, 1, 1)
            )
        else:
            self.classifier = nn.Sequential(Conv(3, classes, 1, 1))


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
        

        # self.external_attention = ExternalAttention(d_model=64)
        # self.conv_64_to_128 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        # self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        # self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)
        # self.conv_256_to_128 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(1, 1), padding=0), nn.ReLU())
        
        self.upsample = nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True)
        self.conv_32_to_1 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=(1, 1), padding=0,groups=1), nn.InstanceNorm2d(1, affine = True), nn.ReLU())
        self.conv_128_to_1 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=(1, 1), padding=0,groups=1), nn.InstanceNorm2d(1, affine = True), nn.ReLU())
        self.conv_256_to_1 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=(1, 1), padding=0,groups=1), nn.InstanceNorm2d(1, affine = True), nn.ReLU())

#================================================================================================#
        # self.conv_256_to_128 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=1,padding=0)
        # self.conv_256_to_128_IN = nn.InstanceNorm2d(128, affine=True)
        # self.upsample_to_64x64 = nn.Upsample(size=(64, 64), mode="bilinear", align_corners=True)

        # self.conv_256_to_32 = nn.Conv2d(256, 32, kernel_size=(1, 1), stride=1,padding=0)
        # self.conv_256_to_32_IN = nn.InstanceNorm2d(32, affine=True)
        # self.upsample_to_128x128 = nn.Upsample(size=(128, 128), mode="bilinear", align_corners=True)

        # self.conv_96_to_1 = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=1,padding=0)
        # self.conv_96_to_1_IN = nn.InstanceNorm2d(1, affine=True)
        # self.upsample_to_256x256 = nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True)

        # self.relu = nn.ReLU()
#================================================================================================#
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """

        # stage 1
        stage1_output= self.level1_0(input)
        stage1_output = self.level1_1(stage1_output)
        stage1_output = self.level1_2(stage1_output)
        # inp1 = self.sample1(input)
        # inp2 = self.sample2(input)

        input_inverted = 1 - input
        inverted_output = self.aux_net(input_inverted)
        stage1_ewp_inverted_output = stage1_output * inverted_output


        # stage 2
        initial_stage2_output = self.level2_0(stage1_ewp_inverted_output)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                processed_stage2_output = layer(initial_stage2_output)
            else:
                processed_stage2_output = layer(processed_stage2_output)

        final_stage2_output = self.in_relu_2(torch.cat([initial_stage2_output, processed_stage2_output], 1))


        # b, c, w, h = initial_stage2_output.size()
        # initial_stage2_output_3channel = initial_stage2_output.view(b, c, w * h).permute(0, 2, 1)

        # ea_output = self.external_attention(initial_stage2_output_3channel)
        # ea_output = ea_output.permute(0, 2, 1).view(b, c, w, h)
        # ea_output = self.conv_64_to_128(ea_output)
        # ea_output = self.avg_pool(ea_output) + self.max_pool(ea_output)

        # final_stage2_cat_ea_output = torch.cat([final_stage2_output, ea_output], 1)
        # final_stage2_cat_ea_output = self.conv_256_to_128(final_stage2_cat_ea_output)


        # stage 3
        initial_stage3_output = self.level3_0(final_stage2_output)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                processed_stage3_output = layer(initial_stage3_output)
            else:
                processed_stage3_output = layer(processed_stage3_output)

        final_stage3_output = self.in_relu_3(torch.cat([initial_stage3_output, processed_stage3_output], 1))

        stage1_ewp_inverted_output_up = self.upsample(stage1_ewp_inverted_output)
        stage1_ewp_inverted_output_up = self.conv_32_to_1(stage1_ewp_inverted_output_up)

        final_stage2_output_up = self.upsample(final_stage2_output)
        final_stage2_output_up = self.conv_128_to_1(final_stage2_output_up)

        final_stage3_output_up = self.upsample(final_stage3_output)
        final_stage3_output_up = self.conv_256_to_1(final_stage3_output_up)

        output = torch.cat([stage1_ewp_inverted_output_up, final_stage2_output_up, final_stage3_output_up], 1)
        output = self.classifier(output)

#================================================================================================#
        # upsample_stage3_output = self.upsample_to_64x64(final_stage3_output)
        # convolved_stage3_output = self.conv_256_to_128(upsample_stage3_output)
        # convolved_stage3_output = self.conv_256_to_128_IN(convolved_stage3_output)
        # # convolved_stage3_output = F.layer_norm(convolved_stage3_output, convolved_stage3_output.size()[1:])
        # convolved_stage3_output = self.relu(convolved_stage3_output)

        # stage3_cat_stage2_output = torch.cat([convolved_stage3_output, final_stage2_output], 1)
        # upsample_stage2_output = self.upsample_to_128x128(stage3_cat_stage2_output)
        # convolved_stage2_output = self.conv_256_to_32(upsample_stage2_output)
        # convolved_stage2_output = self.conv_256_to_32_IN(convolved_stage2_output)
        # # convolved_stage2_output = F.layer_norm(convolved_stage2_output, convolved_stage2_output.size()[1:])
        # convolved_stage2_output = self.relu(convolved_stage2_output)

        # stage2_cat_stage1_output = torch.cat([convolved_stage2_output, stage1_output], 1)
        # upsample_stage1_output = self.upsample_to_256x256(stage2_cat_stage1_output)
        # convolved_stage1_output = self.conv_96_to_1(upsample_stage1_output)
        # convolved_stage1_output = self.conv_96_to_1_IN(convolved_stage1_output)
        # # convolved_stage1_output = F.layer_norm(convolved_stage1_output, convolved_stage1_output.size()[1:])
        # convolved_stage1_output = self.relu(convolved_stage1_output)
#================================================================================================#

        output = self.sigmoid(output)

        return output

if __name__ == "__main__":
    model = Net()
    x = torch.randn(16, 3, 256, 256)
    output = model(x)
    print(output.shape)
    summary(model,input_data=x,verbose=1)