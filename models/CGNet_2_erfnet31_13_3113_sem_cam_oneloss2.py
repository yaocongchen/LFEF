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

        self.bn = nn.BatchNorm2d(4 * nOut, eps=1e-3)
        self.act = nn.PReLU(4 * nOut)
        self.reduce = Conv(4 * nOut, nOut, 1, 1)  # reduce dimension: 2*nOut--->nOut

        self.F_glo = FGlo(nOut, reduction)


    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        sur_4 = self.F_sur_4(output)
        sur_8 = self.F_sur_8(output)

        joi_feat = torch.cat([loc, sur, sur_4, sur_8], 1)  #  the joint feature
        # joi_feat = torch.cat([sur_4, sur_8], 1)  #  the joint feature

        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)  # channel= nOut

        output = self.F_glo(joi_feat)  # F_glo is employed to refine the joint feature

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
        self.conv1x1 = ConvBNPReLU(
            nIn, n, 1, 1
        )  # 1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(n, n, 3, 1)  # local feature
        self.F_sur = ChannelWiseDilatedConv(
            n, n, 3, 1, dilation_rate
        )  # surrounding context
        self.F_sur_4 = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate * 2)
        self.F_sur_8 = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate * 4)

        self.bn_prelu = BNPReLU(4*n)
        self.add = add
        self.F_glo = FGlo(4*n, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        sur_4 = self.F_sur_4(output)
        sur_8 = self.F_sur_8(output)
        
        #joi_feat = torch.cat([loc, sur], 1)
        joi_feat = torch.cat([loc, sur, sur_4, sur_8], 1)  #  the joint feature

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
    def __init__(self, chann, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True
        )

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True
        )

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

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
        self.prelu = nn.PReLU(chann)
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.prelu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.prelu(output)

        output = self.conv3x1_2(output)
        output = self.prelu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        return self.prelu(output + input)  # +input = identity (residual connection)
    
#===============================SEM====================================#
class SEM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv11_64out32 = nn.Sequential(
            nn.Conv2d(
                in_ch, in_ch // 2, kernel_size=(1, 1), padding="same"
            ),  # TODO:有自行除於2
            nn.BatchNorm2d(in_ch // 2),
        )
        self.avgpl = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpl = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv11_130 = nn.Sequential(
            nn.Conv2d(in_ch-1, in_ch, kernel_size=(1, 1), padding="same"),
            nn.BatchNorm2d(in_ch),
        )

        self.gavgpl = nn.AdaptiveAvgPool2d(1)

        self.conv11_131 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(1, 1), padding="same"),
            nn.BatchNorm2d(in_ch),
        )
        self.mysigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv11_64out32(x)
        f20 = F.relu(out)
        f21 = self.avgpl(f20)
        f22 = self.maxpl(f20)
        f23 = torch.cat((f21, f22), dim=1)
        out = self.conv11_130(f23)
        f27 = F.relu(out)

        f24 = self.gavgpl(x)
        f25 = self.conv11_131(f24)
        f26 = self.mysigmoid(f25)

        f28 = f26 * f27
        f29 = f28 + x

        return f29 
    
#===============================CAM====================================#
k=2
class CAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        ck = in_ch // k
        self.conv33_cin_cin = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(in_ch),
        )
        self.conv33_cin_ckout = nn.Sequential(
            nn.Conv2d(in_ch, ck, kernel_size=(3, 3), padding="same"), nn.BatchNorm2d(ck)
        )

        self.conv11_cin_ckout = nn.Conv1d(in_ch, ck, kernel_size=3, padding="same")
        self.conv11_ckin_cout = nn.Conv1d(ck, in_ch, kernel_size=3, padding="same")
        self.mysoftmax = nn.Softmax(dim=1)

        self.conv33_cin_cout = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(in_ch),
        )

    def forward(self, x):
        f4 = self.conv33_cin_cin(x)
        batchsize, num_channels, height, width = f4.data.size()

        # reshape (torch版的)
        f5 = f4.view(-1, num_channels, height * width)
        f6 = self.conv11_cin_ckout(f5)
        f9 = self.mysoftmax(f6)

        f7 = self.conv33_cin_ckout(x)

        batchsize, num_channels, height, width = f7.data.size()
        f8 = f7.view(-1, num_channels, height * width)

        f10 = f9 * f8

        f11 = self.conv11_ckin_cout(f10)
        batchsize, num_channels, HW = f11.data.size()
        f11 = f11.view(batchsize, num_channels, 64, 64)
        f12 = self.conv33_cin_cout(f11)

        f13 = f11 + f12

        return f13
#===============================GCP====================================#
class GCP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.glavgpl = nn.AdaptiveAvgPool2d(1)
        self.glmaxpl = nn.AdaptiveMaxPool2d(1)
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_ch*2, out_ch, kernel_size=(1, 1), padding="same"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        f1 = self.glavgpl(x)
        f2 = self.glmaxpl(x)
        f3 = torch.cat((f1, f2), dim=1)
        f4 = self.conv11(f3)

        return f4
#===============================FFM====================================#
class FFM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), padding="same"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.upsamp = nn.Upsample(size=(128, 128), mode="bilinear", align_corners=True)

        self.conv11_in192_out128 = nn.Sequential(
            nn.Conv2d(291, out_ch+3, kernel_size=(1, 1), padding="same"),
            nn.BatchNorm2d(out_ch+3),
            nn.ReLU(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, sem_out, cam_out, gcp_out): 
        f1 = self.conv11(cam_out)
        f1 = self.upsamp(f1)
        f2 = torch.cat((sem_out, f1), dim=1)
        f2 = self.conv11_in192_out128(f2)
        f3 = self.upsamp(gcp_out)
        f4 = f2 * f3
        f4 = self.sigmoid(f4)

        return f4
    
class CSSAM_Down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        in_ch_2 = in_ch // 2
        self.conv31 = nn.Conv2d(
            in_ch_2, in_ch_2, kernel_size=(3, 1), padding="same", dilation=dilation
        )
        self.conv13 = nn.Conv2d(
            in_ch_2, in_ch_2, kernel_size=(1, 3), padding="same", dilation=dilation
        )
        self.batch_norm_2 = nn.BatchNorm2d(in_ch_2)

        self.conv11 = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1),stride=2)
        self.maxpl = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.mysigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = split(x)
        out31 = self.conv31(x1)
        out31norm = self.batch_norm_2(out31)
        out13 = self.conv13(out31norm)
        out13norm = self.batch_norm_2(out13)
        out13normre = F.relu(out13norm)

        out13 = self.conv13(x2)
        out13norm = self.batch_norm_2(out13)
        out31 = self.conv31(out13norm)
        out31norm = self.batch_norm_2(out31)
        out31normre = F.relu(out31norm)
        out11cat = self.conv11(torch.cat((out13normre, out31normre), dim=1))

        outmp = self.maxpl(x)

        outmp11 = self.conv11(outmp)
        outmp11norm = self.batch_norm(outmp11)
        outmp11normsgm = self.mysigmoid(outmp11norm)

        Ewp = out11cat * outmp11normsgm
        Ews = out11cat + Ewp

        return channel_shuffle(Ews, 2)

class CSSAM(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        in_ch_2 = in_ch // 2
        self.conv31 = nn.Conv2d(
            in_ch_2, in_ch_2, kernel_size=(3, 1), padding="same", dilation=dilation
        )
        self.conv13 = nn.Conv2d(
            in_ch_2, in_ch_2, kernel_size=(1, 3), padding="same", dilation=dilation
        )
        self.batch_norm_2 = nn.BatchNorm2d(in_ch_2)

        self.conv11 = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), padding="same")
        self.maxpl = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.mysigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = split(x)
        out31 = self.conv31(x1)
        out31norm = self.batch_norm_2(out31)
        out13 = self.conv13(out31norm)
        out13norm = self.batch_norm_2(out13)
        out13normre = F.relu(out13norm)

        out13 = self.conv13(x2)
        out13norm = self.batch_norm_2(out13)
        out31 = self.conv31(out13norm)
        out31norm = self.batch_norm_2(out31)
        out31normre = F.relu(out31norm)
        out11cat = self.conv11(torch.cat((out13normre, out31normre), dim=1))

        outmp = self.maxpl(x)

        outmp11 = self.conv11(outmp)
        outmp11norm = self.batch_norm(outmp11)
        outmp11normsgm = self.mysigmoid(outmp11norm)

        Ewp = out11cat * outmp11normsgm
        Ews = out11cat + Ewp

        return channel_shuffle(Ews, 2)
#===============================Net====================================#
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
        self.level1_1 = non_bottleneck_1d(32, 1)
        self.level1_2 = non_bottleneck_1d(32, 2)

        self.sample1 = InputInjection(1)  # down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(2)  # down-sample for Input Injiection, factor=4
        self.sample3 = InputInjection(3)

        self.b1 = BNPReLU(32 + 3)

        # stage 2
        self.level2_0 = ContextGuidedBlock_Down(
            32 + 3, 64,dilation_rate=2, reduction=8
        )
        self.level2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.level2.append(
                ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8)
            )  # CG block
        self.bn_prelu_2 = BNPReLU(128 + 3)

        self.sem = SEM(32+3, 6)

        # stage 3
        self.level3_0 = ContextGuidedBlock_Down(
            128 + 3, 128, dilation_rate=4, reduction=16
        )
        self.level3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.level3.append(
                ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16)
            )  # CG bloc
        self.bn_prelu_3 = BNPReLU(256+3)

        self.cam = CAM(128+3, 12)

        self.ffm = FFM(128+3, 256)

        if dropout_flag:
            print("have droput layer")
            self.classifier = nn.Sequential(
                nn.Dropout2d(0.1, False), Conv(425, classes, 1, 1)
            )
        else:
            self.classifier = nn.Sequential(Conv(425, classes, 1, 1))

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
        
        self.upsample = nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True)
        self.conv11_32 = nn.Sequential(nn.Conv2d(35, 1, kernel_size=(1, 1), padding="same"), nn.BatchNorm2d(1), nn.ReLU())
        self.conv11_64 = nn.Sequential(nn.Conv2d(131, 1, kernel_size=(1, 1), padding="same"), nn.BatchNorm2d(1), nn.ReLU())
        self.conv11_128 = nn.Sequential(nn.Conv2d(259, 1, kernel_size=(1, 1), padding="same"), nn.BatchNorm2d(1), nn.ReLU())
        self.conv11_256 = nn.Sequential(nn.Conv2d(259, 1, kernel_size=(1, 1), padding="same"), nn.BatchNorm2d(1), nn.ReLU())

        # self.my_simgoid = nn.Sigmoid()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """

        # stage 1
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        inp3 = self.sample3(input)

        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))

        # sem_out = self.sem(output0_cat)
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))

        # cam_out = self.cam(output1_cat)

        # stage 3
        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2,inp3], 1))

        # output_ffm = self.ffm(sem_out, cam_out, output2_cat)

        output0_up = self.upsample(output0_cat)
        output1_up = self.upsample(output1_cat)
        output2_up = self.upsample(output2_cat)
        # output_ffm_up = self.upsample(output_ffm)
        output = torch.cat([output0_up, output2_up], 1)
        # classifier
        classifier = self.classifier(output)
        # output = self.my_simgoid(classifier)
        # classifier2 = self.classifier(output2_cat)

        # upsample segmenation map ---> the input image size
        # out = F.interpolate(
        #     classifier, input.size()[2:], mode="bilinear", align_corners=False
        # )  # Upsample score map, factor=8
        # out2 = F.interpolate(
        #     classifier2, input.size()[2:], mode="bilinear", align_corners=False
        # )
        # out = self.my_simgoid(out)
        return classifier


if __name__ == "__main__":
    model = Net()
    x = torch.randn(16, 3, 256, 256)
    output = model(x)
    print(output.shape)
    summary(model, input_size=(16, 3, 256, 256))
