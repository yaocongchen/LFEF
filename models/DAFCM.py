###########################################################################
# Created by: Yao-Cong,Chen
# Email: yaocongchen@outlook.com
# Copyright (c) 2024 Yao-Cong,Chen. All rights reserved.
###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import base_blocks, SRDEM, DAFAM
from torchinfo import summary


__all__ = ["Net"]

class Net(nn.Module):
    """
    The proposed DAFCM model
    """

    def __init__(self, M=3, N=3):
        """
        args:
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()


        self.dafam = DAFAM()

        # stage 2
        self.level2_0 = SRDEM.Block_Down(
            32, 64, dilation_rate=2, reduction=8
        )
        self.level2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.level2.append(
                SRDEM.Block(64, 64, dilation_rate=2, reduction=8)
            )  # CG block
        self.in_relu_stage2 = base_blocks.INReLU(64)


        # stage 3
        self.level3_0 = SRDEM.Block_Down(
            64, 128, dilation_rate=4, reduction=16
        )
        self.level3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.level3.append(
                SRDEM.Block(128, 128, dilation_rate=4, reduction=16)
            )  # CG bloc
        self.in_relu_stage3 = base_blocks.INReLU(128)

        self.conv3x3_in_rule = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1,groups=64), nn.InstanceNorm2d(64, affine=True), nn.ReLU())
        self.conv1x1_IN = nn.Sequential(nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0), nn.InstanceNorm2d(1, affine=True))

#================================================================================================#
        self.conv_128_to_64 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=1,padding=0)
        self.conv_128_to_64_IN = nn.InstanceNorm2d(64, affine=True)
        self.upsample_to_64x64 = nn.Upsample(size=(64, 64), mode="bilinear", align_corners=True)

        self.conv_64_to_32 = nn.Conv2d(64, 32, kernel_size=(1, 1), stride=1,padding=0)
        self.conv_64_to_32_IN = nn.InstanceNorm2d(32, affine=True)
        self.upsample_to_128x128 = nn.Upsample(size=(128, 128), mode="bilinear", align_corners=True)
        
        self.conv_32_to_1 = nn.Conv2d(32, 1, kernel_size=(1, 1), stride=1,padding=0)
        self.conv_32_to_1_IN = nn.InstanceNorm2d(1, affine=True)
        self.upsample_to_256x256 = nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True)

#================================================================================================#
        
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
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
            elif classname.find("Linear") != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

                        
    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1
        dafam_output = self.dafam(input)

        # stage 2
        initial_stage2_output = self.level2_0(dafam_output)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                processed_stage2_output = layer(initial_stage2_output)
            else:
                processed_stage2_output = layer(processed_stage2_output)

        final_stage2_output = initial_stage2_output + processed_stage2_output
        final_stage2_output = self.in_relu_stage2(final_stage2_output)


        # stage 3
        initial_stage3_output = self.level3_0(final_stage2_output)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                processed_stage3_output = layer(initial_stage3_output)
            else:
                processed_stage3_output = layer(processed_stage3_output)

        processed_stage3_output_aux = self.conv3x3_in_rule(processed_stage3_output)
        processed_stage3_output_aux = self.upsample_to_256x256(processed_stage3_output_aux)
        processed_stage3_output_aux = self.conv1x1_IN(processed_stage3_output_aux)
        processed_stage3_output_aux = self.sigmoid(processed_stage3_output_aux)

        final_stage3_output = initial_stage3_output + processed_stage3_output
        final_stage3_output = self.in_relu_stage3(final_stage3_output)

        upsample_stage3_output = self.upsample_to_64x64(final_stage3_output)
        convolved_stage3_output = self.conv_128_to_64(upsample_stage3_output)
        convolved_stage3_output = self.conv_128_to_64_IN(convolved_stage3_output)

        stage3_add_stage2_output = convolved_stage3_output + final_stage2_output
        upsample_stage2_output = self.upsample_to_128x128(stage3_add_stage2_output)
        convolved_stage2_output = self.conv_64_to_32(upsample_stage2_output)
        convolved_stage2_output = self.conv_64_to_32_IN(convolved_stage2_output)

        stage2_add_stage1_output = convolved_stage2_output + dafam_output

        upsample_stage1_output = self.upsample_to_256x256(stage2_add_stage1_output)
        convolved_stage1_output = self.conv_32_to_1(upsample_stage1_output)
        convolved_stage1_output = self.conv_32_to_1_IN(convolved_stage1_output)

        output = self.sigmoid(convolved_stage1_output)

        return output , processed_stage3_output_aux

if __name__ == "__main__":
    model = Net()
    x = torch.randn(16, 3, 256, 256)
    output,aux = model(x)
    # print(output.shape)
    summary(model,input_data=x,verbose=1)