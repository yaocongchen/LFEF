import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torchvision.models as models
import segmentation_models_pytorch as smp
from typing import Any, Dict, Union, Tuple
from torch import Tensor

alpha = 0.2

S = nn.Sigmoid()
L = nn.BCELoss(reduction="mean")
smp_loss = smp.losses.MCCLoss()


def IoU(model_output: Tensor, mask: Tensor, smooth: int = 1) -> Tensor:
    # "Smooth" avoids a denominsator of 0 "Smooth"避免分母為0
    model_output = (model_output > 0.5).float()


    intersection = torch.sum(
        model_output * mask, dim=[1, 2, 3]
    )  # Calculate the intersection 算出交集

    union = (
        torch.sum(model_output, dim=[1, 2, 3])
        + torch.sum(mask, dim=[1, 2, 3])
        - intersection
        + 1e-6
    )

    return torch.mean(
        (intersection + smooth) / (union + smooth), dim=0
    )  # 2*考慮重疊的部份 #計算模型輸出和真實標籤的Dice係數，用於評估二元分割模型的性能。參數model_output和mask分別為模型輸出和真實標籤，smooth是一個常數，用於避免分母為0的情況。

def ssim_val(model_output: Tensor, mask: Tensor) -> Any:

    model_output = (model_output > 0.5).float()
    msssim = ssim(model_output, mask, data_range=1)
    
    return msssim

def dice_coef(model_output: Tensor, mask: Tensor) -> Tensor:
    model_output = (model_output > 0.5).float()
    intersection = torch.sum(model_output * mask)
    union = torch.sum(model_output) + torch.sum(mask)
    dice = 1 - (2.0 * intersection + 1) / (union + 1)  # 加上平滑項
    return dice

def boundary_loss(pred: Tensor, target: Tensor) -> Tensor:

    pred = (pred > 0.5).float()
    # 計算邊界
    pred_boundary = F.max_pool2d(1 - pred, kernel_size=3, stride=1, padding=1) - (1 - pred)
    target_boundary = F.max_pool2d(1 - target, kernel_size=3, stride=1, padding=1) - (1 - target)

    # 計算邊界損失
    boundary_loss = L(pred_boundary, target_boundary)
    
    return boundary_loss

def CustomLoss(*args: Tensor, **kwargs: Any) -> Tensor:
    model_output = args[0]
    mask = args[-1] 
    
    # dice_loss = dice_coef(model_output, mask)
    loss_1 = L(model_output, mask)

    if len(args) == 2:  # 只有 model_output 和 mask
        total_loss = loss_1
    elif len(args) == 3:  # model_output, aux, 和 mask
        aux = args[1]
        loss_2 = L(aux, mask)
        total_loss = loss_1 * (1 - alpha) + loss_2 * alpha
    else:
        raise ValueError("Unsupported number of arguments")
    
    return total_loss

if __name__ == '__main__':
    import torch
    x = torch.randn(2, 1, 256, 256)
    model = models.vgg16(weights=['VGG16_Weights.DEFAULT'])
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1) 
    model_vgg16 = nn.Sequential(model.features)
    print(model_vgg16)
    #print(model.features(x).shape)


    out = model_vgg16(x)
    print(out.shape)
    f1 = torch.sum(out, dim=[2, 3])

