import torch
import torch.nn as nn
import numpy as np

alpha = 0.5
lambda_reg = 0.01

S = nn.Sigmoid()
L = nn.BCELoss(reduction="mean")


# def Sigmoid_IoU(
#     model_output, mask, smooth=1
# ):  # "Smooth" avoids a denominsator of 0 "Smooth"避免分母為0
#     model_output = S(model_output)
#     intersection = torch.sum(
#         model_output * mask, dim=[1, 2, 3]
#     )  # Calculate the intersection 算出交集
#     union = (
#         torch.sum(model_output, dim=[1, 2, 3])
#         + torch.sum(mask, dim=[1, 2, 3])
#         - intersection
#         + 1e-6
#     )
#     return torch.mean(
#         (intersection + smooth) / (union + smooth), dim=0
#     )  # 2*考慮重疊的部份 #計算模型輸出和真實標籤的Dice係數，用於評估二元分割模型的性能。參數model_output和mask分別為模型輸出和真實標籤，smooth是一個常數，用於避免分母為0的情況。

def IoU(
    model_output, mask, device, smooth=1
):  # "Smooth" avoids a denominsator of 0 "Smooth"避免分母為0
    torch.set_printoptions(profile="full")
    # print("model_output:",model_output.shape)
    output_np = (
        model_output.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .contiguous()
        .to("cpu", torch.uint8)
        .detach()
        .numpy()
    )

    np.set_printoptions(threshold=np.inf)
    output_np[output_np >= 1] = 1
    # output_np[1< output_np] = 0

    model_output = torch.from_numpy(output_np).to(device).float()

    intersection = torch.sum(
        model_output * mask, dim=[1, 2, 3]
    )  # Calculate the intersection 算出交集
    # print("intersection",intersection)
    # print("torch.sum(model_output, dim=[1, 2, 3]:",torch.sum(model_output, dim=[1, 2, 3]))
    # print("torch.sum(mask, dim=[1, 2, 3]:",torch.sum(mask, dim=[1, 2, 3]))
    union = (
        torch.sum(model_output, dim=[1, 2, 3])
        + torch.sum(mask, dim=[1, 2, 3])
        - intersection
        + 1e-6
    )
    # print("union",union)
    # print("mean:",torch.mean((intersection + smooth) / (union + smooth), dim=0))
    return torch.mean(
        (intersection + smooth) / (union + smooth), dim=0
    )  # 2*考慮重疊的部份 #計算模型輸出和真實標籤的Dice係數，用於評估二元分割模型的性能。參數model_output和mask分別為模型輸出和真實標籤，smooth是一個常數，用於避免分母為0的情況。


def CustomLoss(model_output, mask,device):
    # s_iou = Sigmoid_IoU(model_output,mask)
    iou = IoU(model_output,mask,device)

    loss_1 = L(S(model_output), mask)

    # total_loss = loss_1 * (1 - alpha) + (1 - iou) * alpha
    total_loss = loss_1

    return total_loss

    
