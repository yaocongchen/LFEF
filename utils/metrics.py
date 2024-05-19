import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
import torchvision.transforms as T
import numpy as np
from skimage.metrics import structural_similarity
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from typing import Tuple, Union
from torch.nn import Module
from torch import Tensor

import utils.HausdorffDistance_losses as HD
S = nn.Sigmoid()
L = nn.BCELoss(reduction="mean")

def Sigmoid_IoU(model_output: Tensor, mask: Tensor, smooth: Union[int, float] = 1) -> Tensor: 
    # "Smooth" avoids a denominsator of 0 "Smooth"避免分母為0
    model_output = S(model_output)
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


def IoU(model_output: Tensor, mask: Tensor, smooth: Union[int, float] = 1) -> Tensor:
    # "Smooth" avoids a denominsator of 0 "Smooth"避免分母為0
    # model_output = S(model_output)
    # print("model_output:",model_output.shape)
#==============================================================================================================#
    # model_output = (
    #     model_output.mul(255)
    #     .add_(0.5)
    #     .clamp_(0, 255)
    # )

    model_output = (model_output > 0.5).float()
#==============================================================================================================#

############################################################################################################################
    # torch.set_printoptions(profile="full")
    # # model_output = S(model_output)
    # # print("model_output:",model_output.shape)
    # output_np = (
    #     model_output.mul(255)
    #     .add_(0.5)
    #     .clamp_(0, 255)
    #     .contiguous()
    #     .to("cpu", torch.uint8)
    #     .detach()
    #     .numpy()
    # )

    # np.set_printoptions(threshold=np.inf)
    # output_np[output_np >= 1] = 1
    # # output_np[1< output_np] = 0

    # model_output = torch.from_numpy(output_np).to(device).float()
#############################################################################################################
    # model_output = S(model_output)

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

def ssim_val(model_output: Tensor, mask: Tensor) -> float:

    # model_output = (
    #     model_output.squeeze()
    #     .mul(255)
    #     .add_(0.5)
    #     .clamp_(0, 255)
    #     .contiguous()
    #     .to("cpu")
    #     .detach()
    #     .numpy()
    # )

    # model_output = (model_output > 0.5)
    # # print("model_output",model_output)
    # mask = (mask.squeeze()
    #     .contiguous()
    #     .to("cpu")
    #     .detach()
    #     .numpy()
    # )
    # # print("mask",mask)
    # # # Compute SSIM between two images
    # msssim = structural_similarity(model_output, mask,data_range=1,win_size=11,win_sigma=1.5,size_average=True,k1=0.01,k2=0.03,gaussian_weights=True)


    # model_output = (
    #     model_output.mul(255)
    #     .add_(0.5)
    #     .clamp_(0, 255)
    # )

    model_output = (model_output > 0.5).float()
    msssim = ssim(model_output, mask, data_range=1)
    
    return msssim

def Sobel_hausdorffDistance_metric(model_output: Tensor, mask: Tensor, device: torch.device) -> float:
    #Sobel
    def Sobel_process(input):
        # gray_image_tensor = 0.2989 * input[:, 0, :, :] + 0.5870 * input[:, 1, :, :] + 0.1140 * input[:, 2, :, :]
        # gray_image_tensor = gray_image_tensor.unsqueeze(0)

        # 定義 Sobel 運算子
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).to(device).view(1, 1, 3, 3)
        
        #Sobel 運算子得到垂直方向的梯度
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).to(device).view(1, 1, 3, 3)
        
        # 計算圖像在水平和垂直方向上的梯度

        gradient_x = nn.functional.conv2d(input, sobel_x)
        gradient_y = nn.functional.conv2d(input, sobel_y)
        
        # 計算梯度的大小
        output = torch.sqrt(gradient_x**2 + gradient_y**2)
        
        return output
    
    model_output = (model_output > 0.5).float()
    model_output = Sobel_process(model_output)
    # torchvision.utils.save_image (model_output, '/home/yaocong/Experimental/speed_smoke_segmentation/sobel_test.jpg')

    mask = Sobel_process(mask)
    # torchvision.utils.save_image (mask, '/home/yaocong/Experimental/speed_smoke_segmentation/sobel_test_mask.jpg')

    #去掉為度為1的部份
    model_output = model_output.squeeze()
    mask = mask.squeeze()
    HausdorffDistance = HD.AveragedHausdorffLoss()
    hd = HausdorffDistance(model_output, mask)

    return hd

def dice_coef(model_output: Tensor, mask: Tensor) -> float:
    model_output = (model_output > 0.5).float()
    intersection = torch.sum(model_output * mask)
    union = torch.sum(model_output) + torch.sum(mask)
    dice = 1 - (2.0 * intersection + 1) / (union + 1)  # 加上平滑項
    return dice


# def dice_p_bce(in_gt, in_pred):
#     return 1e-3 * L(in_gt, in_pred) - dice_coef(in_gt, in_pred)


# def true_positive_rate(y_true, y_pred):
#     return torch.sum(
#         torch.flatten(y_true) * torch.flatten(torch.round(y_pred))
#     ) / torch.sum(y_true)


# def true_positive_rate(model_output,mask):  ##Calculate the true positive rate, use the torch.flatten function to flatten the model_output and mask into a one-dimensional tensor, and then convert the value of the mask to 0 or 1 through torch.round to obtain a binarized label tensor. Next, multiply the two flattened tensors to get the number of overlapping pixels, and then divide by the total number of pixels in the mask tensor to get the true positive rate. Specifically, the numerator is the intersection of the pixels predicted to be positive in the model output and the pixels marked as positive in the label, and the denominator is the number of all positive pixels in the label.
#     return torch.sum(torch.flatten(model_output)*torch.flatten(torch.round(mask)))/torch.sum(mask)   #計算真陽性率，使用 torch.flatten 函數將 model_output 和 mask 展平成一維張量，然後通過 torch.round 將 mask 的值轉換為 0 或 1，以得到二值化的標籤張量。接下來，將這兩個展平後的張量相乘，得到重合的像素點的個數，再除以 mask 張量中的總像素點個數，即可得到真陽性率。具體地，分子為模型輸出中被預測為正例的像素點與標籤中被標記為正例的像素點的交集，分母為標籤中所有正例像素點的個數。


class Calculate:
    def __init__(self, model: Module):
        self.model = model
        self.model_size = self.Calculate_model_size()
        self.params = self.Calculations_and_parameters()

    # Calculate model size 計算模型大小
    def Calculate_model_size(self) -> str:
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print("{:<30}  {:.3f} {}".format("model size: ", size_all_mb, "MB"))
        model_size_mb = "%.2f" % size_all_mb + " MB"
        return model_size_mb

    # Display calculation amount and parameter amount 顯示計算量與參數量
    def Calculations_and_parameters(self) -> Tuple[str, str]:
        FLOPs, params = get_model_complexity_info(
            self.model,
            (4, 256, 256),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )  # Calculation model calculation amount and parameter amount print_per_layer_stat: List the parameter amount and calculation amount of each layer
        assert FLOPs is not None, 'FLOPs is None'
        assert params is not None, 'Params is None'
        print(
            "{:<30}  {:<8}".format("Computational complexity(FLOPs): ", FLOPs)
        )  # 計算模型計算量與參數量 print_per_layer_stat：列出每一層的參數量與計算量
        print("{:<30}  {:<8}".format("Number of parameters: ", params))
        return FLOPs, params

    def get_model_size(self) -> str:
        return self.model_size

    def get_params(self) -> Tuple[str, str]:
        return self.params
    
def report_fps_and_time(total_image: int, time_start: float, time_end: float) -> Tuple[float, int, int]:
    fps = total_image / (time_end - time_start)
    fps = round(fps, 1)
    print(f"FPS:{fps}")
    spend_time = int(time_end - time_start )
    time_min = spend_time // 60
    time_sec = spend_time % 60
    print("totally cost:", f"{time_min}m {time_sec}s")
    return fps, time_min, time_sec

if __name__ == "__main__":
    x = torch.rand(1, 1, 3, 3)
    print("x", x)
    y = torch.rand(1, 1, 3, 3)
    print("y", y)
