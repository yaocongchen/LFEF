import torch
import torch.nn as nn
from ptflops import get_model_complexity_info

S = nn.Sigmoid()
L = nn.BCELoss(reduction="mean")


def mIoU(
    model_output, mask, smooth=1
):  # "Smooth" avoids a denominsator of 0 "Smooth"避免分母為0
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


def dice_coef(
    model_output, mask, smooth=1
):  # "Smooth" avoids a denominsator of 0 "Smooth"避免分母為0
    model_output = S(model_output)
    intersection = torch.sum(
        model_output * mask, dim=[1, 2, 3]
    )  # Calculate the intersection 算出交集
    union = torch.sum(model_output, dim=[1, 2, 3]) + torch.sum(
        mask, dim=[1, 2, 3]
    )  # 2*Consider the overlapping part # Calculate the Dice coefficient of the model output and the real label, which is used to evaluate the performance of the binary segmentation model. The parameters model_output and mask are the model output and the real label respectively, and smooth is a constant used to avoid the case where the denominator is 0.
    return torch.mean(
        (2.0 * intersection + smooth) / (union + smooth), dim=0
    )  # 2*考慮重疊的部份 #計算模型輸出和真實標籤的Dice係數，用於評估二元分割模型的性能。參數model_output和mask分別為模型輸出和真實標籤，smooth是一個常數，用於避免分母為0的情況。


# def dice_p_bce(in_gt, in_pred):
#     return 1e-3 * L(in_gt, in_pred) - dice_coef(in_gt, in_pred)


# def true_positive_rate(y_true, y_pred):
#     return torch.sum(
#         torch.flatten(y_true) * torch.flatten(torch.round(y_pred))
#     ) / torch.sum(y_true)


# def true_positive_rate(model_output,mask):  ##Calculate the true positive rate, use the torch.flatten function to flatten the model_output and mask into a one-dimensional tensor, and then convert the value of the mask to 0 or 1 through torch.round to obtain a binarized label tensor. Next, multiply the two flattened tensors to get the number of overlapping pixels, and then divide by the total number of pixels in the mask tensor to get the true positive rate. Specifically, the numerator is the intersection of the pixels predicted to be positive in the model output and the pixels marked as positive in the label, and the denominator is the number of all positive pixels in the label.
#     return torch.sum(torch.flatten(model_output)*torch.flatten(torch.round(mask)))/torch.sum(mask)   #計算真陽性率，使用 torch.flatten 函數將 model_output 和 mask 展平成一維張量，然後通過 torch.round 將 mask 的值轉換為 0 或 1，以得到二值化的標籤張量。接下來，將這兩個展平後的張量相乘，得到重合的像素點的個數，再除以 mask 張量中的總像素點個數，即可得到真陽性率。具體地，分子為模型輸出中被預測為正例的像素點與標籤中被標記為正例的像素點的交集，分母為標籤中所有正例像素點的個數。


class Calculate:
    def __init__(self, model):
        self.model = model
        self.model_size = self.Calculate_model_size()
        self.params = self.Calculations_and_parameters()

    # Calculate model size 計算模型大小
    def Calculate_model_size(self):
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
    def Calculations_and_parameters(self):
        macs, params = get_model_complexity_info(
            self.model,
            (3, 256, 256),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=True,
        )  # Calculation model calculation amount and parameter amount print_per_layer_stat: List the parameter amount and calculation amount of each layer
        print(
            "{:<30}  {:<8}".format("Computational complexity(FLOPs): ", macs)
        )  # 計算模型計算量與參數量 print_per_layer_stat：列出每一層的參數量與計算量
        print("{:<30}  {:<8}".format("Number of parameters: ", params))
        return macs, params

    def get_model_size(self):
        return self.model_size

    def get_params(self):
        return self.params
