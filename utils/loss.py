import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torchvision.models as models

alpha = 0.2
lambda_reg = 0.2

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
    model_output, mask, smooth=1
):  # "Smooth" avoids a denominsator of 0 "Smooth"避免分母為0
    
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
############################################################################################################################
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

def ssim_val(model_output, mask):
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

def dice_coef(model_output, mask):
    model_output = (model_output > 0.5).float()
    intersection = torch.sum(model_output * mask)
    union = torch.sum(model_output) + torch.sum(mask)
    dice = 1 - (2.0 * intersection + 1) / (union + 1)  # 加上平滑項
    return dice

def boundary_loss(pred, target):

    pred = (pred > 0.5).float()
    # 計算邊界
    pred_boundary = F.max_pool2d(1 - pred, kernel_size=3, stride=1, padding=1) - (1 - pred)
    target_boundary = F.max_pool2d(1 - target, kernel_size=3, stride=1, padding=1) - (1 - target)

    # 計算邊界損失
    boundary_loss = L(pred_boundary, target_boundary)
    
    return boundary_loss

def l2_normalize(x, dim=None, epsilon=1e-12):
        square_sum = torch.sum(torch.square(x), dim=dim, keepdims=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.ones_like(square_sum) * epsilon))
        return torch.multiply(x, x_inv_norm)

def contrastive_loss(image_feat, cond_feat, temperature=0.07):
    """Calculates contrastive loss."""
    image_feat = F.normalize(image_feat, -1)
    cond_feat = l2_normalize(cond_feat, -1)
    
    bs = image_feat.shape[0]
    labels = torch.arange(bs, device=image_feat.device)

    logits_img2cond = torch.matmul(image_feat, cond_feat.t()) / temperature
    logits_cond2img = torch.matmul(cond_feat, image_feat.t()) / temperature
    loss_img2cond = F.cross_entropy(logits_img2cond, labels)
    loss_cond2img = F.cross_entropy(logits_cond2img, labels)
    loss_img2cond = torch.mean(loss_img2cond)
    loss_cond2img = torch.mean(loss_cond2img)
    loss = loss_img2cond + loss_cond2img
    return loss

def CustomLoss(model_output, mask):
    # s_iou = Sigmoid_IoU(model_output,mask)
    # iou = IoU(model_output,mask)

    # my_ssim = ssim_val(model_output,mask)

    # dice_loss = dice_coef(model_output,mask)

    loss_1 = L(model_output, mask)

    model = models.vgg16(pretrained=True)



    model_output = model_output.cuda()
    mask = mask.cuda()


    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1) 
    model = model.cuda()
    with torch.no_grad():
    # 这里执行模型的推理或评估操作，梯度不会被计算
        model_output = model(model_output)
        mask = model(mask)

    print(model_output.shape)
    print(mask.shape)
    # model_output = torch.sum(model_output, dim=[2, 3])
    # mask = torch.sum(mask, dim=[2, 3])

    loss_2 = contrastive_loss(model_output, mask)

    # loss_2 = boundary_loss(model_output, mask)

    # total_loss = loss_1 * (1 - alpha) + (1 - iou) * (alpha/2) + (1 - my_ssim) * (alpha/2)
    # total_loss = loss_1 * (1 - alpha) + (1 - iou) * (alpha)
    total_loss = loss_1 + loss_2
    # total_loss = loss_1 
    
    return total_loss



if __name__ == '__main__':
    import torch
    x = torch.randn(2, 1, 256, 256)
    model = models.vgg16(pretrained=True)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1) 
    print(model.features(x).shape)

    out = model.features(x)
    print(model)
    f1 = torch.sum(out, dim=[2, 3])
    print(f1.shape)
  


    # loss = contrastive_loss(f1, f1)
    # print(loss)



    #model_x = nn.Sequential(model.features)
    #print(model_x)

