import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity

alpha = 0.75
lambda_reg = 0.01

S = nn.Sigmoid()
L = nn.BCELoss(reduction="mean")


# def SSIM(model_output, mask):
#     output_np = (
#         model_output.squeeze()
#         .mul(255)
#         .add_(0.5)
#         .clamp_(0, 255)
#         .contiguous()
#         .to("cpu")
#         .detach()
#         .numpy()
#     )

#     np.set_printoptions(threshold=np.inf)
#     output_np[output_np >= 1] = 1
#     # output_np[1< output_np] = 0

#     # model_output = torch.from_numpy(output_np).to("cuda")

#     mask = mask.squeeze().contiguous().to("cpu").detach().numpy()
#     # Compute SSIM between two images
#     (score, diff) = structural_similarity(output_np, mask, data_range=1, full=True)
#     # print("Image similarity", score)
#     return score


def CustomLoss(input1, input2, mask, mode):
    loss_1 = L(S(input1), mask)
    loss_2 = L(S(input2), mask)
    # ssim_loss = 1- SSIM(input1,mask)

    if mode == "train":
        total_loss = loss_2
        return total_loss
    else:
        return loss_2
