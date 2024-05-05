###########################################################################
# Created by: Yao-Cong,Chen
# Email: yaocongchen@outlook.com
# Copyright (c) 2024 Yao-Cong,Chen. All rights reserved.
###########################################################################
import torch
import torchvision
import os
import argparse
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import segmentation_models_pytorch as smp

import utils
import models.CGNet_2_erfnet31_13_3113_oneloss_inv_attention as network_model
import check_feature
from utils.inference import smoke_semantic
from utils.test_setup_utils import wandb_information, folders_and_files_name, parse_arguments
from utils.metrics import report_fps_and_time

model_name = str(network_model)
print("model_name:", model_name)


# Main function 主函式
def smoke_segmentation(model, device, names, args):
    print("test_data:", args["test_images"])

    epoch_loss = []
    epoch_iou = []
    epoch_SSIM = []
    epoch_hd = []
    time_train = []
    i = 0

    testing_data = utils.dataset.DatasetSegmentation(
        args["test_images"], args["test_masks"], mode="all"
    )
    testing_data_loader = DataLoader(
        testing_data,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    folders = ["test_RGB_image", "test_mask_image", "test_output", "test_check_feature"]

    for folder in folders:
        os.makedirs(f'./{names["smoke_semantic_dir_name"]}/{folder}', exist_ok=True)
    
    count = 0
    pbar = tqdm((testing_data_loader), total=len(testing_data_loader))
    for RGB_image, mask_image in pbar:
        img_image = RGB_image.to(device)
        mask_image = mask_image.to(device)

        with torch.no_grad():
            output, aux = smoke_semantic(img_image, model, device, time_train, i)


        loss = utils.loss.CustomLoss(output, mask_image)
        mask_image = mask_image.long()
        tp, fp, fn, tn = smp.metrics.get_stats(output, mask_image, mode='binary', threshold=0.5)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        mask_image = mask_image.float()
        # iou = utils.metrics.IoU(output, mask_image)
        customssim = utils.metrics.ssim_val(output, mask_image)
        hd = utils.metrics.Sobel_hausdorffDistance_metric(output, mask_image, device)

        epoch_loss.append(loss.item())
        epoch_iou.append(iou.item())
        epoch_SSIM.append(customssim.item())
        epoch_hd.append(hd.item())

        average_epoch_loss_test = sum(epoch_loss) / len(epoch_loss)
        average_epoch_miou_test = sum(epoch_iou) / len(epoch_iou) * 100
        average_epoch_mSSIM_test = sum(epoch_SSIM) / len(epoch_SSIM) * 100
        average_epoch_hd_test = sum(epoch_hd) / len(epoch_hd)

        pbar.set_postfix(
            test_loss=average_epoch_loss_test,
            test_miou=average_epoch_miou_test,
            test_mSSIM=average_epoch_mSSIM_test,
            test_hd=average_epoch_hd_test,
        )

        count += 1
        output = (output > 0.5).float()

        images_and_labels = [
            (RGB_image, "test_RGB_image"),
            (mask_image, "test_mask_image"),
            (output, "test_output"),
        ]

        for image, label in images_and_labels:
            torchvision.utils.save_image(
                image,
                f'./{names["smoke_semantic_dir_name"]}/{label}/{label}_{count}.jpg',
            )

            if args["wandb_name"] != "no":
                wandb.log(
                    {
                        label: wandb.Image(
                            f'./{names["smoke_semantic_dir_name"]}/{label}/{label}_{count}.jpg'
                        )
                    }
                )

        if args["wandb_name"] != "no":
            wandb.log(
                {
                    "test_loss": average_epoch_loss_test,
                    "test_miou": average_epoch_miou_test,
                    "test_mSSIM": average_epoch_mSSIM_test,
                    "test_hd": average_epoch_hd_test,
                }
            )
            
    return average_epoch_loss_test, average_epoch_miou_test, average_epoch_mSSIM_test, average_epoch_hd_test

if __name__ == "__main__":
    args = parse_arguments()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Testing on device {device}.")

    names = folders_and_files_name()

    model = network_model.Net().to(device)

    # Calculation model size parameter amount and calculation amount
    # 計算模型大小、參數量與計算量
    c = utils.metrics.Calculate(model)
    model_size = c.get_model_size()
    flops, params = c.get_params()

    # wandb.ai
    if args["wandb_name"] != "no":
        wandb_information(model_name, model_size, flops, params,args)

    model = torch.compile(model)  #pytorch2.0編譯功能(舊GPU無法使用)
    torch.set_float32_matmul_precision('high')
    model.load_state_dict(torch.load(args["model_path"], map_location=device))
    model.eval()

    time_start = time.time()
    Avg_loss, Avg_miou, Avg_mSSIM, Avg_hd = smoke_segmentation(model, device, names, args)
    time_end = time.time()
    total_image = len(os.listdir(args["test_images"]))
    fps, time_min, time_sec = report_fps_and_time(total_image, time_start, time_end)

    print(f"loss: {Avg_loss:.4f}")
    print(f"mIoU: {Avg_miou:.2f}%")
    
    # wandb.ai
    if args["wandb_name"] != "no":
        wandb.log({"FPS": fps})
