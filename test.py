import torch
import torchvision
import os
import argparse
import time
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

import utils
import models.CGNet_2_erfnet31_13_3113_oneloss as network_model
from visualization_codes.inference import smoke_semantic
import utils.HausdorffDistance_losses as HD

model_name = str(network_model)
print("model_name:", model_name)

def folders_and_files_name():
    save_smoke_semantic_dir_name = "testing_multiple_result"
    shutil.rmtree(save_smoke_semantic_dir_name, ignore_errors=True)
    os.makedirs(save_smoke_semantic_dir_name)

    return {
        "smoke_semantic_dir_name": save_smoke_semantic_dir_name,
        "smoke_semantic_image_name": "smoke_semantic_image"
    }


def wandb_information(model_size, flops, params,args):
    wandb.init(
        project="lightssd-project-test",
        name=args["wandb_name"],
        config={
            "Model": model_name,
            "Model_size": model_size,
            "FLOPs": flops,
            "Parameters": params,
            **{k: args[k] for k in ["test_images", "test_masks", "batch_size", "num_workers"]}
        }
    )


# Main function 主函式
def smoke_segmentation(model,device, names,args):
    print("test_data:", args["test_images"])

    # Calculation model size parameter amount and calculation amount
    # 計算模型大小、參數量與計算量
    c = utils.metrics.Calculate(model)
    model_size = c.get_model_size()
    flops, params = c.get_params()

    # wandb.ai
    if args["wandb_name"] != "no":
        wandb_time_start1 = time.time()
        wandb_information(model_size, flops, params,args)
        wandb_time_end1 = time.time()
        wandb_time_total1 = wandb_time_end1 - wandb_time_start1
        wandb_time_total2_cache = 0

    epoch_loss = []
    epoch_iou = []
    # epoch_iou_s = []
    # epoch_dice_coef = []
    epoch_SSIM = []
    epoch_hd = []
    time_train = []
    i = 0

    testing_data = utils.dataset.DatasetSegmentation(
        args["test_images"], args["test_masks"], mode="test"
    )
    testing_data_loader = DataLoader(
        testing_data,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    os.makedirs(
        f'./{names["smoke_semantic_dir_name"]}/test_RGB_image'
    )  # Create new folder 創建新的資料夾
    os.makedirs(
        f'./{names["smoke_semantic_dir_name"]}/test_mask_image'
    )  # Create new folder 創建新的資料夾
    os.makedirs(
        f'./{names["smoke_semantic_dir_name"]}/test_output'
    )  # Create new folder 創建新的資料夾

    count = 0
    pbar = tqdm((testing_data_loader), total=len(testing_data_loader))
    for RGB_image, mask_image in pbar:
        img_image = RGB_image.to(device)
        mask_image = mask_image.to(device)

        output = smoke_semantic(img_image, model, device, time_train, i)

        count += 1
        # torchvision.utils.save_image(
        #     torch.cat((mask_image, output), 0),
        #     f'./{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_image_name"]}_{count}.jpg',
        # )

        output = (
            output.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
        )

        output = (output > 0.5).float()
        
        torchvision.utils.save_image(
            RGB_image,
            f'./{names["smoke_semantic_dir_name"]}/test_RGB_image/test_RGB_image_{count}.jpg',
        )
        torchvision.utils.save_image(
            mask_image,
            f'./{names["smoke_semantic_dir_name"]}/test_mask_image/test_mask_image_{count}.jpg',
        )
        torchvision.utils.save_image(
            output,
            f'./{names["smoke_semantic_dir_name"]}/test_output/test_output_{count}.jpg',
        )

        loss = utils.loss.CustomLoss(output, mask_image)
        iou = utils.metrics.IoU(output, mask_image)
        # iou_s = utils.metrics.Sigmoid_IoU(output, mask_image)
        # dice_coef = utils.metrics.dice_coef(output, mask_image, device)
        customssim = utils.metrics.ssim_val(output, mask_image)

        #去掉為度為1的部份
        output_for_HD = output.squeeze()
        mask_image_for_HD = mask_image.squeeze()
        HausdorffDistance = HD.AveragedHausdorffLoss()
        hd = HausdorffDistance(output_for_HD, mask_image_for_HD)


        epoch_loss.append(loss.item())
        epoch_iou.append(iou.item())
        # epoch_iou_s.append(iou_s.item())
        # epoch_dice_coef.append(dice_coef.item())
        epoch_SSIM.append(customssim.item())
        epoch_hd.append(hd.item())

        average_epoch_loss_test = sum(epoch_loss) / len(epoch_loss)
        average_epoch_miou_test = sum(epoch_iou) / len(epoch_iou) * 100
        # average_epoch_miou_s_test = sum(epoch_iou_s) / len(epoch_iou_s) * 100
        # average_epoch_dice_coef_test = sum(epoch_dice_coef) / len(epoch_dice_coef) * 100
        average_epoch_mSSIM_test = sum(epoch_SSIM) / len(epoch_SSIM) * 100
        average_epoch_hd_test = sum(epoch_hd) / len(epoch_hd)

        pbar.set_postfix(
            test_loss=average_epoch_loss_test,
            test_miou=average_epoch_miou_test,
            # test_miou_s=average_epoch_miou_s_test,
            # test_dice_coef=average_epoch_dice_coef_test,
            test_mSSIM=average_epoch_mSSIM_test,
            test_hd=average_epoch_hd_test,
        )

        if args["wandb_name"] != "no":
            wandb_time_start2 = time.time()
            wandb.log(
                {
                    "test_loss": average_epoch_loss_test,
                    "test_miou": average_epoch_miou_test,
                    # "test_miou_s": average_epoch_miou_s_test,
                    # "test_dice_coef": average_epoch_dice_coef_test,
                    "test_mSSIM": average_epoch_mSSIM_test,
                    "test_hd": average_epoch_hd_test,
                }
            )
            wandb.log(
                {
                    "test_RGB_image": wandb.Image(
                        f'./{names["smoke_semantic_dir_name"]}/test_RGB_image/test_RGB_image_{count}.jpg'
                    )
                }
            )
            wandb.log(
                {
                    "test_mask_image": wandb.Image(
                        f'./{names["smoke_semantic_dir_name"]}/test_mask_image/test_mask_image_{count}.jpg'
                    )
                }
            )
            wandb.log(
                {
                    "test_output": wandb.Image(
                        f'./{names["smoke_semantic_dir_name"]}/test_output/test_output_{count}.jpg'
                    )
                }
            )
            wandb_time_end2 = time.time()
            wandb_time_total2 = wandb_time_end2 - wandb_time_start2
            wandb_time_total2_cache += wandb_time_total2
            wandb_time_total = wandb_time_total1 + wandb_time_total2_cache

    if args["wandb_name"] != "no":
        return average_epoch_loss_test, average_epoch_miou_test, average_epoch_mSSIM_test, average_epoch_hd_test, wandb_time_total
    else:
        return average_epoch_loss_test, average_epoch_miou_test, average_epoch_mSSIM_test, average_epoch_hd_test


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument(
    #     "-td", "--test_directory", required=True, help="path to test images directory"
    # )
    ap.add_argument(
        "-ti",
        "--test_images",
        default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/img/",
        help="path to hazy training images",
    )
    ap.add_argument(
        "-tm",
        "--test_masks",
        default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/mask/",
        help="path to mask",
    )
    # ap.add_argument(
    #     "-ti",
    #     "--test_images",
    #     default="/home/yaocong/Experimental/Dataset/Smoke-Segmentation/Dataset/Train/Imag/",
    #     help="path to hazy training images",
    # )
    # ap.add_argument(
    #     "-tm",
    #     "--test_masks",
    #     default="/home/yaocong/Experimental/Dataset/Smoke-Segmentation/Dataset/Train/Mask/",
    #     help="path to mask",
    # )
    # ap.add_argument(
    #     "-ti",
    #     "--test_images",
    #     default="/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/120k/img/",
    #     help="path to hazy training images",
    # )
    # ap.add_argument(
    #     "-tm",
    #     "--test_masks",
    #     default="/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/120k/gt/",
    #     help="path to mask",
    # )
    # ap.add_argument(
    #     "-ti",
    #     "--test_images",
    #     default="/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/t/img/",
    #     help="path to hazy training images",
    # )
    # ap.add_argument(
    #     "-tm",
    #     "--test_masks",
    #     default="/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/t/gt/",
    #     help="path to mask",
    # )
    # ap.add_argument(
    #     "-ti",
    #     "--test_images",
    #     default="/home/yaocong/Experimental/Dataset/SMOKE5K_dataset/SMOKE5K/SMOKE5K/test/img/",
    #     help="path to hazy training images",
    # )
    # ap.add_argument(
    #     "-tm",
    #     "--test_masks",
    #     default="/home/yaocong/Experimental/Dataset/SMOKE5K_dataset/SMOKE5K/SMOKE5K/test/gt_/",
    #     help="path to mask",
    # )
    ap.add_argument("-bs", "--batch_size", type=int, default=1, help="set batch_size")
    ap.add_argument("-nw", "--num_workers", type=int, default=1, help="set num_workers")
    ap.add_argument("-m", "--model_path", required=True, help="load model path")
    ap.add_argument(
        "-wn",
        "--wandb_name",
        type=str,
        default="no",
        help="wandb test name,but 'no' is not use wandb",
    )
    args = vars(ap.parse_args())


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Testing on device {device}.")

    names = folders_and_files_name()
    # Calculate the total implement time 計算總執行時間

    model = network_model.Net().to(device)
    model.load_state_dict(torch.load(args["model_path"], map_location=device))
    model.eval()
    
    def calculate_and_print_fps(total_image, time_start, time_end, wandb_time_total=0):
        fps = total_image / (time_end - time_start - wandb_time_total)
        print("FPS:{:.1f}".format(fps))
        spend_time = int(time_end - time_start - wandb_time_total)
        time_min = spend_time // 60
        time_sec = spend_time % 60
        print("totally cost:", f"{time_min}m {time_sec}s")
        return fps

    if args["wandb_name"] != "no":  # 此方式還是會誤差FPS4~5
        time_start = time.time()
        Avg_loss, Avg_miou, Avg_mSSIM,Avg_hd, wandb_time_total= smoke_segmentation(model,device,names,args)
        time_end = time.time()
        total_image = len(os.listdir(args["test_images"]))
        fps = calculate_and_print_fps(total_image, time_start, time_end, wandb_time_total)
        wandb.log({"FPS": fps})
    else:
        time_start = time.time()
        Avg_loss, Avg_miou, Avg_mSSIM, Avg_hd = smoke_segmentation(model,device,names,args)
        time_end = time.time()
        total_image = len(os.listdir(args["test_images"]))
        calculate_and_print_fps(total_image, time_start, time_end)
