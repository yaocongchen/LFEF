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
import models.CGNet_2_erfnet31_13_3113_oneloss_inv_attention as network_model
import check_feature
from utils.inference import smoke_semantic

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
outputs = []
# 設置一個函數來處理中間層的輸出
def hook_fn(module, input, output):
    # print(f"Output shape of intermediate layer: {output.shape}")
    outputs.append(output)

# Main function 主函式
def smoke_segmentation(model,device, names, args):
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

    os.makedirs(
        f'./{names["smoke_semantic_dir_name"]}/test_RGB_image', exist_ok=True
    )  # Create new folder 創建新的資料夾
    os.makedirs(
        f'./{names["smoke_semantic_dir_name"]}/test_mask_image', exist_ok=True
    )  # Create new folder 創建新的資料夾
    os.makedirs(
        f'./{names["smoke_semantic_dir_name"]}/test_output', exist_ok=True
    )  # Create new folder 創建新的資料夾
    os.makedirs(
        f'./{names["smoke_semantic_dir_name"]}/test_check_feature', exist_ok=True)
    
    count = 0
    pbar = tqdm((testing_data_loader), total=len(testing_data_loader))
    for RGB_image, mask_image in pbar:
        img_image = RGB_image.to(device)
        mask_image = mask_image.to(device)

        # 註冊 hook 到模型的中間層
        # handle = model.main_net.conv11_128.register_forward_hook(hook_fn)
        with torch.no_grad():
            output = smoke_semantic(img_image, model, device, time_train, i)
        # feature = check_feature.check_feature(32)
        # print("output.shape:", outputs[0].shape )

        # print("outputs.shape:", type(outputs[0]))
        # feature = check_feature.check_feature(1).to(device)
        # feat = feature(outputs[0])

        # # 移除 hook
        # handle.remove()

        loss = utils.loss.CustomLoss(output, mask_image)
        iou = utils.metrics.IoU(output, mask_image)
        # iou_s = utils.metrics.Sigmoid_IoU(output, mask_image)
        # dice_coef = utils.metrics.dice_coef(output, mask_image, device)
        customssim = utils.metrics.ssim_val(output, mask_image)
        hd = utils.metrics.Sobel_hausdorffDistance_metric(output, mask_image, device)

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

        count += 1
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
        # torchvision.utils.save_image(
        #     feat,
        #     f'./{names["smoke_semantic_dir_name"]}/test_check_feature/test_check_feature_{count}.jpg',
        # )  

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
        default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/images/",
        help="path to hazy training images",
    )
    ap.add_argument(
        "-tm",
        "--test_masks",
        default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/masks/",
        help="path to mask",
    )
    # ap.add_argument(
    #     "-ti",
    #     "--test_images",
    #     default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/Real/images/",
    #     help="path to hazy training images",
    # )
    # ap.add_argument(
    #     "-tm",
    #     "--test_masks",
    #     default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/Real/masks/",
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
    ap.add_argument("-m", "--model_path", required=False, default="/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/last.pth",help="load model path")
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
