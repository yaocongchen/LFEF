import torch
import torchvision
import os
import argparse
import time
import shutil
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import random

import utils
import models.CGNet as network_model
from visualization_codes.inference import smoke_semantic


def folders_and_files_name():
    # Set save folder and file name 設定存檔資料夾與存檔名稱
    save_smoke_semantic_dir_name = "testing_multiple_result"
    if os.path.exists("./" + save_smoke_semantic_dir_name):
        shutil.rmtree(
            "./" + save_smoke_semantic_dir_name
        )  # Delete the original folder and content 將原有的資料夾與內容刪除
        os.makedirs("./" + save_smoke_semantic_dir_name)  # Create new folder 創建新的資料夾
    else:
        # if not os.path.exists("./" + save_smoke_semantic_dir_name):
        os.makedirs("./" + save_smoke_semantic_dir_name)
    save_smoke_semantic_image_name = "smoke_semantic_image"

    names = {}
    names["smoke_semantic_dir_name"] = save_smoke_semantic_dir_name
    names["smoke_semantic_image_name"] = save_smoke_semantic_image_name

    return names


def wandb_information(model_size, flops, params):
    wandb.init(
        # set the wandb project where this run will be logged
        project="lightssd-project",
        name=args["wandb_name"],
        # track hyperparameters and run metadata
        config={
            "Model_size": model_size,
            "FLOPs": flops,
            "Parameters": params,
            "test_images": args["test_images"],
            "test_masks": args["test_masks"],
            "batch_size": args["batch_size"],
            "num_workers": args["num_workers"],
            "model_path": args["model_path"],
        },
    )


# Main function 主函式
def smoke_segmentation(device, names):
    model = network_model.Net().to(device)
    model.load_state_dict(torch.load(args["model_path"]))

    model.eval()

    # Calculation model size parameter amount and calculation amount
    # 計算模型大小、參數量與計算量
    c = utils.metrics.Calculate(model)
    model_size = c.get_model_size()
    flops, params = c.get_params()

    # wandb.ai
    if args["wandb_name"] != "no":
        wandb_time_start1 = time.time()
        wandb_information(model_size, flops, params)
        wandb_time_end1 = time.time()
        wandb_time_total1 = wandb_time_end1 - wandb_time_start1
        wandb_time_total2_cache = 0

    epoch_loss = []
    epoch_miou = []
    epoch_dice_coef = []
    time_train = []
    i = 0

    seconds = time.time()  # Random number generation 亂數產生
    random.seed(seconds)  # 使用時間秒數當亂數種子

    testing_data = utils.dataset.DataLoaderSegmentation(
        args["test_images"], args["test_masks"], mode="test"
    )
    testing_data_loader = DataLoader(
        testing_data,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    count = 1
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

        torchvision.utils.save_image(
            RGB_image,
            f'./{names["smoke_semantic_dir_name"]}/test_RGB_image_{count}.jpg',
        )
        torchvision.utils.save_image(
            mask_image,
            f'./{names["smoke_semantic_dir_name"]}/test_mask_image_{count}.jpg',
        )

        torchvision.utils.save_image(
            output,
            f'./{names["smoke_semantic_dir_name"]}/test_output_{count}.jpg',
        )

        loss = utils.loss.CustomLoss(output, mask_image)
        miou = utils.metrics.mIoU(output, mask_image)
        dice_coef = utils.metrics.dice_coef(output, mask_image)

        epoch_loss.append(loss.item())
        epoch_miou.append(miou.item())
        epoch_dice_coef.append(dice_coef.item())

        average_epoch_loss_test = sum(epoch_loss) / len(epoch_loss)
        average_epoch_miou_test = sum(epoch_miou) / len(epoch_miou)
        average_epoch_dice_coef_test = sum(epoch_dice_coef) / len(epoch_dice_coef)

        pbar.set_postfix(
            test_loss=average_epoch_loss_test,
            test_miou=average_epoch_miou_test,
            test_dice_coef=average_epoch_dice_coef_test,
        )

        if args["wandb_name"] != "no":
            wandb_time_start2 = time.time()
            wandb.log(
                {
                    "test_loss": average_epoch_loss_test,
                    "test_miou": average_epoch_miou_test,
                    "test_dice_coef": average_epoch_dice_coef_test,
                }
            )
            wandb.log(
                {
                    "test_RGB_image": wandb.Image(
                        f'./{names["smoke_semantic_dir_name"]}/test_RGB_image_{count}.jpg'
                    )
                }
            )
            wandb.log(
                {
                    "test_mask_image": wandb.Image(
                        f'./{names["smoke_semantic_dir_name"]}/test_mask_image_{count}.jpg'
                    )
                }
            )
            wandb.log(
                {
                    "test_output": wandb.Image(
                        f'./{names["smoke_semantic_dir_name"]}/test_output_{count}.jpg'
                    )
                }
            )
            wandb_time_end2 = time.time()
            wandb_time_total2 = wandb_time_end2 - wandb_time_start2
            wandb_time_total2_cache += wandb_time_total2

    if args["wandb_name"] != "no":
        return wandb_time_total1 + wandb_time_total2_cache
    else:
        return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument(
    #     "-td", "--test_directory", required=True, help="path to test images directory"
    # )
    ap.add_argument(
        "-ti",
        "--test_images",
        default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS02/img/",
        help="path to hazy training images",
    )
    ap.add_argument(
        "-tm",
        "--test_masks",
        default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS02/mask/",
        help="path to mask",
    )
    # ap.add_argument(
    #     "-ti",
    #     "--test_images",
    #     default="/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/img/",
    #     help="path to hazy training images",
    # )
    # ap.add_argument(
    #     "-tm",
    #     "--test_masks",
    #     default="/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/gt/",
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
    ap.add_argument("-bs", "--batch_size", type=int, default=8, help="set batch_size")
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

    print("test_data:", args["test_images"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Testing on device {device}.")

    names = folders_and_files_name()
    # Calculate the total implement time 計算總執行時間
    time_start = time.time()
    wandb_time_total = smoke_segmentation(device, names)
    time_end = time.time()
    total_image = len(os.listdir(args["test_images"]))

    if args["wandb_name"] != "no":  # 此方式還是會誤差FPS4~5
        # Calculate FPS
        print(
            "FPS:{:.1f}".format(
                total_image / (time_end - time_start - wandb_time_total)
            )
        )
        spend_time = int(time_end - time_start - wandb_time_total)
        time_min = spend_time // 60
        time_sec = spend_time % 60
        print("totally cost:", f"{time_min}m {time_sec}s")
        wandb.log({"FPS": total_image / (time_end - time_start - wandb_time_total)})
    else:
        print("FPS:{:.1f}".format(total_image / (time_end - time_start)))
        spend_time = int(time_end - time_start)
        time_min = spend_time // 60
        time_sec = spend_time % 60
        print("totally cost:", f"{time_min}m {time_sec}s")
