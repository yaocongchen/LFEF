###########################################################################
# Created by: Yao-Cong,Chen
# Email: yaocongchen@outlook.com
# Copyright (c) 2024 Yao-Cong,Chen. All rights reserved.
###########################################################################
# %%
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import os
import time
import wandb
import torch.onnx

# import self-written modules
import models.CGNet_2_erfnet31_13_3113_oneloss_inv_attention as network_model  # import self-written models 引入自行寫的模型
from utils.setup_utils import set_save_dir_names, create_model_state_dict, time_processing, wandb_information, parse_arguments
from utils.check_GPU import check_have_GPU, check_number_of_GPUs
from utils.metrics import Calculate
from data_processing import data_processing
from training import train_epoch, valid_epoch
from utils.results_saving import save_model_and_state, save_and_log_image, save_experiment_details


onnx_img_image = []

model_name = str(network_model)
print("model_name:", model_name)

def main():
    train_images, train_masks, training_data_loader, validation_data_loader = data_processing(args)

    save_mean_miou = 0
    # save_mean_miou_s = 0
    check_have_GPU(args)
    # The cudnn function library assists in acceleration(if you encounter a problem with the architecture, please turn it off)
    # Cudnn函式庫輔助加速(如遇到架構上無法配合請予以關閉)
    cudnn.enabled = True

    # Model import 模型導入
    model = network_model.Net()

    c = Calculate(model)
    model_size = c.get_model_size()
    flops, params = c.get_params()

    model, device = check_number_of_GPUs(model, args)

    set_save_dir_names(args)

    # 先用Adam測試模型能力
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(args["learning_rate"]), weight_decay=float(args["weight_decay"])
    )

    # model = torch.compile(model)  #pytorch2.0編譯功能(舊GPU無法使用)

    start_epoch = 1

    if args["wandb_name"] != "no":
        wandb_information(model_name, model_size, flops, params, model, train_images, train_masks,args)

    if args["resume"]:
        if os.path.isfile(
            args["resume"]
        ):  # There is a specified file in the path 路徑中有指定檔案
            if args["wandb_name"] != "no":
                checkpoint = torch.load(wandb.restore(args["resume"]).name)
            else:
                checkpoint = torch.load(args["resume"])
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            mean_loss = checkpoint["loss"]
            mean_miou = checkpoint["miou"]
            save_mean_miou = checkpoint["best_miou"]
            print(
                "=====> load checkpoint '{}' (epoch {})".format(
                    args["resume"], checkpoint["epoch"]
                )
            )
        else:
            print("=====> no checkpoint found at '{}'".format(args["resume"]))

    if not os.path.exists("./training_data_captures/"):
        os.makedirs("./training_data_captures/")
    if not os.path.exists("./validation_data_captures/"):
        os.makedirs("./validation_data_captures/")

    time_start = time.time()  # Training start time 訓練開始時間

    for epoch in range(start_epoch, args["epochs"] + 1):
        train_RGB_image, train_mask_image, train_output = train_epoch(
            model, training_data_loader, device, optimizer, epoch, args
        )

        torch.cuda.empty_cache()  # 刪除不需要的變數

        (
            mean_loss,
            mean_miou,
            RGB_image,
            mask_image,
            output,
            onnx_img_image,
        ) = valid_epoch(model, validation_data_loader, device, epoch, args)
        

        state = create_model_state_dict(args, epoch, model, optimizer, mean_loss, mean_miou, save_mean_miou)
        save_model_and_state(model, state,  mean_loss, mean_miou, onnx_img_image,args["save_dir"], "last", args)
        save_experiment_details(args, model_name, train_images, train_masks)


        if args["save_train_image"] != "no":
            save_and_log_image(train_RGB_image, "./training_data_captures", "train_RGB_image", args)
            save_and_log_image(train_mask_image, "./training_data_captures", "train_mask_image", args)
            save_and_log_image(train_output, "./training_data_captures", "train_output", args)

        if args["save_validation_image_last"] != "no":
            save_and_log_image(RGB_image, "./validation_data_captures", "last_RGB_image", args)
            save_and_log_image(mask_image, "./validation_data_captures", "last_mask_image", args)
            save_and_log_image(output, "./validation_data_captures", "last_output", args)


        if mean_miou > save_mean_miou:
            print("best_loss: %.3f , best_miou: %.3f" % (mean_loss, mean_miou))
            save_model_and_state(model, state,  mean_loss, mean_miou, onnx_img_image,args["save_dir"], "best", args)
            
            if args["save_validation_image_best"] != "no":
                save_and_log_image(RGB_image, "./validation_data_captures", "best_RGB_image", args)
                save_and_log_image(mask_image, "./validation_data_captures", "best_mask_image", args)
                save_and_log_image(output, "./validation_data_captures", "best_output", args)

            save_mean_miou = mean_miou

    time_end = time.time()
    spend_time = int(time_end - time_start)
    time_dict = time_processing(spend_time)
    print(
        "totally cost:",
        f"{time_dict['time_day']}d {time_dict['time_hour']}h {time_dict['time_min']}m {time_dict['time_sec']}s",
    )

if __name__ == "__main__":
    args = parse_arguments()
    main()
    wandb.finish()
