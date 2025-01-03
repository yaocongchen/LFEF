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
from typing import Dict, List, Any, Tuple

import models.LFEF as network_model  # import self-written models 引入自行寫的模型
from utils.main_setup_utils import folders_and_files_name, set_model_save_dir_names, create_model_state_dict, time_processing, wandb_information, parse_arguments
from utils.check_GPU import check_have_GPU, check_number_of_GPUs, set_seed
from utils.metrics import Calculate
from utils.data_processing import data_processing_train_8_val_DS01
from utils.training import train_epoch, valid_epoch
from utils.results_saving import save_model_and_state, save_and_log_image, save_experiment_details


onnx_img_image = []

model_name = str(network_model)
print("model_name:", model_name)

seed = 42


def main(args: Dict[str, Any], names: Dict[str, str]) -> None:
    train_images, train_masks, training_data_loader, validation_data_loader = data_processing_train_8_val_DS01(args)

    save_mean_miou = 0
    set_seed(seed)
    check_have_GPU(args)
    
    cudnn.enabled = True  # The cudnn function library assists in acceleration(if you encounter a problem with the architecture, please turn it off)

    model = network_model.Net()

    c = Calculate(model)
    model_size = c.get_model_size()
    flops, params = c.get_params()

    model, device = check_number_of_GPUs(args, model)

    set_model_save_dir_names(args)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(args["learning_rate"]), weight_decay=float(args["weight_decay"])
    )

    # model = torch.compile(model)  #pytorch2.0編譯功能(舊GPU無法使用)
    # torch.set_float32_matmul_precision('high')
    
    start_epoch = 1

    if args["wandb_name"] != "no":
        wandb_information(args, model_name, model_size, flops, params, model, train_images, train_masks)

    if args["resume"]:
        if os.path.isfile(
            args["resume"]
        ):
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
        print("=====> Resume from checkpoint not used")

    time_start = time.time()

    for epoch in range(start_epoch, args["epochs"] + 1):
        train_RGB_image, train_mask_image, train_output = train_epoch(
            args, model, training_data_loader, device, optimizer, epoch
        )

        torch.cuda.empty_cache()

        (
            mean_loss,
            mean_miou,
            RGB_image,
            mask_image,
            output,
            onnx_img_image,
        ) = valid_epoch(args, model, validation_data_loader, device, epoch)
        

        state = create_model_state_dict(epoch, model, optimizer, mean_loss, mean_miou, save_mean_miou)
        save_model_and_state(args, model, state,  mean_loss, mean_miou, onnx_img_image,args["model_save_dir"], "last")
        save_experiment_details(args, model_name, train_images, train_masks)


        if args["save_train_image"]:
            save_and_log_image(args, train_RGB_image, names["training_data_captures_dir_name"], "train_RGB_image")
            save_and_log_image(args, train_mask_image, names["training_data_captures_dir_name"], "train_mask_image")
            save_and_log_image(args, train_output, names["training_data_captures_dir_name"], "train_output")
        
        if args["save_validation_image_last"]:
            save_and_log_image(args, RGB_image, names["validation_data_captures_dir_name"], "last_RGB_image")
            save_and_log_image(args, mask_image, names["validation_data_captures_dir_name"], "last_mask_image")
            save_and_log_image(args, output, names["validation_data_captures_dir_name"], "last_output")
        
        if mean_miou > save_mean_miou:
            print("best_loss: %.3f , best_miou: %.3f" % (mean_loss, mean_miou))
            save_model_and_state(args, model, state,  mean_loss, mean_miou, onnx_img_image,args["model_save_dir"], "best")
            
            if args["save_validation_image_best"]:
                save_and_log_image(args, RGB_image, names["validation_data_captures_dir_name"], "best_RGB_image")
                save_and_log_image(args, mask_image, names["validation_data_captures_dir_name"], "best_mask_image")
                save_and_log_image(args, output, names["validation_data_captures_dir_name"], "best_output")
        
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
    names = folders_and_files_name(args)
    main(args, names)
    wandb.finish()
