import torch
import argparse
import os
import time

from visualization_codes import (
    inference_single_picture,
    inference_multiple_pictures,
    inference_video,
)
import models.CGNet_2_erfnet31_13_3113_oneloss_6ch as network_model  # import self-written models 引入自行寫的模型


def smoke_segmentation(args,device):
    model = network_model.Net().to(device)
    model.load_state_dict(torch.load(args["model_path"], map_location=device))

    model.eval()

    source = args["source"]

    time_train = []

    i = 0

    if os.path.isdir(source):
        names = inference_multiple_pictures.folders_and_files_name()

        total_image = len(os.listdir(args["source"]))
        time_start = time.time()
        inference_multiple_pictures.smoke_segmentation(
            args["source"], model, device, names, time_train, i
        )
        time_end = time.time()
        spend_time = int(time_end - time_start)
        time_min = spend_time // 60
        time_sec = spend_time % 60
        print("totally cost:", f"{time_min}m {time_sec}s")
        # print(total_image)

        # Calculate FPS
        print("folder_process_FPS:{:.1f}".format(total_image / (time_end - time_start)))

    else:
        root, extension = os.path.splitext(source)

        if extension in [".jpg", ".png"]:
            names = inference_single_picture.files_name()
            inference_single_picture.smoke_segmentation(
                args["source"], model, device, names, time_train, i
            )
        elif extension in [".mp4", ".avi"]:
            binary_mode = True
            blend_image = True
            inference_video.smoke_segmentation(
                args["source"],
                model,
                device,
                binary_mode,
                blend_image,
                args["save_video"],
                args["show_video"],
                time_train,
                i,
            )
        elif root in ["0"]:  # camera
            binary_mode = True
            blend_image = False
            inference_video.smoke_segmentation(
                args["source"],
                model,
                device,
                binary_mode,
                blend_image,
                args["save_video"],
                args["show_video"],
                time_train,
                i,
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-s",
        "--source",
        type=str,
        default="/home/yaocong/Experimental/speed_smoke_segmentation/test_files/Dry_leaf_smoke_02.avi",
        required=False,
        help="path to test video path",
    )
    ap.add_argument(
        "-m",
        "--model_path",
        default="/home/yaocong/Experimental/speed_smoke_segmentation/checkpoint/bs8e150/final.pth",
        required=False,
        help="load model path",
    )
    ap.add_argument(
        "-save",
        "--save_video",
        type=str,
        default="True",
        required=False,
        help="save video",
    )  # argparse.ArgumentParser()無法辨識boolean
    ap.add_argument(
        "-show",
        "--show_video",
        type=str,
        default="True",
        required=False,
        help="save video",
    )
    args = vars(ap.parse_args())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"smoke_segmentation on device {device}.")

    smoke_segmentation(args,device)
