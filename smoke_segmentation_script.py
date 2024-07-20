import torch
import argparse
import os
import time
from typing import Dict, Any

from visualization_codes import (
    inference_single_picture,
    inference_multiple_pictures,
    inference_video,
    inference_video_to_frames,
)
from utils.metrics import report_fps_and_time

def smoke_segmentation(args: Dict[str, Any], device: torch.device) -> None:
    model = torch.jit.load(args["model_path"])
    model.to(device)

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
        fps, time_min, time_sec = report_fps_and_time(total_image, time_start, time_end)

    else:
        root, extension = os.path.splitext(source)

        if extension in [".jpg", ".png"]:
            names = inference_single_picture.files_name()
            inference_single_picture.smoke_segmentation(
                args["source"], model, device, names, time_train, i
            )
        elif extension in [".mp4", ".avi"]:
            overlap_image = True
            if args["video_to_frames"]:
                names = inference_video_to_frames.folders_and_files_name()
                inference_video_to_frames.smoke_segmentation(
                    args["source"],
                    model,
                    device,
                    names,
                    overlap_image,
                    time_train,
                    i,
                )
            else:
                inference_video.smoke_segmentation(
                    args["source"],
                    model,
                    device,
                    overlap_image,
                    args["save_video"],
                    args["show_video"],
                    time_train,
                    i,
                )
        elif root in ["0"]:  # camera
            overlap_image = True
            if args["video_to_frames"]:
                names = inference_video_to_frames.folders_and_files_name()
                inference_video_to_frames.smoke_segmentation(
                    args["source"],
                    model,
                    device,
                    names,
                    overlap_image,
                    time_train,
                    i,
                )
            else:
                inference_video.smoke_segmentation(
                    args["source"],
                    model,
                    device,
                    overlap_image,
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
        default="/home/yaocong/Experimental/Dataset/smoke_video_dataset/Black_smoke_517.avi",
        type=str,
        required=True,
        help="Path to the image, video file, or directory to be tested.",
    )
    ap.add_argument(
        "-m",
        "--model_path",
        default="./results/torch_script/model.pt",
        required=True,
        help="Path to the trained model to be used for smoke segmentation.",
    )
    ap.add_argument(
        "-vtf",
        "--video_to_frames",
        default=False,
        action='store_true',
        help="Convert the video to frames. Include this argument to enable this feature.",
    )
    ap.add_argument(
        "-save",
        "--save_video",
        default=False,
        action='store_true',
        help="Save the output video. Include this argument to enable this feature.",
    )
    ap.add_argument(
        "-show",
        "--show_video",
        default=True,
        action='store_true',
        help="Display the output video. Include this argument to enable this feature.",
    )
    args = vars(ap.parse_args())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"smoke_segmentation on device {device}.")

    smoke_segmentation(args,device)
