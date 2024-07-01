import cv2
import torch
import argparse
import time
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import threading
from copy import deepcopy
from typing import Union

from utils.inference import smoke_semantic
import visualization_codes.utils.image_process as image_process


def save(video_W: int, video_H: int, video_FPS: float) -> cv2.VideoWriter:
    os.makedirs("./results/process_video/", exist_ok=True)
    save_file_name = time.strftime("%Y-%m-%d_%I:%M:%S_%p", time.localtime())
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(
        f"./results/process_video/{save_file_name}.mp4", fourcc, video_FPS, (video_W, video_H)
    )
    return output


def image_pre_processing(input: np.ndarray, device: torch.device) -> torch.Tensor:
    process_frame = torch.from_numpy(input).to(device)
    process_frame = process_frame.permute(2, 0, 1).contiguous()
    transform = transforms.Resize([256, 256], antialias=True)  # 插值
    process_frame = transform(process_frame)
    process_frame = process_frame / 255.0
    output = process_frame.unsqueeze(0)

    return output


# Main function 主函式
def smoke_segmentation(
    video_path: Union[str, int],
    model: str,
    device: torch.device,
    overlap_image: bool,
    save_video: bool,
    show_video: bool,
    time_train: float,
    i: int
) -> None:
    print("overlap_image:", overlap_image)
    print("save_video:", save_video)
    print("show_video:", show_video)

    if video_path == "0":
        video_path = int(video_path)
    cap = cv2.VideoCapture(video_path)
    
    # 設定擷取影像的尺寸大小
    video_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_FPS = cap.get(cv2.CAP_PROP_FPS)
    # print(cv2.getBuildInformation())
    # Define the codec and create VideoWriter object
    if save_video:
        out = save(video_W, video_H, video_FPS)

    idx = 0
    freq = 5
    counter = 0
    start_time_avg = time.time()
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        counter += 1

        video_frame = image_pre_processing(frame, device)
        output, aux = smoke_semantic(video_frame, model, device, time_train, i)
        output = (output > 0.5).float()
        # use opencv method
        output_np = (
            output.squeeze(0)
            .mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .contiguous()
            .to("cpu", torch.uint8)
            .detach()
            .numpy()
        )
        output_np = cv2.resize(
            output_np, (video_W, video_H), interpolation=cv2.INTER_AREA
        )  # 插值
        frame = frame.astype(np.int32)

        if overlap_image == True:
            overlapImage = image_process.overlap_v3(
                frame, output_np, read_method="OpenCV_BGRA"
            )

        print("process_time: ", time.time() - start_time)
        print("FPS: ", 1 / (time.time() - start_time))


        if save_video:
            out.write(overlapImage)

        if show_video:
            cv2.imshow("frame", overlapImage)
            # cv2.imshow('frame1',frame)

            if cv2.waitKey(1) == ord("q"):
                break
        i += 1
    print("Average FPS: ", counter / (time.time() - start_time_avg))
    # ====================================================
    # Release everything if job is finished
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-vs",
        "--video_source",
        type=str,
        required=True,
        help="Path to the video file to be tested.",
    )
    ap.add_argument(
        "-m",
        "--model_path",
        required=True,
        help="Path to the trained model to be used for inference.",
    )
    args = vars(ap.parse_args())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"video on device {device}.")

    save_video = True
    smoke_segmentation(
        args["video_source"], args["model_path"], device, save_video
    )
