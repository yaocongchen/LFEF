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
import shutil
#定位到主目錄
import sys
sys.path.append("..")

import models.CGNet_2_erfnet31_13_3113_oneloss_inv_attention as network_model

from utils.inference import smoke_semantic
# import visualization_codes.process_utils_cython_version.image_process_utils_cython as image_process

import visualization_codes.utils.image_process as image_process


def create_directory(dir_name):
    path = f"./results/video_to_frames/{dir_name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def folders_and_files_name():
    dir_names = ["RGB_original_image", "segmentation_image", "overlap_image"]
    for dir_name in dir_names:
        create_directory(dir_name)

    names = {
        "video_RGB_original_dir_name": dir_names[0],
        "video_segmentation_dir_name": dir_names[1],
        "video_overlap_dir_name": dir_names[2],
        "video_capture_image_name": "capture",
    }

    return names


def save(video_W: int, video_H: int, video_FPS):
    os.makedirs("./results/video_to_frames", exist_ok=True)
    save_file_name = time.strftime("%Y-%m-%d_%I:%M:%S_%p", time.localtime())
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(
        f"./results/video_to_frames/{save_file_name}.mp4", fourcc, video_FPS, (video_W, video_H)
    )
    return output


def image_pre_processing(input, device):
    process_frame = torch.from_numpy(input).to(device)
    process_frame = process_frame.permute(2, 0, 1).contiguous()
    transform = transforms.Resize([256, 256], antialias=True)  # 插值
    process_frame = transform(process_frame)
    process_frame = process_frame / 255.0
    output = process_frame.unsqueeze(0)

    return output


# Main function 主函式
def smoke_segmentation(
    video_path: str,
    model_input,
    device: torch.device,
    names: dict,
    overlap_image: bool,
    time_train,
    i,
):
    i = 0
    print("overlap_image:", overlap_image)

    start_time = time.time()
    counter = 0

    if video_path == "0":
        video_path = int(video_path)
    cap = cv2.VideoCapture(video_path)

    # 設定擷取影像的尺寸大小
    video_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_FPS = cap.get(cv2.CAP_PROP_FPS)
    # print(cv2.getBuildInformation())
    # Define the codec and create VideoWriter object

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        counter += 1

        cv2.imwrite(
            f'./results/video_to_frames/{names["video_RGB_original_dir_name"]}/{names["video_capture_image_name"]}_{i}.png',
            frame,
        )

        video_frame = image_pre_processing(frame, device)
        output, aux = smoke_semantic(video_frame, model_input, device, time_train, i)
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
        cv2.imwrite(
            f'./results/video_to_frames/{names["video_segmentation_dir_name"]}/{names["video_capture_image_name"]}_{i}.png',
            output_np,
        )

        frame = frame.astype(np.int32)

        if overlap_image == True:
            overlapImage = image_process.overlap_v3(
                frame, output_np, read_method="OpenCV_BGRA"
            )

        cv2.imwrite(
            f'./results/video_to_frames/{names["video_overlap_dir_name"]}/{names["video_capture_image_name"]}_{i}.png',
            overlapImage,
        )
        print("process_time: ", time.time() - start_time)
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()

        i += 1

    # ====================================================
    # Release everything if job is finished
    cap.release()
    # if save_video == "True":
    #     out.release()
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-vs",
        "--video_source",
        type=str,
        default="/home/yaocong/Experimental/Dataset/smoke_video_dataset/Black_smoke_517.avi",
        required=False,
        help="path to test video path",
    )
    ap.add_argument(
        "-m",
        "--model_path",
        default="/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/CGNet_best.pth",
        required=False,
        help="load model path",
    )
    args = vars(ap.parse_args())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"video on device {device}.")

    i = 0
    time_train = []
    names = folders_and_files_name()

    overlap_image = True
    # save_video = True

    model = network_model.Net().to(device)
    model = torch.compile(model)
    torch.set_float32_matmul_precision('high')
    model.load_state_dict(torch.load(args["model_path"], map_location=device))

    model.eval()

    smoke_segmentation(
        args["video_source"],
        model,
        device,
        names,
        overlap_image,
        time_train,
        i,
    )
