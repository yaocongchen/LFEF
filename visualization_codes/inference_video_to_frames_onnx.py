import cv2
import onnxruntime as ort
import argparse
import time
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import threading
from copy import deepcopy
import shutil
from typing import Dict, Union
#定位到主目錄
import sys
sys.path.append("..")

import models.LFEF as network_model
from utils.inference_onnx import smoke_semantic
import visualization_codes.utils.image_process as image_process


def create_directory(dir_name: str) -> None:
    path = f"./results/video_to_frames/{dir_name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def folders_and_files_name() -> Dict[str, str]:
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


def save(video_W: int, video_H: int, video_FPS: float) -> cv2.VideoWriter:
    os.makedirs("./results/video_to_frames", exist_ok=True)
    save_file_name = time.strftime("%Y-%m-%d_%I:%M:%S_%p", time.localtime())
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(
        f"./results/video_to_frames/{save_file_name}.mp4", fourcc, video_FPS, (video_W, video_H)
    )
    return output


def image_pre_processing(input:np.ndarray) -> np.ndarray:
    output = cv2.resize(input, (256, 256), interpolation=cv2.INTER_AREA)
    output = np.array(output).astype(np.float32) / 255.0
    output = np.transpose(output, (2, 0, 1))
    output = np.expand_dims(output, axis=0)
    return output


def smoke_segmentation(
    video_path: Union[str, int],
    model: str,
    ort_session:ort.InferenceSession,
    names: Dict[str, str],
    overlap_image: bool,
    time_train: float,
    i: int
) -> None:
    i = 0
    print("overlap_image:", overlap_image)

    if video_path == "0":
        video_path = int(video_path)
    cap = cv2.VideoCapture(video_path)

    # 設定擷取影像的尺寸大小
    video_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_FPS = cap.get(cv2.CAP_PROP_FPS)
    # print(cv2.getBuildInformation())
    # Define the codec and create VideoWriter object
    time_list = []
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imwrite(
            f'./results/video_to_frames/{names["video_RGB_original_dir_name"]}/{names["video_capture_image_name"]}_{i}.png',
            frame,
        )

        video_frame = image_pre_processing(frame)
        output = smoke_semantic(video_frame, ort_session, time_train, i)
        output = (output[0, 0] > 0.5).astype(np.uint8) * 255
        # use opencv method

        output = cv2.resize(
            output, (video_W, video_H), interpolation=cv2.INTER_AREA
        )  # 插值
        cv2.imwrite(
            f'./results/video_to_frames/{names["video_segmentation_dir_name"]}/{names["video_capture_image_name"]}_{i}.png',
            output,
        )

        frame = frame.astype(np.int32)

        if overlap_image == True:
            overlapImage = image_process.overlap_v3(
                frame, output, read_method="OpenCV_BGRA"
            )

        cv2.imwrite(
            f'./results/video_to_frames/{names["video_overlap_dir_name"]}/{names["video_capture_image_name"]}_{i}.png',
            overlapImage,
        )
        print("process_time: ", time.time() - start_time)
        print("FPS: ", 1 / (time.time() - start_time))
        time_list.append(time.time() - start_time)
        
        i += 1
    print("Average FPS: ", len(time_list[1:]) / sum(time_list[1:]))
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

    i = 0
    time_train = []
    names = folders_and_files_name()

    overlap_image = True
    # save_video = True

    model = network_model.Net().to(device)
    # model = torch.compile(model)
    # torch.set_float32_matmul_precision('high')
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
