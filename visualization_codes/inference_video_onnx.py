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
from typing import Union

from utils.inference_onnx import smoke_semantic
import visualization_codes.utils.image_process as image_process


def save(video_W: int, video_H: int, video_FPS: float) -> cv2.VideoWriter:
    os.makedirs("./results/process_video/", exist_ok=True)
    save_file_name = time.strftime("%Y-%m-%d_%I:%M:%S_%p", time.localtime())
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(
        f"./results/process_video/{save_file_name}.mp4", fourcc, video_FPS, (video_W, video_H)
    )
    return output


def image_pre_processing(input:np.ndarray) -> np.ndarray:
    output = cv2.resize(input, (256, 256), interpolation=cv2.INTER_AREA)
    output = np.array(output).astype(np.float32) / 255.0
    output = np.transpose(output, (2, 0, 1))
    output = np.expand_dims(output, axis=0)
    return output

# Main function 主函式
def smoke_segmentation(
    video_path: Union[str, int],
    model: str,
    ort_session:ort.InferenceSession,
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
    time_list = []
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        video_frame = image_pre_processing(frame)
        output = smoke_semantic(video_frame, ort_session, time_train, i)
        output = (output[0, 0] > 0.5).astype(np.uint8) * 255
        # use opencv method

        output = cv2.resize(
            output, (video_W, video_H), interpolation=cv2.INTER_AREA
        )  # 插值
        frame = frame.astype(np.int32)

        if overlap_image == True:
            overlapImage = image_process.overlap_v3(
                frame, output, read_method="OpenCV_BGRA"
            )

        print("process_time: ", time.time() - start_time)
        print("FPS: ", 1 / (time.time() - start_time))

        time_list.append(time.time() - start_time)

        if save_video:
            out.write(overlapImage)

        if show_video:
            cv2.imshow("frame", overlapImage)
            # cv2.imshow('frame1',frame)

            if cv2.waitKey(1) == ord("q"):
                break
        i += 1
    
    print("Average FPS: ", len(time_list[1:]) / sum(time_list[1:]))
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

    # 檢查是否有可用的 CUDA 裝置
    def is_cuda_available():
        providers = ort.get_available_providers()
        return 'CUDAExecutionProvider' in providers

    # 設定 ONNX Runtime 的執行提供者
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if is_cuda_available() else ['CPUExecutionProvider']

    # 列印當前使用的裝置
    device = "cuda" if is_cuda_available() else "cpu"
    print(f"inference_multiple_Dataset on device {device}.")
    
    model = args["model_path"]
    ort_session = ort.InferenceSession(model, providers=providers)

    save_video = True
    smoke_segmentation(
        args["video_source"], args["model_path"], ort_session, save_video
    )
