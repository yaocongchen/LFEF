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

from utils.inference import smoke_semantic

import visualization_codes.utils.image_process as image_process
# import visualization_codes.process_utils_cython_version.image_process_utils_cython as image_process

def save(video_W: int, video_H: int, video_FPS):
    if not os.path.exists("./" + "results"):
        os.makedirs("./" + "results")
    localtime = time.localtime()
    save_file_name = time.strftime("%Y-%m-%d_%I:%M:%S_%p", localtime)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(
        f"./results/{save_file_name}.mp4", fourcc, video_FPS, (video_W, video_H), 3
    )  # mp4 only RGB
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
    model: str,
    device: torch.device,
    binary_mode: bool,
    blend_image: bool,
    save_video: str,
    show_video: str,
    time_train,
    i,
):
    print("binary_mode:", binary_mode)
    print("blend_image:", blend_image)
    print("save_video:", save_video)
    print("show_video:", show_video)

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
    if save_video == "True":
        out = save(video_W, video_H, video_FPS)

    idx = 0
    freq = 5
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        counter += 1

        video_frame = image_pre_processing(frame, device)
        output = smoke_semantic(video_frame, model, device, time_train, i)
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
        output_np = Image.fromarray(output_np)

        # use torchvision method
        # output_np = output.squeeze(0).mul(255).add_(0.5).clamp_(0, 255)
        # transform = transforms.Resize([video_H,video_W],antialias=True)    #插值
        # output_np = transform(output_np)

        # PILtransform = transforms.ToPILImage()
        # output_np = PILtransform(output_np)

        # output_np to binarization output_np轉二值化
        binary_image = image_process.gray_to_binary(output_np)

        if binary_mode == True:
            output_np_RGBA = binary_image.convert("RGBA")
        else:
            output_np_RGBA = output_np.convert("RGBA")

        frame_image = Image.fromarray(frame)
        frame_RGBA = frame_image.convert("RGBA")

        if blend_image == True:
            blendImage = image_process.overlap_PIL(
                frame_RGBA, output_np_RGBA, read_method="OpenCV_BGRA"
            )
            output_np = blendImage.convert("RGB")
        output_np = np.asarray(output_np)

        print("process_time: ", time.time() - start_time)
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()

        if save_video == "True":
            out.write(output_np)

        if show_video == "True":
            cv2.imshow("frame", output_np)
            # cv2.imshow('frame1',frame)

            if cv2.waitKey(1) == ord("q"):
                break
        i += 1

        # 5fps samples ====================================================

        # idx += 1
        # ret = cap.grab()

        # # ret,frame = cap.read()
        # # if frame is read correctly ret is True
        # if not ret:
        #     print("Can't receive frame (stream end?). Exiting ...")
        #     break

        # if idx % freq == 1:
        #     ret, frame = cap.retrieve()
        #     if frame is None:  # exist broken frame
        #         break
        #     else:
        #         counter += freq

        #         video_frame = image_pre_processing(frame, device)
        #         output = smoke_semantic(video_frame, model_input, device, time_train, i)
        #         # use opencv method
        #         output_np = (
        #             output.squeeze(0)
        #             .mul(255)
        #             .add_(0.5)
        #             .clamp_(0, 255)
        #             .permute(1, 2, 0)
        #             .contiguous()
        #             .to("cpu", torch.uint8)
        #             .detach()
        #             .numpy()
        #         )
        #         output_np = cv2.resize(
        #             output_np, (video_W, video_H), interpolation=cv2.INTER_AREA
        #         )  # 插值
        #         output_np = Image.fromarray(output_np)

        #         # output_np to binarization output_np轉二值化
        #         binary_image = image_process.gray_to_binary(output_np)

        #         if binary_mode == True:
        #             output_np_RGBA = binary_image.convert("RGBA")
        #         else:
        #             output_np_RGBA = output_np.convert("RGBA")

        #         frame_image = Image.fromarray(frame)
        #         frame_RGBA = frame_image.convert("RGBA")

        #         if blend_image == True:
        #             blendImage = image_process.overlap_PIL(
        #                 frame_RGBA, output_np_RGBA, read_method="OpenCV_BGRA"
        #             )
        #             output_np = blendImage.convert("RGB")
        #         output_np = np.asarray(output_np)

        #         print("FPS: ", counter / (time.time() - start_time))
        #         counter = 0
        #         start_time = time.time()

        #         if save_video == "True":
        #             out.write(output_np)

        #         if show_video == "True":
        #             cv2.imshow("frame", output_np)
        #             # cv2.imshow('frame1',frame)

        #             if cv2.waitKey(1) == ord("q"):
        #                 break
        #     i += 1
    # ====================================================
    # Release everything if job is finished
    cap.release()
    if save_video == "True":
        out.release()
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-vs",
        "--video_source",
        type=str,
        default="/home/yaocong/Experimental/speed_smoke_segmentation/test_files/Dry_leaf_smoke_02.avi",
        required=True,
        help="path to test video path",
    )
    ap.add_argument(
        "-m",
        "--model_path",
        default="/home/yaocong/Experimental/speed_smoke_segmentation/checkpoint/bs8e150/final.pth",
        required=True,
        help="load model path",
    )
    args = vars(ap.parse_args())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"video on device {device}.")

    binary_mode = False
    print("binary_mode:", binary_mode)

    save_video = True
    smoke_segmentation(
        args["video_source"], args["model_path"], device, binary_mode, save_video
    )
