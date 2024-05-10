import torch
import torchvision
import os
import argparse
import time
import shutil
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image, ImageOps

from utils.inference import smoke_semantic

import visualization_codes.utils.image_process as image_process


def create_directory(dir_name):
    path = f"./results/process_folder/{dir_name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def folders_and_files_name():
    dir_names = ["multiple_result", "multiple_overlap", "multiple_stitching"]
    image_names = ["smoke_semantic_image", "image_overlap", "image_stitching"]

    for dir_name in dir_names:
        create_directory(dir_name)

    names = {
        "smoke_semantic_dir_name": dir_names[0],
        "smoke_semantic_image_name": image_names[0],
        "image_overlap_dir_name": dir_names[1],
        "image_overlap_name": image_names[1],
        "image_stitching_dir_name": dir_names[2],
        "image_stitching_name": image_names[2],
    }

    return names

def load_and_process_image(image_path, size=(256, 256)):
    img = Image.open(image_path)
    img = img.resize(size)
    img = ImageOps.expand(img, 20, "#ffffff")
    return img

# Merge all resulting images 合併所有產生之圖像
def image_stitching(input_image, i, names):
    bg = Image.new("RGB", (900, 300), "#000000")
    image_paths = [
        input_image,
        f'./results/process_folder/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_image_name"]}_{i}.jpg',
        f'./results/process_folder/{names["image_overlap_dir_name"]}/{names["image_overlap_name"]}_{i}.png'
    ]

    for idx, image_path in enumerate(image_paths):
        img = load_and_process_image(image_path)
        bg.paste(img, (300 * idx, 0))

    bg.save(f'./results/process_folder/{names["image_stitching_dir_name"]}/{names["image_stitching_name"]}_{i}.jpg')

    return

# The trained feature map is fuse d with the original image 訓練出的特徵圖融合原圖
def image_overlap(input_image, i, names):
    img1 = Image.open(input_image)
    img2 = Image.open(
        f'./results/process_folder/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_image_name"]}_{i}.jpg'
    )

    img1 = image_process.process_pil_to_np(img1, gray=False)
    img2 = image_process.process_pil_to_np(img2, gray=True)
    
    blendImage = image_process.overlap_v3(img1, img2, read_method="PIL_RGBA")

    # Display image 顯示影像
    # blendImage.show()
    Image.fromarray(blendImage).save(
        f'./results/process_folder/{names["image_overlap_dir_name"]}/{names["image_overlap_name"]}_{i}.png'
    )

    return


# Main function 主函式
def smoke_segmentation(
    directory: str, model: str, device: torch.device, names: dict, time_train, i
):
    i = 0
    pbar = tqdm((os.listdir(directory)), total=len(os.listdir(directory)))
    for filename in pbar:
        smoke_input_image = read_image(os.path.join(directory, filename)).to(device)
        transform = transforms.Resize([256, 256],antialias=True)
        smoke_input_image = transform(smoke_input_image)
        smoke_input_image = (smoke_input_image) / 255.0
        smoke_input_image = smoke_input_image.unsqueeze(0).to(device)
        output, aux = smoke_semantic(smoke_input_image, model, device, time_train, i)
        output = (output > 0.5).float()
        i += 1
        torchvision.utils.save_image(
            output,
            f'./results/process_folder/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_image_name"]}_{i}.jpg',
        )
        image_overlap(os.path.join(directory, filename), i, names)
        image_stitching(os.path.join(directory, filename), i, names)

    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-td", "--test_directory", required=True, help="path to test images directory"
    )
    ap.add_argument("-m", "--model_path", required=True, help="load model path")
    args = vars(ap.parse_args())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"inference_multiple_Dataset on device {device}.")

    names = folders_and_files_name()
    print(names)
    # Calculate the total execution time 計算總執行時間
    time_start = time.time()
    smoke_segmentation(args["test_directory"], args["model_path"], device, names)
    total_image = len(os.listdir(args["test_directory"]))
    time_end = time.time()
    spend_time = int(time_end - time_start)
    time_min = spend_time // 60
    time_sec = spend_time % 60
    print("totally cost:", f"{time_min}m {time_sec}s")
    # print(total_image)

    # Calculate FPS
    print("FPS:{:.1f}".format(total_image / (time_end - time_start)))
