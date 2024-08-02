import os
import argparse
import time
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
from typing import Dict, Tuple, Union
import onnxruntime as ort

from utils.inference_onnx import smoke_semantic

import visualization_codes.utils.image_process as image_process

def create_directory(dir_name: str) -> None:
    path = f"./results/process_folder/{dir_name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def folders_and_files_name() -> Dict[str, str]:
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

def load_and_process_image(image_path: str, size: Tuple[int, int] = (256, 256)) -> Image:
    img = Image.open(image_path)
    img = img.resize(size)
    img = ImageOps.expand(img, 20, "#ffffff")
    return img

# Merge all resulting images 合併所有產生之圖像
def image_stitching(input_image: str, i: int, names: Dict[str, str]) -> None:
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
def image_overlap(input_image: str, i: int, names: Dict[str, str]) -> None:
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

def read_image(file_path: str) -> np.ndarray:
    image = Image.open(file_path)
    image = image.resize((256, 256))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# Main function 主函式
def smoke_segmentation(
    directory: str, model: str, ort_session:ort.InferenceSession, names: Dict[str, str], time_train: float, i: int
) -> None:

    i = 0
    pbar = tqdm((os.listdir(directory)), total=len(os.listdir(directory)))
    for filename in pbar:
        smoke_input_image = read_image(os.path.join(directory, filename))
        # print(smoke_input_image.shape)

        output = smoke_semantic(smoke_input_image, ort_session, time_train, i)

        output = (output > 0.5).astype(np.float32)

        i += 1
        output_image = Image.fromarray(output[0, 0] * 255).convert("L")
        output_image.save(f'./results/process_folder/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_image_name"]}_{i}.jpg')
        image_overlap(os.path.join(directory, filename), i, names)
        image_stitching(os.path.join(directory, filename), i, names)

    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-td", 
        "--test_directory", 
        default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/images/",
        required=True, 
        help="Path to the directory containing test images."
    )
    ap.add_argument(
        "-m", 
        "--model_path", 
        default="/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/best.onnx",
        required=True, 
        help="Path to the trained model to be used for inference."
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

    names = folders_and_files_name()
    # print(names)
    # Calculate the total execution time 計算總執行時間
    time_train = []

    i = 0
    time_start = time.time()
    smoke_segmentation(args["test_directory"], args["model_path"], providers, names, time_train, i)
    total_image = len(os.listdir(args["test_directory"]))
    time_end = time.time()
    spend_time = int(time_end - time_start)
    time_min = spend_time // 60
    time_sec = spend_time % 60
    print("totally cost:", f"{time_min}m {time_sec}s")
    # print(total_image)

    # Calculate FPS
    print("FPS:{:.1f}".format(total_image / (time_end - time_start)))
