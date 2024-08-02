import onnxruntime as ort
import argparse
import os
from PIL import Image, ImageOps
from typing import Dict, Tuple, Union
import numpy as np

from utils.inference_onnx import smoke_semantic
import visualization_codes.utils.image_process as image_process


def files_name() -> Dict[str, str]:
    # Set archive name 設定存檔名稱
    os.makedirs("./results/process_image", exist_ok=True)
    save_smoke_semantic_image_name = "./results/process_image/smoke_semantic"
    save_image_overlap_name = "./results/process_image/image_overlap"
    save_image_stitching_name = "./results/process_image/image_stitching"

    names = {}
    names["smoke_semantic_image_name"] = save_smoke_semantic_image_name
    names["image_overlap_name"] = save_image_overlap_name
    names["image_stitching_name"] = save_image_stitching_name

    return names

def load_and_process_image(input_image: Union[str, np.ndarray], size: Tuple[int, int] = (256, 256)) -> Image:
    if isinstance(input_image, str):
        img = Image.open(input_image)
    else:
        img = Image.fromarray(input_image)
    img = img.resize(size)
    img = ImageOps.expand(img, 20, "#ffffff")
    return img

# Merge all resulting images 合併所有產生之圖像
def image_stitching(input_image: Union[str, np.ndarray], names: Dict[str, str]) -> None:
    bg = Image.new("RGB", (900, 300), "#000000")
    img1 = load_and_process_image(input_image)
    img2 = load_and_process_image(names["smoke_semantic_image_name"] + ".jpg")
    img3 = load_and_process_image(names["image_overlap_name"] + ".png")

    bg.paste(img1, (0, 0))
    bg.paste(img2, (300, 0))
    bg.paste(img3, (600, 0))

    bg.save(names["image_stitching_name"] + ".jpg")

    return

# The trained feature map is fused with the original image 訓練出的特徵圖融合原圖
def image_overlap(input_image: Union[str, np.ndarray], names: Dict[str, str]) -> None:
    if isinstance(input_image, str):
        img1 = Image.open(input_image)
    else:
        img1 = Image.fromarray(input_image)
    img2 = Image.open(f'{names["smoke_semantic_image_name"]}.jpg')

    img1 = image_process.process_pil_to_np(img1, size=(256, 256), gray=False)
    img2 = image_process.process_pil_to_np(img2, size=(256, 256), gray=True)
    
    blendImage = image_process.overlap_v3(img1, img2, read_method="PIL_RGBA")

    # Display image 顯示影像
    # blendImg.show()
    Image.fromarray(blendImage).save(f'{names["image_overlap_name"]}.png')

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
    input: Union[str, np.ndarray], model: str, ort_session:ort.InferenceSession, names: Dict[str, str], time_train: float, i: int
) -> None:

    if isinstance(input, str):
        smoke_input_image = read_image(input)
    else:
        smoke_input_image = input.float()
        smoke_input_image = smoke_input_image.permute(2, 0, 1).contiguous()

    output = smoke_semantic(smoke_input_image, ort_session, time_train, i)
    output_image_ori = Image.fromarray(output[0, 0] * 255).convert("L")  
    output_image_ori.save(f'{names["smoke_semantic_image_name"]}_ori_output.jpg')  
    output = (output > 0.5).astype(np.float32)
    output_image = Image.fromarray(output[0, 0] * 255).convert("L")
    output_image.save(f'{names["smoke_semantic_image_name"]}.jpg')

    image_overlap(input, names)
    image_stitching(input, names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", 
        "--image", 
        required=True, 
        help="Path to the input image. If the name of the input file in the terminal has '(' or ')', please rewrite it as '\\(' or '\\)'."
    )
    ap.add_argument(
        "-m", 
        "--model_path", 
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

    names = files_name()

    smoke_segmentation(args["image"], args["model_path"], providers, names)
