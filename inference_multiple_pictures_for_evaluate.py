import torch
import torchvision
import os
import argparse
import time
import shutil
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image, ImageOps ,ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

import utils
import models.CGNet_2_erfnet31_13_3113_oneloss_inv_attention as network_model  # import self-written models 引入自行寫的模型
from utils.inference import smoke_semantic

import visualization_codes.utils.image_process as image_process
# import visualization_codes.process_utils_cython_version.image_process_utils_cython as image_process

model_name = str(network_model)
print("model_name:", model_name)

def create_directory(directory_name):
    directory_path = f"./results/evaluate_folder/{directory_name}"
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)
    return directory_name

def folders_and_files_name():
    names = {
        "smoke_semantic_dir_name": create_directory("multiple_result"),
        "smoke_semantic_image_name": "smoke_semantic_image",
        "smoke_semantic_aux_name": "smoke_semantic_aux",
        "image_overlap_dir_name": create_directory("multiple_overlap"),
        "image_overlap_name": "image_overlap",
        "image_overlap_masks_dir_name": create_directory("multiple_overlap_masks"),
        "image_overlap_masks_name": "image_overlap_masks",
        "image_stitching_dir_name": create_directory("multiple_stitching"),
        "image_stitching_name": "image_stitching",
        "image_stitching_dir_down_name": create_directory("multiple_stitching_down"),
        "image_stitching_down_name": "image_stitching_down",
    }

    return names

iou_list = []


def load_and_process_image_with_border(path, size=(256, 256), border=5):
    img = Image.open(path)
    img = img.resize(size)
    img = ImageOps.expand(img, border, "#ffffff")
    return img
# Merge all resulting images 合併所有產生之圖像
def image_stitching(input_image, filename_no_extension, names, mask_image, iou_np, customssim_np, hd_np, dice_np):
    bg = Image.new("RGB", (905, 980), "#000000")

    img1 = load_and_process_image_with_border(input_image)
    img2 = load_and_process_image_with_border(mask_image)
    img3 = load_and_process_image_with_border(f'./results/evaluate_folder/{names["image_overlap_masks_dir_name"]}/{names["image_overlap_masks_name"]}_{filename_no_extension}.png')
    img4 = load_and_process_image_with_border(f'./results/evaluate_folder/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_image_name"]}_{filename_no_extension}.jpg')
    img5 = load_and_process_image_with_border(f'./results/evaluate_folder/{names["image_overlap_dir_name"]}/{names["image_overlap_name"]}_{filename_no_extension}.png')
    img6 = load_and_process_image_with_border(f'./results/evaluate_folder/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_aux_name"]}_{filename_no_extension}.jpg')
    img7 = load_and_process_image_with_border(f'./results/evaluate_folder/{names["image_overlap_dir_name"]}/{names["image_overlap_name"]}_{filename_no_extension}_aux.png')


    draw = ImageDraw.Draw(bg)
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-BI.ttf", 20)

    draw.text((230, 5), "Original Image", fill=(255, 255, 255), font=font)
    bg.paste(img1, (170, 40))

    draw.text((90, 320), "Mask Image", fill=(255, 255, 255), font=font)
    bg.paste(img2, (20, 350))
    bg.paste(img3, (20, 630))

    draw.text((380, 320), "Model Output", fill=(255, 255, 255), font=font)
    bg.paste(img4, (320, 350))
    bg.paste(img5, (320, 630))

    draw.text((670, 320), "Auxiliary Output", fill=(255, 255, 255), font=font)
    bg.paste(img6, (620, 350))
    bg.paste(img7, (620, 630))


    draw.text((230, 910), "IoU:  " + str(iou_np) + "%", fill=(255, 255, 255), font=font)
    draw.text((230, 930), "Dice: " + str(dice_np), fill=(255, 255, 255), font=font)

    bg.save(f'./results/evaluate_folder/{names["image_stitching_dir_name"]}/{names["image_stitching_name"]}_{filename_no_extension}.jpg')
    if iou_np < 20:
        bg.save(f'./results/evaluate_folder/{names["image_stitching_dir_down_name"]}/{names["image_stitching_down_name"]}_{filename_no_extension}.jpg')

    return

# The trained feature map is fuse d with the original image 訓練出的特徵圖融合原圖
def image_overlap(input_image, filename_no_extension, names, mask_image):
    img1 = Image.open(input_image)
    img2 = Image.open(
        f'./results/evaluate_folder/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_image_name"]}_{filename_no_extension}.jpg',
    )
    img3 = Image.open(mask_image)
    img4 = Image.open(f'./results/evaluate_folder/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_aux_name"]}_{filename_no_extension}.jpg',)

    img1 = image_process.process_pil_to_np(img1, gray=False)
    img2 = image_process.process_pil_to_np(img2, gray=True)
    img3 = image_process.process_pil_to_np(img3, gray=True)
    img4 = image_process.process_pil_to_np(img4, gray=True)
    
    blendImage = image_process.overlap_v3(img1, img2, read_method="PIL_RGBA")
    blendImage_mask = image_process.overlap_v3(img1, img3, read_method="PIL_RGBA")
    blendImage_aux = image_process.overlap_v3(img1, img4, read_method="PIL_RGBA")

    Image.fromarray(blendImage).save(
        f'./results/evaluate_folder/{names["image_overlap_dir_name"]}/{names["image_overlap_name"]}_{filename_no_extension}.png'
    )
    Image.fromarray(blendImage_mask).save(
        f'./results/evaluate_folder/{names["image_overlap_masks_dir_name"]}/{names["image_overlap_masks_name"]}_{filename_no_extension}.png'
    )
    Image.fromarray(blendImage_aux).save(
        f'./results/evaluate_folder/{names["image_overlap_dir_name"]}/{names["image_overlap_name"]}_{filename_no_extension}_aux.png'
    )

    return


def load_and_process_image_torch(path, device, transform):
    img = read_image(path).to(device)
    img = transform(img)
    img = img / 255.0
    img = img.unsqueeze(0).to(device)
    return img
# Main function 主函式
def smoke_segmentation(directory: str, model: str, device: torch.device, names: dict, time_train, i):
    n_element = 0
    mean_miou = 0
    images_dir = os.path.join(directory,"images")
    masks_dir = os.path.join(directory,"masks")

    transform = transforms.Resize([256, 256],antialias=True)

    pbar = tqdm((os.listdir(images_dir)), total=len(os.listdir(images_dir)))
    filename = sorted(os.listdir(images_dir), key=lambda name: int(name.split('.')[0]))

    for filename in pbar:
        filename_no_extension = os.path.splitext(filename)[0] 

        smoke_input_image = load_and_process_image_torch(os.path.join(images_dir, filename), device, transform)
        mask_input_image = load_and_process_image_torch(os.path.join(masks_dir, filename), device, transform)

        output, aux = smoke_semantic(smoke_input_image, model, device, time_train, i)

        # iou = utils.metrics.IoU(output, mask_input_image)
        mask_input_image = mask_input_image.long()
        tp, fp, fn, tn = smp.metrics.get_stats(output, mask_input_image, mode='binary', threshold=0.5)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        mask_input_image = mask_input_image.float()
        customssim = utils.metrics.ssim_val(output, mask_input_image)
        hd = utils.metrics.Sobel_hausdorffDistance_metric(output, mask_input_image, device)
        dice = utils.metrics.dice_coef(output, mask_input_image)

        output = (output > 0.5).float()
        aux = (aux > 0.5).float()
        torchvision.utils.save_image(
            output,
            f'./results/evaluate_folder/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_image_name"]}_{filename_no_extension}.jpg',
        )
        torchvision.utils.save_image(
            aux,
            f'./results/evaluate_folder/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_aux_name"]}_{filename_no_extension}.jpg',
        )

        image_overlap(os.path.join(images_dir, filename), filename_no_extension, names, os.path.join(masks_dir, filename))

        iou_np = np.round(iou.cpu().detach().numpy() * 100, 2)
        customssim_np = np.round(customssim.cpu().detach().numpy() * 100, 2)
        hd_np = hd.cpu().detach().numpy()
        dice_np = dice.cpu().detach().numpy()

        n_element += 1
        mean_miou += (iou_np.item() - mean_miou) / n_element  # 別人研究出的算平均的方法

        image_stitching(os.path.join(images_dir, filename), filename_no_extension, names, os.path.join(masks_dir, filename), iou_np, customssim_np, hd_np, dice_np)
        
        iou_list.append(iou_np)

    return iou_list, mean_miou


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-td", "--test_directory", required=True, help="path to test images directory"
    )
    ap.add_argument("-m", "--model_path", required=True, help="load model path")
    args = vars(ap.parse_args())

    print(f"test directory: {args['test_directory']}")  # test directory 測試目錄
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"inference_multiple_Dataset on device {device}.")

    names = folders_and_files_name()
    # print(names)
    # Calculate the total execution time 計算總執行時間
    model = network_model.Net().to(device)

    c = utils.metrics.Calculate(model)
    model_size = c.get_model_size()
    flops, params = c.get_params()

    model = torch.compile(model)  #pytorch2.0編譯功能(舊GPU無法使用)
    torch.set_float32_matmul_precision('high')
    model.load_state_dict(torch.load(args["model_path"], map_location=device))
    model.eval()

    i = 0
    time_train = []
    time_start = time.time()
    
    iou_list,miou = smoke_segmentation(args["test_directory"], model, device, names, time_train, i)
    

    images_dir = os.path.join(args["test_directory"],"images")
    total_image = len(os.listdir(images_dir))
    time_end = time.time()
    utils.metrics.report_fps_and_time(total_image, time_start, time_end)

    counts, bins, patches = plt.hist(iou_list, bins=100, edgecolor="black")
    plt.xlabel("IoU")
    plt.ylabel("Number of Images")

    # # 獲取直方圖的最高點
    # max_count = int(max(counts))
    # 顯示 miou的線 並顯示數值
    plt.axvline(x=miou, color='r', linestyle='--', label=f'mIoU:{miou:.2f}%')
    std_iou = np.std(iou_list)
    plt.axvline(x=miou + std_iou, color='m', linestyle='--', label=f'mIoU+std:{miou + std_iou:.2f}%')
    plt.axvline(x=miou - std_iou, color='m', linestyle='--', label=f'mIoU-std:{miou - std_iou:.2f}%')

    # plt.yticks(range(0, max_count+1, 5))
    # 在 y=1 的位置繪製一條水平線
    plt.axhline(y=1, color='y', linestyle='--', label='y=1')

    # 顯示標籤
    plt.legend()
    
    plt.title(f"IoU Histogram \n std:{std_iou:.2f}%")
    
    save_path = f'./results/evaluate_folder/IoU_histogram.png'
    plt.savefig(save_path)

    print(f"mIoU: {miou:.2f}%")
    print(f"update time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    with open(f"./results/evaluate_folder/log.txt", "w") as f:
        f.write(f"{model_name}\n"
                f"test directory: {args['test_directory']}\n"
                f"model path: {args['model_path']}\n"
                f"model size: {model_size}\n"
                f"flops: {flops}\n"
                f"params: {params}\n"
                f"mIoU: {miou:.2f}%\n"
                f"update time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        

