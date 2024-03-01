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

import utils
import models.CGNet_2_erfnet31_13_3113_oneloss_inv_attention as network_model  # import self-written models 引入自行寫的模型
from utils.inference import smoke_semantic

import visualization_codes.utils.image_process as image_process
# import visualization_codes.process_utils_cython_version.image_process_utils_cython as image_process

model_name = str(network_model)
print("model_name:", model_name)

def folders_and_files_name():
    # Set save folder and save name 設定存檔資料夾與存檔名稱
    save_smoke_semantic_dir_name = "multiple_result"
    if os.path.exists("./results/" + save_smoke_semantic_dir_name):
        shutil.rmtree(
            "./results/" + save_smoke_semantic_dir_name
        )  # Delete the original folder and content 將原有的資料夾與內容刪除
        os.makedirs(
            "./results/" + save_smoke_semantic_dir_name
        )  # Create new folder 創建新的資料夾
    else:
        # if not os.path.exists("./" + save_smoke_semantic_dir_name):
        os.makedirs("./results/" + save_smoke_semantic_dir_name)
    save_smoke_semantic_image_name = "smoke_semantic_image"

    save_image_overlap_dir_name = "multiple_overlap"
    if os.path.exists("./results/" + save_image_overlap_dir_name):
        shutil.rmtree("./results/" + save_image_overlap_dir_name)
        os.makedirs("./results/" + save_image_overlap_dir_name)
    else:
        # if not os.path.exists("./" + save_image_overlap_dir_name):
        os.makedirs("./results/" + save_image_overlap_dir_name)
    save_image_overlap_name = "image_overlap"

    save_image_overlap_masks_dir_name = "multiple_overlap_masks"
    if os.path.exists("./results/" + save_image_overlap_masks_dir_name):
        shutil.rmtree("./results/" + save_image_overlap_masks_dir_name)
        os.makedirs("./results/" + save_image_overlap_masks_dir_name)
    else:
        # if not os.path.exists("./" + save_image_overlap_dir_name):
        os.makedirs("./results/" + save_image_overlap_masks_dir_name)
    save_image_overlap_masks_name = "image_overlap_masks"

    save_image_stitching_dir_name = "multiple_stitching"
    if os.path.exists("./results/" + save_image_stitching_dir_name):
        shutil.rmtree("./results/" + save_image_stitching_dir_name)
        os.makedirs("./results/" + save_image_stitching_dir_name)
    else:
        # if not os.path.exists("./" + save_image_stitching_dir_name):
        os.makedirs("./results/" + save_image_stitching_dir_name)
    save_image_stitching_name = "image_stitching"

    names = {}
    names["smoke_semantic_dir_name"] = save_smoke_semantic_dir_name
    names["smoke_semantic_image_name"] = save_smoke_semantic_image_name
    names["image_overlap_dir_name"] = save_image_overlap_dir_name
    names["image_overlap_name"] = save_image_overlap_name
    names["image_overlap_masks_dir_name"] = save_image_overlap_masks_dir_name
    names["image_overlap_masks_name"] = save_image_overlap_masks_name
    names["image_stitching_dir_name"] = save_image_stitching_dir_name
    names["image_stitching_name"] = save_image_stitching_name

    return names

iou_list = []
# Merge all resulting images 合併所有產生之圖像
def image_stitching(input_image, filename_no_extension, names, mask_image, iou_np):
    bg = Image.new(
        "RGB", (605, 940), "#000000"
    )  # Produces a 1200 x 300 all black image 產生一張 1200x300 的全黑圖片
    # Load two images 載入兩張影像
    img1 = Image.open(input_image)
    img2 = Image.open(mask_image)
    img3 = Image.open(
        f'./results/{names["image_overlap_masks_dir_name"]}/{names["image_overlap_masks_name"]}_{filename_no_extension}.png'
    )
    img4 = Image.open(
        f'./results/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_image_name"]}_{filename_no_extension}.jpg'
    )
    img5 = Image.open(
        f'./results/{names["image_overlap_dir_name"]}/{names["image_overlap_name"]}_{filename_no_extension}.png'
    )

    # Check if the two images are the same size 檢查兩張影像大小是否一致
    # print(img1.size)
    # print(img2.size)

    # Specify target image size 指定目標圖片大小
    imgSize = (256, 256)

    # Change image size 改變影像大小
    img1 = img1.resize(imgSize)
    img2 = img2.resize(imgSize)
    img3 = img3.resize(imgSize)
    img4 = img4.resize(imgSize)
    img5 = img5.resize(imgSize)

    img1 = ImageOps.expand(
        img1, 5, "#ffffff"
    )  # Dilates edges, producing borders 擴張邊緣，產生邊框
    img2 = ImageOps.expand(
        img2, 5, "#ffffff"
    )  # Dilates edges, producing borders 擴張邊緣，產生邊框
    img3 = ImageOps.expand(
        img3, 5, "#ffffff"
    )  # Dilates edges, producing borders 擴張邊緣，產生邊框
    img4 = ImageOps.expand(
        img4, 5, "#ffffff"
    )  # Dilates edges, producing borders 擴張邊緣，產生邊框
    img5 = ImageOps.expand(
        img5, 5, "#ffffff"
    )  # Dilates edges, producing borders 擴張邊緣，產生邊框

    # 創建一個繪圖對象
    draw = ImageDraw.Draw(bg)

    # 選擇一種字體和字體大小
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-BI.ttf", 20)

    # 在圖像上添加文字 顏色為紅色
    draw.text((230, 5), "Original Image", fill=(255, 255, 255), font=font)
    bg.paste(img1, (170, 40))

    draw.text((90, 320), "Mask Image", fill=(255, 255, 255), font=font)
    bg.paste(img2, (20, 350))
    bg.paste(img3, (20, 630))

    draw.text((380, 320), "Model Output", fill=(255, 255, 255), font=font)
    bg.paste(img4, (320, 350))
    bg.paste(img5, (320, 630))


    draw.text((230, 910), "IoU:  " + str(iou_np) + "%", fill=(255, 255, 255), font=font)


    # bg.show()
    bg.save(
        f'./results/{names["image_stitching_dir_name"]}/{names["image_stitching_name"]}_{filename_no_extension}.jpg'
    )

    return


# The trained feature map is fuse d with the original image 訓練出的特徵圖融合原圖
def image_overlap(input_image, filename_no_extension, names, mask_image):
    img1 = Image.open(input_image)
    img1 = img1.convert("RGBA")
    img2 = Image.open(
        f'./results/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_image_name"]}_{filename_no_extension}.jpg'
    )
    img3 = Image.open(mask_image)

    # # img2 to binarization img2轉二值化
    # binary_image = image_process.gray_to_binary(img2)

    # binary_image.save(
    #     f'./results/{names["image_binary_dir_name"]}/{names["image_binary_name"]}_{filename_no_extension}.jpg'
    # )

    img2 = img2.convert("RGBA")
    img3 = img3.convert("RGBA")

    # Specify target image size 指定目標圖片大小
    imgSize = (256, 256)

    # Change image size 改變影像大小
    img1 = img1.resize(imgSize)
    img2 = img2.resize(imgSize)
    img3 = img3.resize(imgSize)

    blendImage = image_process.overlap_v2(img1, img2, read_method="PIL_RGBA")
    blendImage_mask = image_process.overlap_v2(img1, img3, read_method="PIL_RGBA")

    # Display image 顯示影像
    # blendImage.show()
    blendImage.save(
        f'./results/{names["image_overlap_dir_name"]}/{names["image_overlap_name"]}_{filename_no_extension}.png'
    )
    blendImage_mask.save(
        f'./results/{names["image_overlap_masks_dir_name"]}/{names["image_overlap_masks_name"]}_{filename_no_extension}.png'
    )

    return


# Main function 主函式
def smoke_segmentation(
    directory: str, model: str, device: torch.device, names: dict, time_train, i
):
    n_element = 0
    mean_miou = 0
    images_dir = os.path.join(directory,"images")
    masks_dir = os.path.join(directory,"masks")
    pbar = tqdm((os.listdir(images_dir)), total=len(os.listdir(images_dir)))
    filename = sorted(os.listdir(images_dir), key=lambda name: int(name.split('.')[0]))
    for filename in pbar:
        filename_no_extension = os.path.splitext(filename)[0] 
        smoke_input_image = read_image(os.path.join(images_dir, filename)).to(device)
        transform = transforms.Resize([256, 256],antialias=True)
        smoke_input_image = transform(smoke_input_image)
        smoke_input_image = (smoke_input_image) / 255.0
        smoke_input_image = smoke_input_image.unsqueeze(0).to(device)
        mask_input_image = read_image(os.path.join(masks_dir, filename)).to(device)
        mask_input_image = transform(mask_input_image)
        mask_input_image = (mask_input_image) / 255.0
        mask_input_image = mask_input_image.unsqueeze(0).to(device)
        output = smoke_semantic(smoke_input_image, model, device, time_train, i)
        iou = utils.metrics.IoU(output, mask_input_image)
        output = (output > 0.5).float()
        torchvision.utils.save_image(
            output,
            f'./results/{names["smoke_semantic_dir_name"]}/{names["smoke_semantic_image_name"]}_{filename_no_extension}.jpg',
        )
        image_overlap(os.path.join(images_dir, filename), filename_no_extension, names, os.path.join(masks_dir, filename))
        iou_np = np.round(iou.cpu().detach().numpy() * 100, 2)
        n_element += 1
        mean_miou += (iou_np.item() - mean_miou) / n_element  # 別人研究出的算平均的方法
        image_stitching(os.path.join(images_dir, filename), filename_no_extension, names, os.path.join(masks_dir, filename), iou_np)
        iou_list.append(iou_np)
        # i += 1
    return iou_list, mean_miou


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
    # print(names)
    # Calculate the total execution time 計算總執行時間
    model = network_model.Net().to(device)
    model.load_state_dict(torch.load(args["model_path"], map_location=device))

    model.eval()
    i = 0
    time_train = []
    time_start = time.time()
    iou_list,miou = smoke_segmentation(args["test_directory"], model, device, names, time_train, i)
    plt.hist(iou_list, bins=len(iou_list))
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.title(f"IoU Histogram \n mIoU:{miou:.2f}%")
    
    save_path = f'./results/{names["image_stitching_dir_name"]}/IoU_histogram.png'
    plt.savefig(save_path)

    c = utils.metrics.Calculate(model)
    model_size = c.get_model_size()
    flops, params = c.get_params()

    with open(f"./results/{names['image_stitching_dir_name']}/log.txt", "w") as f:
        f.write(f"{model_name}\n"
                f"test directory: {args['test_directory']}\n"
                f"model path: {args['model_path']}\n"
                f"model size: {model_size}\n"
                f"flops: {flops}\n"
                f"params: {params}\n"
                f"mIoU: {miou:.2f}%\n"
                f"update time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        
    total_image = len(os.listdir(args["test_directory"]))
    time_end = time.time()
    spend_time = int(time_end - time_start)
    time_min = spend_time // 60
    time_sec = spend_time % 60
    print("totally cost:", f"{time_min}m {time_sec}s")
    # print(total_image)

    # Calculate FPS
    print("FPS:{:.3f}".format(total_image / spend_time))
