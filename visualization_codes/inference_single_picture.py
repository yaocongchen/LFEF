import torch
import torchvision
from visualization_codes.inference import smoke_semantic
# import visualization_codes.image_process_utils as image_process
import visualization_codes.testcpy.image_process_utils_cython as image_process
import argparse
import os
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image, ImageOps
import time


def timeit(func):
    def warp(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print(f"{func.__name__} time cost: {time.time()- start}")
    return warp


def files_name():
    # Set archive name 設定存檔名稱
    if not os.path.exists("./" + "results"):
        os.makedirs("./" + "results")
    save_smoke_semantic_image_name = "./results/smoke_semantic"
    save_image_binary_name = "./results/binary"
    save_image_overlap_name  = "./results/image_overlap"
    save_image_stitching_name = "./results/image_stitching"

    names = {}
    names["smoke_semantic_image_name"] = save_smoke_semantic_image_name
    names["image_binary_name"] = save_image_binary_name
    names["image_overlap_name"] = save_image_overlap_name
    names["image_stitching_name"] = save_image_stitching_name

    return names

# Merge all resulting images 合併所有產生之圖像
def image_stitching(input_image,names):
    bg = Image.new('RGB',(1200, 300), '#000000') # 產生一張 600x300 的全黑圖片
    # Load two images 載入兩張影像
    img1 = Image.open(input_image)
    img2 = Image.open(names["smoke_semantic_image_name"] + ".jpg")
    img3 = Image.open(names["image_binary_name"] + ".jpg")
    img4 = Image.open(names["image_overlap_name"] + ".png")


    # Check if the two images are the same size 檢查兩張影像大小是否一致
    # print(img1.size)
    # print(img2.size)

    # Specify target image size 指定目標圖片大小
    imgSize = (256,256)

    # Change image size 改變影像大小
    img1=img1.resize(imgSize)
    img2=img2.resize(imgSize)
    img3=img3.resize(imgSize)

    img1 = ImageOps.expand(img1, 20, '#ffffff')  # Dilates edges, producing borders 擴張邊緣，產生邊框
    img2 = ImageOps.expand(img2, 20, '#ffffff')  # Dilates edges, producing borders 擴張邊緣，產生邊框
    img3 = ImageOps.expand(img3, 20, '#ffffff')  # Dilates edges, producing borders 擴張邊緣，產生邊框
    img4 = ImageOps.expand(img4, 20, '#ffffff')  # Dilates edges, producing borders 擴張邊緣，產生邊框    
    bg.paste(img1, (0, 0))
    bg.paste(img2, (300, 0))
    bg.paste(img3, (600, 0))
    bg.paste(img4, (900, 0))

    #bg.show()
    bg.save(names["image_stitching_name"]+ ".jpg")

    return

# The trained feature map is fused with the original image 訓練出的特徵圖融合原圖
def image_overlap(input_image,names):
    img1 = Image.open(input_image)
    img1 = img1.convert('RGBA')
    img2 = Image.open(names["smoke_semantic_image_name"] + ".jpg")

    # img2 to binarization img2轉二值化
    binary_image = image_process.gray_to_binary(img2)

    binary_image.save(names["image_binary_name"] + ".jpg")
    img2 = binary_image.convert('RGBA')

    # Specify target image size 指定目標圖片大小
    imgSize = (256,256)

    # Change image size 改變影像大小
    img1=img1.resize(imgSize)
    img2=img2.resize(imgSize)

    blendImage = image_process.overlap_v2(img1,img2,read_method = "PIL_RGBA")
    
    # Display image 顯示影像
    #blendImg.show()
    blendImage.save(names["image_overlap_name"] + ".png")
    
    return

# Main function 主函式 
def smoke_segmentation(input:str,model_input:str,device:torch.device,names:dict,time_train,i):
    smoke_input_image = read_image(input)
    # print(smoke_input_image.shape)
    transform = transforms.Resize([256, 256])
    smoke_input_image = transform(smoke_input_image)
    # print(smoke_input_image.shape)
    smoke_input_image = (smoke_input_image)/255.0
    smoke_input_image  = smoke_input_image.unsqueeze(0).to(device)
    output = smoke_semantic(smoke_input_image,model_input,device,time_train,i)
    torchvision.utils.save_image(output ,names["smoke_semantic_image_name"] + ".jpg")

    image_overlap(input,names)
    image_stitching(input,names)

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help="path to input image")      # If the name of the input file in the terminal has "(" ")", please rewrite it as "\(" "\)" #如果在terminal輸入檔案的名稱有"("  ")"請改寫為  "\("   "\)"
    ap.add_argument('-m','--model_path', required=True, help="load model path")
    args = vars(ap.parse_args())
    
    device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
    print(f"inference_single_Dataset on device {device}.")

    names=files_name()

    smoke_segmentation(args['image'],args['model_path'],device,names)


