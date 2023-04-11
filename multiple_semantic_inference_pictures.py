from inference import smoke_semantic
import torch
import torchvision
import os
import argparse
from time import sleep
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image, ImageOps
import shutil
import time

device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
print(f"Training on device {device}.")

# Set save folder and save name 設定存檔資料夾與存檔名稱
save_smoke_semantic_dir_name = "multiple_result"
if os.path.exists("./" + save_smoke_semantic_dir_name):
    shutil.rmtree("./" + save_smoke_semantic_dir_name)      # Delete the original folder and content 將原有的資料夾與內容刪除
    os.makedirs("./" + save_smoke_semantic_dir_name)        # Create new folder 創建新的資料夾
else:
# if not os.path.exists("./" + save_smoke_semantic_dir_name):
    os.makedirs("./" + save_smoke_semantic_dir_name)
save_smoke_semantic_image_name = "smoke_semantic_image_"

save_image_binary_dir_name = "multiple_binary"
if os.path.exists("./" + save_image_binary_dir_name):
    shutil.rmtree("./" + save_image_binary_dir_name)
    os.makedirs("./" + save_image_binary_dir_name)
else:
#if not os.path.exists("./" + save_image_binary_dir_name):
    os.makedirs("./" + save_image_binary_dir_name)
save_image_binary_name = "binary_"

save_image_overlap_dir_name  = "multiple_overlap"
if os.path.exists("./" + save_image_overlap_dir_name):
    shutil.rmtree("./" + save_image_overlap_dir_name)
    os.makedirs("./" + save_image_overlap_dir_name)
else:
#if not os.path.exists("./" + save_image_overlap_dir_name):
    os.makedirs("./" + save_image_overlap_dir_name)
save_image_overlap_name  = "image_overlap_"

save_image_stitching_dir_name = "multiple_stitching"
if os.path.exists("./" + save_image_stitching_dir_name):
    shutil.rmtree("./" + save_image_stitching_dir_name)
    os.makedirs("./" + save_image_stitching_dir_name)
else:
#if not os.path.exists("./" + save_image_stitching_dir_name):
    os.makedirs("./" + save_image_stitching_dir_name)
save_image_stitching_name = "image_stitching_"

# Merge all resulting images 合併所有產生之圖像
def image_stitching(input_image,i):
    bg = Image.new('RGB',(1200, 300), '#000000') # Produces a 1200 x 300 all black image 產生一張 1200x300 的全黑圖片
    # Load two images 載入兩張影像
    img1 = Image.open(input_image)
    img2 = Image.open("./" + save_smoke_semantic_dir_name + "/" + save_smoke_semantic_image_name  + f"{i}.jpg")
    img3 = Image.open("./" + save_image_binary_dir_name + "/" + save_image_binary_name + f"{i}.jpg")
    img4 = Image.open("./" + save_image_overlap_dir_name + "/" + save_image_overlap_name + f"{i}.png")


    # Check if the two images are the same size 檢查兩張影像大小是否一致
    # print(img1.size)
    # print(img2.size)

    # Specify target image size 指定目標圖片大小
    imgSize = (256,256)

    # Change image size 改變影像大小
    img1=img1.resize(imgSize)
    img2=img2.resize(imgSize)
    img3=img3.resize(imgSize)
    img4=img4.resize(imgSize)

    img1 = ImageOps.expand(img1, 20, '#ffffff')  # Dilates edges, producing borders 擴張邊緣，產生邊框
    img2 = ImageOps.expand(img2, 20, '#ffffff')  # Dilates edges, producing borders 擴張邊緣，產生邊框
    img3 = ImageOps.expand(img3, 20, '#ffffff')  # Dilates edges, producing borders 擴張邊緣，產生邊框
    img4 = ImageOps.expand(img4, 20, '#ffffff')  # Dilates edges, producing borders 擴張邊緣，產生邊框

    bg.paste(img1, (0, 0))
    bg.paste(img2, (300, 0))
    bg.paste(img3, (600, 0))
    bg.paste(img4, (900, 0))

    #bg.show()
    bg.save("./" + save_image_stitching_dir_name + "/" + save_image_stitching_dir_name + f"{i}.jpg")

    return

# The trained feature map is fuse d with the original image 訓練出的特徵圖融合原圖
def image_overlap(input_image,i):
    img1 = Image.open(input_image)
    img1 = img1.convert('RGBA')
    img2 = Image.open("./" + save_smoke_semantic_dir_name + "/" + save_smoke_semantic_image_name  + f"{i}.jpg")

    # img2 to binarization img2轉二值化
    gray = img2.convert('L')
    threshold = 200


    table = []
    for pixel_g in range(256):
        if pixel_g < threshold:
            table.append(0)
        else:
            table.append(1)

    binary = gray.point(table, '1')
    binary.save("./" + save_image_binary_dir_name + "/" + save_image_binary_name + f"{i}.jpg")

    img2 = binary.convert('RGBA')

    # Specify target image size 指定目標圖片大小
    imgSize = (256,256)

    # Change image size 改變影像大小
    img1=img1.resize(imgSize)
    img2=img2.resize(imgSize)

    L,H = img2.size
    black_background = (0, 0, 0, 255)
    #white_mask = (255, 255, 255, 255)

    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img2.getpixel(dot)
            if color_1 == black_background:
                color_1 = color_1[:-1] + (0,)   # Commas are used to create a (tuple) 逗號是用於創造一個(tuple)
                img2.putpixel(dot,color_1)
            else:
                color_1 = (255,0,0,) + color_1[3:]  #逗號是用於創造一個(tuple)
                img2.putpixel(dot,color_1)
    #img2.show()
    # Overlay image 疊合影像
    blendImg = Image.blend(img1, img2 , alpha = 0.2)
    # Display image 顯示影像
    #blendImg.show()
    blendImg.save("./" + save_image_overlap_dir_name + "/" + save_image_overlap_name + f"{i}.png")

    return

# Main function 主函式
def multiple_smoke_semantic_test(directory,model_input):
    i = 0
    pbar = tqdm((os.listdir(directory)),total=len(os.listdir(directory)))
    for filename in pbar:
        smoke_input_image = read_image(os.path.join(directory,filename))
        transform = transforms.Resize([256, 256])
        smoke_input_image = transform(smoke_input_image)
        smoke_input_image = (smoke_input_image)/255.0
        smoke_input_image  = smoke_input_image.unsqueeze(0).to(device)
        output = smoke_semantic(smoke_input_image,model_input)
        i+=1
        torchvision.utils.save_image(output ,"./" + save_smoke_semantic_dir_name + "/" + save_smoke_semantic_image_name  + f"{i}.jpg")
        image_overlap(os.path.join(directory,filename),i)
        image_stitching(os.path.join(directory,filename),i)

    return 
if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-td", "--test_directory",default = "/home/yaocong/Experimental/speed_smoke_segmentation/ttt/img",required=True, help="path to test images directory")
    ap.add_argument('-m','--model_path' ,required=True, help="load model path")
    args = vars(ap.parse_args())

# Calculate the total execution time 計算總執行時間  
    time_start = time.time()
    multiple_smoke_semantic_test(args["test_directory"],args['model_path'])
    total_image = len(os.listdir(args["test_directory"]))
    time_end = time.time()
    spend_time = int(time_end-time_start) 
    time_min = spend_time // 60 
    time_sec = spend_time % 60
    print('totally cost:',f"{time_min}m {time_sec}s")
    #print(total_image)

# Calculate FPS
    print("FPS:{:.1f}".format(total_image/(time_end-time_start)))



