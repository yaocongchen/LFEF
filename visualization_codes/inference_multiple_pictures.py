
import torch
import torchvision
import os
import argparse
import time
import shutil
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image, ImageOps

from visualization_codes.inference import smoke_semantic
# import visualization_codes.image_process_utils as image_process
import visualization_codes.process_utils_cython_version.image_process_utils_cython as image_process

def folders_and_files_name():
        # Set save folder and save name 設定存檔資料夾與存檔名稱
    save_smoke_semantic_dir_name = "multiple_result"
    if os.path.exists("./results/" + save_smoke_semantic_dir_name):
        shutil.rmtree("./results/" + save_smoke_semantic_dir_name)      # Delete the original folder and content 將原有的資料夾與內容刪除
        os.makedirs("./results/" + save_smoke_semantic_dir_name)        # Create new folder 創建新的資料夾
    else:
    # if not os.path.exists("./" + save_smoke_semantic_dir_name):
        os.makedirs("./results/" + save_smoke_semantic_dir_name)
    save_smoke_semantic_image_name = "smoke_semantic_image_"

    save_image_binary_dir_name = "multiple_binary"
    if os.path.exists("./results/" + save_image_binary_dir_name):
        shutil.rmtree("./results/" + save_image_binary_dir_name)
        os.makedirs("./results/" + save_image_binary_dir_name)
    else:
    #if not os.path.exists("./" + save_image_binary_dir_name):
        os.makedirs("./results/" + save_image_binary_dir_name)
    save_image_binary_name = "binary_"

    save_image_overlap_dir_name  = "multiple_overlap"
    if os.path.exists("./results/" + save_image_overlap_dir_name):
        shutil.rmtree("./results/" + save_image_overlap_dir_name)
        os.makedirs("./results/" + save_image_overlap_dir_name)
    else:
    #if not os.path.exists("./" + save_image_overlap_dir_name):
        os.makedirs("./results/" + save_image_overlap_dir_name)
    save_image_overlap_name  = "image_overlap_"

    save_image_stitching_dir_name = "multiple_stitching"
    if os.path.exists("./results/" + save_image_stitching_dir_name):
        shutil.rmtree("./results/" + save_image_stitching_dir_name)
        os.makedirs("./results/" + save_image_stitching_dir_name)
    else:
    #if not os.path.exists("./" + save_image_stitching_dir_name):
        os.makedirs("./results/" + save_image_stitching_dir_name)
    save_image_stitching_name = "image_stitching_"

    names = {}
    names["smoke_semantic_dir_name"] = save_smoke_semantic_dir_name
    names["smoke_semantic_image_name"] = save_smoke_semantic_image_name
    names["image_binary_dir_name"] = save_image_binary_dir_name
    names["image_binary_name"] = save_image_binary_name
    names["image_overlap_dir_name"] = save_image_overlap_dir_name
    names["image_overlap_name"] = save_image_overlap_name
    names["image_stitching_dir_name"] = save_image_stitching_dir_name
    names["image_stitching_name"] = save_image_stitching_name

    return names

# Merge all resulting images 合併所有產生之圖像
def image_stitching(input_image,i,names):
    bg = Image.new('RGB',(1200, 300), '#000000') # Produces a 1200 x 300 all black image 產生一張 1200x300 的全黑圖片
    # Load two images 載入兩張影像
    img1 = Image.open(input_image)
    img2 = Image.open("./results/" + names["smoke_semantic_dir_name"] + "/" + names["smoke_semantic_image_name"]  + f"{i}.jpg")
    img3 = Image.open("./results/" + names["image_binary_dir_name"] + "/" + names["image_binary_name"]  + f"{i}.jpg")
    img4 = Image.open("./results/" + names["image_overlap_dir_name"]  + "/" + names["image_overlap_name"] + f"{i}.png")


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
    bg.save("./results/" + names["image_stitching_dir_name"] + "/" + names["image_stitching_name"] + f"{i}.jpg")

    return

# The trained feature map is fuse d with the original image 訓練出的特徵圖融合原圖
def image_overlap(input_image,i,names):
    img1 = Image.open(input_image)
    img1 = img1.convert('RGBA')
    img2 = Image.open("./results/" + names["smoke_semantic_dir_name"] + "/" + names["smoke_semantic_image_name"]  + f"{i}.jpg")

    # img2 to binarization img2轉二值化
    binary_image = image_process.gray_to_binary(img2)

    binary_image.save("./results/" + names["image_binary_dir_name"] + "/" + names["image_binary_name"]  + f"{i}.jpg")

    img2 = binary_image.convert('RGBA')

    # Specify target image size 指定目標圖片大小
    imgSize = (256,256)

    # Change image size 改變影像大小
    img1=img1.resize(imgSize)
    img2=img2.resize(imgSize)

    blendImage = image_process.overlap_v2(img1,img2,read_method = "PIL_RGBA")

    # Display image 顯示影像
    #blendImage.show()
    blendImage.save("./results/" + names["image_overlap_dir_name"]  + "/" + names["image_overlap_name"] + f"{i}.png")

    return

# Main function 主函式
def smoke_segmentation(directory:str,model_input:str,device:torch.device,names:dict,time_train,i):
    i = 0
    pbar = tqdm((os.listdir(directory)),total=len(os.listdir(directory)))
    for filename in pbar:
        smoke_input_image = read_image(os.path.join(directory,filename)).to(device)
        transform = transforms.Resize([256, 256])
        smoke_input_image = transform(smoke_input_image)
        smoke_input_image = (smoke_input_image)/255.0
        smoke_input_image  = smoke_input_image.unsqueeze(0).to(device)
        output = smoke_semantic(smoke_input_image,model_input,device,time_train,i)
        i+=1
        torchvision.utils.save_image(output ,"./results/" + names["smoke_semantic_dir_name"] + "/" + names["smoke_semantic_image_name"]  + f"{i}.jpg")
        image_overlap(os.path.join(directory,filename),i,names)
        image_stitching(os.path.join(directory,filename),i,names)

    return 

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-td", "--test_directory",required=True, help="path to test images directory")
    ap.add_argument('-m','--model_path' ,required=True, help="load model path")
    args = vars(ap.parse_args())

    device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
    print(f"inference_multiple_Dataset on device {device}.")

    names = folders_and_files_name()
    print(names)
# Calculate the total execution time 計算總執行時間  
    time_start = time.time()
    smoke_segmentation(args["test_directory"],args['model_path'],device,names)
    total_image = len(os.listdir(args["test_directory"]))
    time_end = time.time()
    spend_time = int(time_end-time_start) 
    time_min = spend_time // 60 
    time_sec = spend_time % 60
    print('totally cost:',f"{time_min}m {time_sec}s")
    #print(total_image)

# Calculate FPS
    print("FPS:{:.1f}".format(total_image/(time_end-time_start)))



