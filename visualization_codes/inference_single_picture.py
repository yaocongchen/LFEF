import torch
import torchvision
from visualization_codes.inference import smoke_semantic
import argparse
import os
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image, ImageOps

def files_name():
    # Set archive name 設定存檔名稱
    if not os.path.exists("./" + "results"):
        os.makedirs("./" + "results")
    save_smoke_semantic_image_name = "./results/smoke_semantic"
    save_image_binary_name = "./results/binary"
    save_image_overlap_name  = "./results/image_overlap"
    save_image_stitching_name = "./results/image_stitching"

    return save_smoke_semantic_image_name,save_image_binary_name,save_image_overlap_name,save_image_stitching_name

# Merge all resulting images 合併所有產生之圖像
def image_stitching(input_image,names):
    bg = Image.new('RGB',(1200, 300), '#000000') # 產生一張 600x300 的全黑圖片
    # Load two images 載入兩張影像
    img1 = Image.open(input_image)
    img2 = Image.open(names[0] + ".jpg")
    img3 = Image.open(names[1] + ".jpg")
    img4 = Image.open(names[2] + ".png")


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
    bg.save(names[3]+ ".jpg")

    return

# The trained feature map is fused with the original image 訓練出的特徵圖融合原圖
def image_overlap(input_image,names):
    img1 = Image.open(input_image)
    img1 = img1.convert('RGBA')
    img2 = Image.open(names[0] + ".jpg")

    # img2 to binarization img2轉二值化
    gray = img2.convert('L')
    threshold = 200

    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    binary = gray.point(table, '1')
    binary.save(names[1] + ".jpg")
    img2 = binary.convert('RGBA')


    # Specify target image size 指定目標圖片大小
    imgSize = (256,256)

    # Change image size 改變影像大小
    img1=img1.resize(imgSize)
    img2=img2.resize(imgSize)

    W,H = img2.size
    black_background = (0, 0, 0, 255)

    for h in range(H):
        for w in range(W):
            dot = (w,h)
            color_1 = img2.getpixel(dot)
            if color_1 == black_background:
                color_1 = color_1[:-1] + (0,)   # Commas are used to create a (tuple) 逗號是用於創造一個(tuple)
                img2.putpixel(dot,color_1)
            else:
                color_1 = (255,0,0,) + color_1[3:]  #  逗號是用於創造一個(tuple)
                img2.putpixel(dot,color_1)
            
    #img2.show()
    # Overlay image 疊合影像
    blendImg = Image.blend(img1, img2 , alpha = 0.2)
    # Display image 顯示影像
    #blendImg.show()
    blendImg.save(names[2] + ".png")
    
    return

# Main function 主函式 
def smoke_segmentation(input,model_input,device,names):
    smoke_input_image = read_image(input)
    # print(smoke_input_image.shape)
    transform = transforms.Resize([256, 256])
    smoke_input_image = transform(smoke_input_image)
    # print(smoke_input_image.shape)
    smoke_input_image = (smoke_input_image)/255.0
    smoke_input_image  = smoke_input_image.unsqueeze(0).to(device)
    output = smoke_semantic(smoke_input_image,model_input,device)
    torchvision.utils.save_image(output ,names[0] + ".jpg")

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

    names=files_name

    smoke_segmentation(args['image'],args['model_path'],device,names)