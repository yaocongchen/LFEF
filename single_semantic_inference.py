import torch
import torchvision
from inference import smoke_semantic
import argparse
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image, ImageOps

#設定存檔名稱
save_smoke_semantic_image_name = "./smoke_semantic"
save_image_binary_name = "./binary"
save_image_overlap_name  = "./image_overlap"
save_image_stitching_name = "./image_stitching"

#合併所有產生之圖像
def image_stitching(input_image):
    bg = Image.new('RGB',(1200, 300), '#000000') # 產生一張 600x300 的全黑圖片
    # 載入兩張影像
    img1 = Image.open(input_image)
    img2 = Image.open(save_smoke_semantic_image_name + ".jpg")
    img3 = Image.open(save_image_binary_name + ".jpg")
    img4 = Image.open(save_image_overlap_name + ".png")


    # 檢查兩張影像大小是否一致
    # print(img1.size)
    # print(img2.size)

    # 指定目標圖片大小
    imgSize = (256,256)

    # 改變影像大小
    img1=img1.resize(imgSize)
    img2=img2.resize(imgSize)
    img3=img3.resize(imgSize)

    img1 = ImageOps.expand(img1, 20, '#ffffff')  # 擴張邊緣，產生邊框
    img2 = ImageOps.expand(img2, 20, '#ffffff')  # 擴張邊緣，產生邊框
    img3 = ImageOps.expand(img3, 20, '#ffffff')  # 擴張邊緣，產生邊框
    img4 = ImageOps.expand(img4, 20, '#ffffff')  # 擴張邊緣，產生邊框    
    bg.paste(img1, (0, 0))
    bg.paste(img2, (300, 0))
    bg.paste(img3, (600, 0))
    bg.paste(img4, (900, 0))

    #bg.show()
    bg.save(save_image_stitching_name+ ".jpg")

    return

#訓練出的特徵圖融合原圖
def image_overlap(input_image):
    img1 = Image.open(input_image)
    img1 = img1.convert('RGBA')
    img2 = Image.open(save_smoke_semantic_image_name + ".jpg")

    #img2轉二值化
    gray = img2.convert('L')
    threshold = 200

    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)

    binary = gray.point(table, '1')
    binary.save(save_image_binary_name + ".jpg")
    img2 = binary.convert('RGBA')


    # 指定目標圖片大小
    imgSize = (256,256)

    # 改變影像大小
    img1=img1.resize(imgSize)
    img2=img2.resize(imgSize)

    L,H = img2.size
    black_background = (0, 0, 0, 255)

    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img2.getpixel(dot)
            if color_1 == black_background:
                color_1 = color_1[:-1] + (0,)   #逗號是用於創造一個(tuple)
                img2.putpixel(dot,color_1)
            else:
                color_1 = (255,0,0,) + color_1[3:]  #逗號是用於創造一個(tuple)
                img2.putpixel(dot,color_1)
            
    #img2.show()
    #疊合影像
    blendImg = Image.blend(img1, img2 , alpha = 0.2)
    #顯示影像
    #blendImg.show()
    blendImg.save(save_image_overlap_name + ".png")
    
    return

device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
print(f"Training on device {device}.")

#主函式 
def single_smoke_semantic_test(input,model_input):
    smoke_input_image = read_image(input)
    # print(smoke_input_image.shape)
    transform = transforms.Resize([256, 256])
    smoke_input_image = transform(smoke_input_image)
    # print(smoke_input_image.shape)
    smoke_input_image = (smoke_input_image)/255.0
    smoke_input_image  = smoke_input_image.unsqueeze(0).to(device)
    output_f19, output_f34 = smoke_semantic(smoke_input_image,model_input)
    torchvision.utils.save_image(output_f34 ,save_smoke_semantic_image_name + ".jpg")

    image_overlap(input)
    image_stitching(input)

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help="path to input image")      #如果在terminal輸入檔案的名稱有"("  ")"請改寫為  "\("   "\)"
    ap.add_argument('-m','--model_path', required=True, help="load model path")
    args = vars(ap.parse_args())
    
    single_smoke_semantic_test(args['image'],args['model_path'])