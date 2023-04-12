import cv2
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
from PIL import Image
import numpy as np

device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
print(f"Training on device {device}.")


binary_mode = False
print("binary_mode:",binary_mode)

# Main function 主函式
def video_smoke_semantic_test(video_path,model_input):
    
    start_time = time.time()
    counter = 0

    cap = cv2.VideoCapture(video_path)
    # 設定擷取影像的尺寸大小
    video_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_FPS = cap.get(cv2.CAP_PROP_FPS)
    #print(cv2.getBuildInformation())
    #Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./output.mp4', fourcc, video_FPS, (video_W,video_H),3)   #mp4 only RGB

    while cap.isOpened():
        ret,frame = cap.read()
        #if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        #水平上下反轉影像
        #frame = cv2.flip(frame,0)
        #write the flipped frame
        
        counter += 1
        
        process_frame = frame  
        process_frame = cv2.resize(process_frame,(256,256),interpolation = cv2.INTER_AREA)    #插值
        process_frame = process_frame.astype('float32')      # Normalized 歸一化
        process_frame = process_frame / 255.0

        video_frame = torch.from_numpy(process_frame).float()
        video_frame=video_frame.permute(2,0,1)
        smoke_input_image  = video_frame.unsqueeze(0).to(device)  #add batch
        output = smoke_semantic(smoke_input_image,model_input)
        output_np=output.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).detach().numpy()   # remove batch
        #cv2.imwrite('test_mask.jpg',output_np)

        frame_image = Image.fromarray(frame)
        frame_RGBA = frame_image.convert('RGBA')
        output_np = cv2.resize(output_np,(video_W,video_H),interpolation = cv2.INTER_AREA)    #插值
        output_np = Image.fromarray(output_np)

        # output_np to binarization output_np轉二值化
        gray = output_np.convert('L')
        threshold = 200

        table = []
        for pixel_g in range(256):
            if pixel_g < threshold:
                table.append(0)
            else:
                table.append(1)
        binary = gray.point(table, '1')
        
        if binary_mode == True:
            output_np_RGBA = binary.convert('RGBA')
        else:    
            output_np_RGBA = output_np.convert('RGBA')

        # output_np_RGBA_show = np.asarray(output_np_RGBA)
        # cv2.imshow('output_np_RGBA',output_np_RGBA_show)

        W,H = output_np_RGBA.size
        black_background = (0, 0, 0, 255)
        #white_mask = (255, 255, 255, 255)
        for h in range(H):
            for w in range(W):
                dot = (w,h)
                color_1 = output_np_RGBA.getpixel(dot)
                if color_1 == black_background:
                    color_1 = color_1[:-1] + (0,)   # Commas are used to create a (tuple) 逗號是用於創造一個(tuple)
                    output_np_RGBA.putpixel(dot,color_1)
                else:
                    color_1 = (0,0,255,) + color_1[3:]  #逗號是用於創造一個(tuple)
                    output_np_RGBA.putpixel(dot,color_1)

        # Overlay image 疊合影像
        blendImg = Image.blend(frame_RGBA, output_np_RGBA , alpha = 0.2)
        output_np = blendImg.convert('RGB')
        output_np = np.asarray(output_np)

        
        print("FPS: ",counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()

        out.write(output_np)
        cv2.imshow('frame',output_np)
        #cv2.imshow('frame1',frame)

        if cv2.waitKey(1) == ord('q'):
            break

    #Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return 


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-tv", "--test_video",type=int,required=True, help="path to test video path")
    ap.add_argument('-m',"--model_path" ,required=True, help="load model path")
    args = vars(ap.parse_args())

# Calculate the total execution time 計算總執行時間  
    time_start = time.time()
    video_smoke_semantic_test(args["test_video"],args['model_path'])
#     total_image = len(os.listdir(args["test_directory"]))
#     time_end = time.time()
#     spend_time = int(time_end-time_start) 
#     time_min = spend_time // 60 
#     time_sec = spend_time % 60
#     print('totally cost:',f"{time_min}m {time_sec}s")
#     #print(total_image)

# # Calculate FPS
#     print("FPS:{:.1f}".format(total_image/(time_end-time_start)))