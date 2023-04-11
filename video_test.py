import cv2
from inference import smoke_semantic
import torch
import torchvision
import torch.nn as nn
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

S = nn.Sigmoid()

# Main function 主函式
def video_smoke_semantic_test(video_path,model_input):
    
    cap = cv2.VideoCapture(video_path)
    # 設定擷取影像的尺寸大小
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #print(cv2.getBuildInformation())
    #Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./output.mp4', fourcc, 30.0, (256,256),3)

    while cap.isOpened():
        ret,frame = cap.read()
        #if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        #水平上下反轉影像
        #frame = cv2.flip(frame,0)
        #write the flipped frame
        



        
        process_frame = frame
        
        process_frame = cv2.resize(process_frame,(256,256),interpolation = cv2.INTER_AREA)    #插值

        process_frame = process_frame.astype('float32')      # Normalized 歸一化
        process_frame = process_frame / 255.0


        video_frame = torch.from_numpy(process_frame).float()
        video_frame=video_frame.permute(2,0,1)

        smoke_input_image  = video_frame.unsqueeze(0).to(device)  #add batch
        output = smoke_semantic(smoke_input_image,model_input)
        #output = S(output)
        #print(output)
        output_np=output.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).detach().numpy()   # remove batch
        if output_np.shape[0] == 1:
            output_np = (output_np[0] + 1)/2.0 *255.0

        #im = Image.fromarray(output_np)

        #im.show()
        cv2.imwrite('test_mask.jpg',output_np)
        #print(output_np.shape)
        output_np = cv2.resize(output_np,(1280,720),interpolation = cv2.INTER_AREA)    #插值
        cv2.imshow('frame',output_np)
        cv2.imshow('frame1',frame)
        #out.write(frame)

        #print(output_np.shape)

        if cv2.waitKey(1) == ord('q'):
            break

    #Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindow

    return 


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-tv", "--test_video",default = "/home/yaocong/Experimental/speed_smoke_segmentation/smoke_video.mp4",required=True, help="path to test video path")
    ap.add_argument('-m','--model_path' ,required=True, help="load model path")
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