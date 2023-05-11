import cv2
from visualization_codes.inference import smoke_semantic
import visualization_codes.image_process_utils as image_process
import torch
import argparse
import time
from PIL import Image
import numpy as np
import os
import threading
from copy import deepcopy

def save(video_W:int,video_H:int,video_FPS):
    if not os.path.exists("./" + "results"):
        os.makedirs("./" + "results")
    localtime = time.localtime()
    save_file_name = time.strftime("%Y-%m-%d_%I:%M:%S_%p", localtime)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(f'./results/{save_file_name}.mp4', fourcc, video_FPS, (video_W,video_H),3)   #mp4 only RGB
    return output

def image_pre_processing(input):
    process_frame = input  
    process_frame = cv2.resize(process_frame,(256,256),interpolation = cv2.INTER_AREA)    #插值
    process_frame = process_frame.astype('float32')      # Normalized 歸一化
    process_frame = process_frame / 255.0

    video_frame = torch.from_numpy(process_frame).float()
    output=video_frame.permute(2,0,1).contiguous()

    return output

# thread_lock = threading.RLock()
# thread_exit = False

# class myThread(threading.Thread):
#     def __init__(self,video_path,img_height,img_width):
#         super(myThread,self).__init__()
#         self.video_path = video_path
#         self.img_height = img_height
#         self.img_width = img_width
#         self.frame = np.zeros((img_height,img_width, 3), dtype=np.uint8)

#     def get_frame(self):
#         return deepcopy(self.frame)
    
#     def run(self):
#         global thread_exit

#         if self.video_path == '0':
#             self.video_path=int(self.video_path)

#         cap = cv2.VideoCapture(self.video_path)

#         # 設定擷取影像的尺寸大小
#         video_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         video_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         video_FPS = cap.get(cv2.CAP_PROP_FPS)

#         while not thread_exit:
#             ret,frame = cap.read()
#             if ret:
#                 thread_lock.acquire()
#                 self.frame = frame
#                 thread_lock.release()
#             else:
#                 thread_exit = True    
#         cap.release()
#         return video_W,video_H,video_FPS


# def smoke_segmentation(video_path:str,model_input:str,device:torch.device,binary_mode:bool,save_video:str,show_video:str):
#     global thread_exit
#     print("binary_mode:",binary_mode)
#     print("save_video:",save_video)
#     print("show_video:",show_video)
#     img_height = 256
#     img_width = 256

#     thread = myThread(video_path,img_height,img_width)
#     thread.start()
    
#     start_time = time.time()
#     counter = 0

#     while not thread_exit:
#         counter += 1

#         thread_lock.acquire()
#         frame = thread.get_frame()
#         thread_lock.release()
#         video_W,video_H,video_FPS = thread.run()

#         if save_video == 'True':
#             out = save(video_W,video_H,video_FPS)
        
#         video_frame = image_pre_processing(frame)
#         smoke_input_image  = video_frame.unsqueeze(0).to(device)  #add batch
#         output = smoke_semantic(smoke_input_image,model_input,device)
#         output_np=output.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to("cpu", torch.uint8).detach().numpy()   # remove batch
#         output_np = cv2.resize(output_np,(video_W,video_H),interpolation = cv2.INTER_AREA)    #插值
#         output_np = Image.fromarray(output_np)

#         # output_np to binarization output_np轉二值化
#         binary_image = image_process.gray_to_binary(output_np)
        
#         if binary_mode == True:
#             output_np_RGBA = binary_image.convert('RGBA')
#         else:    
#             output_np_RGBA = output_np.convert('RGBA')

#         frame = cv2.resize(frame,(video_W,video_H),interpolation = cv2.INTER_AREA)    #插值
#         frame_image = Image.fromarray(frame)
#         frame_RGBA = frame_image.convert('RGBA')
        
#         blendImage = image_process.overlap(frame_RGBA,output_np_RGBA,read_method = "OpenCV_BGRA")
#         output_np = blendImage.convert('RGB')
#         output_np = np.asarray(output_np)

#         print("Video_FPS: ",counter / (time.time() - start_time))
#         counter = 0
#         start_time = time.time()

        
#         if save_video == 'True':
#             out.write(output_np)
            
#         if show_video == 'True':
#             cv2.imshow('frame',output_np)
#             #cv2.imshow('frame1',frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             thread_exit = True   
#     thread.join()

#     #Release everything if job is finished
#     if save_video == 'True':
#         out.release()
#     cv2.destroyAllWindows()

#     return 


#Main function 主函式
def smoke_segmentation(video_path:str,model_input:str,device:torch.device,binary_mode:bool,save_video:str,show_video:str):
    print("binary_mode:",binary_mode)
    print("save_video:",save_video)
    print("show_video:",show_video)

    start_time = time.time()
    counter = 0
    
    if video_path == '0':
        video_path=int(video_path)
    cap = cv2.VideoCapture(video_path)

    # 設定擷取影像的尺寸大小
    video_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_FPS = cap.get(cv2.CAP_PROP_FPS)
    #print(cv2.getBuildInformation())
    #Define the codec and create VideoWriter object
    if save_video == 'True':
        out = save(video_W,video_H,video_FPS)

    idx = 0
    freq =1
    while cap.isOpened():
        ret,frame = cap.read()
        #if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        counter += 1

        video_frame = image_pre_processing(frame)
        smoke_input_image  = video_frame.unsqueeze(0).to(device)  #add batch
        output = smoke_semantic(smoke_input_image,model_input,device)
        output_np=output.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to("cpu", torch.uint8).detach().numpy()   # remove batch
        output_np = cv2.resize(output_np,(video_W,video_H),interpolation = cv2.INTER_AREA)    #插值
        output_np = Image.fromarray(output_np)

        # output_np to binarization output_np轉二值化
        binary_image = image_process.gray_to_binary(output_np)
        
        if binary_mode == True:
            output_np_RGBA = binary_image.convert('RGBA')
        else:    
            output_np_RGBA = output_np.convert('RGBA')

        frame_image = Image.fromarray(frame)
        frame_RGBA = frame_image.convert('RGBA')
        
        blendImage = image_process.overlap(frame_RGBA,output_np_RGBA,read_method = "OpenCV_BGRA")
        output_np = blendImage.convert('RGB')
        output_np = np.asarray(output_np)

        print("FPS: ",counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()

        if save_video == 'True':
            out.write(output_np)
            
        if show_video == 'True':
            cv2.imshow('frame',output_np)
            #cv2.imshow('frame1',frame)

            if cv2.waitKey(1) == ord('q'):
                break
#====================================================

        # idx += 1
        # ret = cap.grab()

        # #ret,frame = cap.read()
        # #if frame is read correctly ret is True
        # if not ret:
        #     print("Can't receive frame (stream end?). Exiting ...")
        #     break
        
        # if idx % freq == 1:
            
        #     ret, frame = cap.retrieve()
        #     if frame is None: #exist broken frame
        #         break
        #     else:

        #         counter += freq

        #         video_frame = image_pre_processing(frame)
        #         smoke_input_image  = video_frame.unsqueeze(0).to(device)  #add batch
        #         output = smoke_semantic(smoke_input_image,model_input,device)
        #         output_np=output.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to("cpu", torch.uint8).detach().numpy()   # remove batch
        #         output_np = cv2.resize(output_np,(video_W,video_H),interpolation = cv2.INTER_AREA)    #插值
        #         output_np = Image.fromarray(output_np)

        #         # output_np to binarization output_np轉二值化
        #         binary_image = image_process.gray_to_binary(output_np)
                
        #         if binary_mode == True:
        #             output_np_RGBA = binary_image.convert('RGBA')
        #         else:    
        #             output_np_RGBA = output_np.convert('RGBA')

        #         frame_image = Image.fromarray(frame)
        #         frame_RGBA = frame_image.convert('RGBA')
                
        #         blendImage = image_process.overlap(frame_RGBA,output_np_RGBA,read_method = "OpenCV_BGRA")
        #         output_np = blendImage.convert('RGB')
        #         output_np = np.asarray(output_np)

        #         print("FPS: ",counter / (time.time() - start_time))
        #         counter = 0
        #         start_time = time.time()

        #         if save_video == 'True':
        #             out.write(output_np)
                    
        #         if show_video == 'True':
        #             cv2.imshow('frame',output_np)
        #             #cv2.imshow('frame1',frame)

        #             if cv2.waitKey(1) == ord('q'):
        #                 break
#====================================================
    #Release everything if job is finished
    cap.release()
    if save_video == 'True':
        out.release()
    cv2.destroyAllWindows()

    return 

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-vs", "--video_source",type = str,default = '/home/yaocong/Experimental/speed_smoke_segmentation/test_files/Dry_leaf_smoke_02.avi',required=True, help="path to test video path")
    ap.add_argument('-m',"--model_path" ,default = '/home/yaocong/Experimental/speed_smoke_segmentation/checkpoint/bs8e150/final.pth',required=True, help="load model path")
    args = vars(ap.parse_args())

    device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
    print(f"video on device {device}.")

    binary_mode = False
    print("binary_mode:",binary_mode)

    save_video = True
    smoke_segmentation(args["video_source"],args['model_path'],device,binary_mode,save_video)