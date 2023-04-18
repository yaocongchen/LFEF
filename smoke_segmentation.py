import torch
import argparse
import os
import time
from visualization_codes import inference_single_picture,inference_multiple_pictures,inference_video

def smoke_segmentation(device,save_img_or_video = False):
    source = args['source']

    if os.path.isdir(source):

        names = inference_multiple_pictures.folders_and_files_name()
        total_image = len(os.listdir(args["source"]))
        time_start = time.time()
        inference_multiple_pictures.smoke_segmentation(args['source'],args['model_path'],device,names)
        time_end = time.time()
        spend_time = int(time_end-time_start) 
        time_min = spend_time // 60 
        time_sec = spend_time % 60
        print('totally cost:',f"{time_min}m {time_sec}s")
        #print(total_image)

        # Calculate FPS
        print("FPS:{:.1f}".format(total_image/(time_end-time_start)))

    else:
        root,extension = os.path.splitext(source)

        if extension in ['.jpg','.png']:

            names=inference_single_picture.files_name()
            inference_single_picture.smoke_segmentation(args['source'],args['model_path'],device,names)
        elif extension in ['.mp4', '.avi']:
            binary_mode = True
            inference_video.smoke_segmentation(args['source'],args['model_path'],device,binary_mode)

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source",type = str,default = '/home/yaocong/Experimental/speed_smoke_segmentation/test_files/smoke_video.mp4',required=False, help="path to test video path")
    ap.add_argument('-m',"--model_path" ,default = '/home/yaocong/Experimental/speed_smoke_segmentation/checkpoint/bs8e150/final.pth ',required=False, help="load model path")
    args = vars(ap.parse_args())

    device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
    print(f"smoke_segmentation on device {device}.")

    smoke_segmentation(device)


    
