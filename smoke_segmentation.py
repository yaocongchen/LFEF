import argparse
import os
from visualization_codes import inference_single_picture,inference_multiple_pictures,inference_video

def smoke_segmentation(save_img_or_video = False):
    source = args['source']

    if os.path.isdir(source):
        inference_multiple_pictures.smoke_segmentation(args['source'],args['model_path'])
    else:
        root,extension = os.path.splitext(source)
        print(type(extension))
        if extension in ['.jpg','.png']:
            print("1")
            inference_single_picture.smoke_segmentation(args['source'],args['model_path'])
        elif extension in ['.mp4', '.avi']:
            print("2")
            inference_video.smoke_segmentation(args['source'],args['model_path'])

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source",type = str,default = '/home/yaocong/Experimental/speed_smoke_segmentation/test_files/smoke_video.mp4',required=False, help="path to test video path")
    ap.add_argument('-m',"--model_path" ,default = '/home/yaocong/Experimental/speed_smoke_segmentation/checkpoint/bs8e150/final.pth ',required=False, help="load model path")
    args = vars(ap.parse_args())

    smoke_segmentation()
    #inference_single_picture.smoke_segmentation(args['source'],args['model_path'])
    #inference_multiple_pictures.smoke_segmentation(args['source'],args['model_path'])
    #inference_video.smoke_segmentation(args['source'],args['model_path'])

    
