import argparse
import test


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source",type = str,required=True, help="path to test video path")
    ap.add_argument('-m',"--model_path" ,required=True, help="load model path")
    args = vars(ap.parse_args())

    #test.video.smoke_segmentation(args['source'],args['model_path'])
    #test.single_picture.smoke_segmentation(args['source'],args['model_path'])
    #test.multiple_pictures.smoke_segmentation(args['source'],args['model_path'])

    
