from inference import smoke_semantic
import torch
import torchvision
import os
import argparse
from time import sleep
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import read_image
import shutil
import time
import dataset
import utils
from torch.utils.data import DataLoader

device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
#print(f"Training on device {device}.")

#設定存檔資料夾與存檔名稱  
save_smoke_semantic_dir_name = "multiple_result"
if os.path.exists("./" + save_smoke_semantic_dir_name):
    shutil.rmtree("./" + save_smoke_semantic_dir_name)      #將原有的資料夾與內容刪除
    os.makedirs("./" + save_smoke_semantic_dir_name)        #創建新的資料夾
else:
# if not os.path.exists("./" + save_smoke_semantic_dir_name):
    os.makedirs("./" + save_smoke_semantic_dir_name)
save_smoke_semantic_image_name = "smoke_semantic_image_"

#主函式 
def multiple_smoke_semantic_test(args):
    testing_data = dataset.DataLoaderSegmentation(args['test_images'],
                                                args['test_masks'],mode = 'test')
    testing_data_loader = DataLoader(testing_data ,batch_size= args['batch_size'], shuffle = True, num_workers =args['num_workers'], pin_memory = True, drop_last=True)

    count=0
    pbar = tqdm((testing_data_loader),total=len(testing_data_loader))
    for img_image,mask_image in pbar:
        img_image = img_image.to(device)
        mask_image = mask_image.to(device)

        output_f19, output_f34 = smoke_semantic(img_image,args['model_path'])
        count += 1
        torchvision.utils.save_image(torch.cat((mask_image,output_f34),0),"./" + save_smoke_semantic_dir_name + "/" + save_smoke_semantic_image_name  + f"{count}.jpg")

        loss = utils.CustomLoss(output_f19, output_f34, mask_image)
        acc = utils.acc_miou(output_f34,mask_image)


        pbar.set_postfix(test_loss=loss.item(),test_acc=acc.item())

    return 

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    #ap.add_argument("-td", "--test_directory",required=True, help="path to test images directory")
    ap.add_argument('-ti', '--test_images',default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/img/" , help="path to hazy training images")
    ap.add_argument('-tm', '--test_masks',default= "/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/mask/",  help="path to mask") 
    ap.add_argument('-bs','--batch_size',type=int, default = 8, help="set batch_size")
    ap.add_argument('-nw','--num_workers' ,type=int,default = 1 , help="set num_workers")   
    ap.add_argument('-m','--model_path' ,required=True, help="load model path")
    args = vars(ap.parse_args())

    #計算總執行時間
    time_start = time.time()
    multiple_smoke_semantic_test(args)
    time_end = time.time()
    total_image = len(os.listdir(args["test_images"]))
    spend_time = int(time_end-time_start) 
    time_min = spend_time // 60 
    time_sec = spend_time % 60
    print('totally cost:',f"{time_min}m {time_sec}s")

    #計算FPS
    print("FPS:{:.1f}".format(total_image/(time_end-time_start)))



