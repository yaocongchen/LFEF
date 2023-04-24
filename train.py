# %%
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os 
import argparse
import time
from tqdm import tqdm
from torch.autograd import Variable
import wandb
import torch.onnx
from torch.utils.data import DataLoader

#import self-written modules
import models.erfnet as network_model                    # import self-written models 引入自行寫的模型
import utils
onnx_img_image = []

def check_have_GPU():
    # Check have GPU device 確認是否有GPU裝置 
    if args['device']=='GPU':
        print("====> Use gpu id:'{}'".format(args['gpus']))
        os.environ["CUDA_VISIBLE_DEVICES"] = args['gpus']
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --device")    #例外事件跳出

def check_number_of_GPUs(model):
    if args['device']=='GPU':

        #args.gpu_nums = 1
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count()=",torch.cuda.device_count())
            #args.gpu_nums = torch.cuda.device_count()
            model = torch.nn.DataParallel(model).cuda() #multi-card data parallel
            device = torch.device('cuda')
        else:
            print("Single GPU for training")
            model = model.cuda() #1-card data parallel
            device = torch.device('cuda')
    else:
        print("CPU for training")
        model = model.cpu()   #use cpu data parallel
        device = torch.device('cpu')

    return device

def set_save_dir_names():
    args['save_dir'] = ( args['save_dir'] + '/' + 'bs' + str(args['batch_size']) + 'e' + str(args['epochs']) + '/')
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

def checkpoint_training(model):
    if os.path.isfile(args['resume']):    # There is a specified file in the path 路徑中有指定檔案
        checkpoint = torch.load(args['resume'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        print("=====> load checkpoint '{}' (epoch {})".format(args['resume'], checkpoint['epoch']))
    else:
        print("=====> no checkpoint found at '{}'".format(args['resume']))

def wandb_information(model_size,flops,params,model):
    wandb.init(
        # set the wandb project where this run will be logged
        project="lightssd-project",
        name = args["wandb_name"],
        # track hyperparameters and run metadata
        config={
        "Model_size":model_size,
        "FLOPs":flops,
        "Parameters":params,
        "train_images": args["train_images"],
        "train_masks": args["train_masks"],
        "device": args["device"],
        "gpus":args["gpus"],
        "batch_size": args["batch_size"],
        "num_workers": args["num_workers"],
        "epochs":args["epochs"],
        "learning_rate":args["learning_rate"],
        "save_dir":args["save_dir"],
        "resume":args["resume"],
        }
    )

    wandb.config.epochs = args["epochs"]
    wandb.config.batch_size = args["batch_size"]
    wandb.config.learning_rate = args["learning_rate"]
    #wandb.config.architecture = "resnet"

    #Log gradients and model parameters
    wandb.watch(model)

def time_processing(spend_time):
    time_day = spend_time // 86400
    spend_time = spend_time % 86400
    time_hour = spend_time // 3600
    spend_time = spend_time % 3600
    time_min = spend_time // 60
    time_sec = spend_time % 60
    
    time_dict = {}
    time_dict['time_day'] = time_day
    time_dict['time_hour'] = time_hour
    time_dict['time_min'] = time_min
    time_dict['time_sec'] = time_sec

    return time_dict

def train_epoch(model,training_data_loader,device,optimizer,epoch):
    model.train()
    cudnn.benchmark= False
    count=0
    # Training loop 訓練迴圈 
    pbar = tqdm((training_data_loader),total=len(training_data_loader))
    #for iteration,(img_image, mask_image) in enumerate(training_data_loader):
    for img_image, mask_image in pbar:
        img_image = img_image.to(device)
        mask_image = mask_image.to(device)
        onnx_img_image=img_image

        img_image = Variable(img_image, requires_grad=True)    # Variable storage data supports almost all tensor operations, requires_grad=True: Derivatives can be obtained, and the backwards method can be used to calculate and accumulate gradients
        mask_image = Variable(mask_image, requires_grad=True)  # Variable存放資料支援幾乎所有的tensor操作,requires_grad=True:可求導數，方可使用backwards的方法計算並累積梯度

        output = model(img_image)
        
        optimizer.zero_grad()     # Clear before loss.backward() to avoid gradient residue 在loss.backward()前先清除，避免梯度殘留
        
        loss = utils.loss.CustomLoss(output, mask_image)
        acc = utils.metrics.acc_miou(output,mask_image)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.1)
        optimizer.step()

        pbar.set_description(f"trian_epoch [{epoch}/{args['epochs']}]")
        pbar.set_postfix(train_loss=loss.item(),train_acc=acc.item())
        if args["wandb_name"]!="no":
            wandb.log({"train_loss": loss.item(),"train_acc": acc.item()})

        # Graphical archive of the epoch test set 
        # epoch 測試集中的圖示化存檔
        count +=1
        if not os.path.exists("./training_data_captures/"):
            os.makedirs("./training_data_captures/")
        torchvision.utils.save_image(torch.cat((mask_image,output),0), "./training_data_captures/" +str(count)+".jpg")

def valid_epoch(model,validation_data_loader,device,epoch):
    # Validation loop 驗證迴圈
    count=0
    model.eval()
    pbar = tqdm((validation_data_loader),total=len(validation_data_loader))
    for img_image,mask_image in pbar:
        img_image = img_image.to(device)
        mask_image = mask_image.to(device)
        
        output = model(img_image)

        loss = utils.loss.CustomLoss(output, mask_image)
        acc = utils.metrics.acc_miou(output,mask_image)

        pbar.set_description(f"val_epoch [{epoch}/{args['epochs']}]")
        pbar.set_postfix(val_loss=loss.item(),val_acc=acc.item())
        
        if args["wandb_name"]!="no":
            wandb.log({"val_loss": loss.item(),"val_acc": acc.item()})

        # Graphical archive of the epoch test set 
        # epoch 測試集中的圖示化存檔
        count +=1
        if not os.path.exists("./validation_data_captures/"):
            os.makedirs("./validation_data_captures/")
        torchvision.utils.save_image(torch.cat((mask_image,output),0), "./validation_data_captures/" +str(count)+".jpg")

def train():
    check_have_GPU()
    # The cudnn function library assists in acceleration(if you encounter a problem with the architecture, please turn it off)
    # Cudnn函式庫輔助加速(如遇到架構上無法配合請予以關閉)
    cudnn.enabled = False

    # Model import 模型導入
    model = network_model.Net(1)

    # Calculation model size parameter amount and calculation amount
    # 計算模型大小、參數量與計算量
    c = utils.metrics.Calculate(model)
    model_size = c.get_model_size()
    flops,params = c.get_params()

    # Set up the device for training 
    # 設定用於訓練之裝置
    device = check_number_of_GPUs(model)
        
    set_save_dir_names()

    # Import data導入資料
    training_data = utils.dataset.DataLoaderSegmentation(args['train_images'],
                                                args['train_masks'])
    validation_data = utils.dataset.DataLoaderSegmentation(args['train_images'],
                                                args['train_masks'],mode = 'val')
    training_data_loader = DataLoader(training_data ,batch_size= args['batch_size'], shuffle = True, num_workers = args['num_workers'], pin_memory = True, drop_last=True)
    validation_data_loader = DataLoader(validation_data, batch_size = args['batch_size'], shuffle = True, num_workers = args['num_workers'], pin_memory = True, drop_last=True)

    # Import optimizer導入優化器   
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['learning_rate']), weight_decay=0.0001)

    start_epoch = 1     # Initial epoch 初始epoch值

    # Checkpoint training 斷點訓練      
    if args['resume']:
        checkpoint_training(model)

    #wandb.ai
    if args["wandb_name"]!="no":
        wandb_information(model_size,flops,params,model)
        
    time_start = time.time()      # Training start time 訓練開始時間

    for epoch in range(start_epoch, args['epochs']+1):
        
        train_epoch(model,training_data_loader,device,optimizer,epoch)
        valid_epoch(model,validation_data_loader,device,epoch)

        # Save model 模型存檔              
        model_file_name = args['save_dir'] + 'model_' + str(epoch) + '.pth'
        model_file_nameonnx = args['save_dir'] + 'onnxmodel_' + str(epoch) + '.onnx'
        state = model.state_dict()
        if epoch > args['epochs'] - 10 :
            torch.save(state, model_file_name)
            #torch.onnx.export(model, onnx_img_image, model_file_nameonnx, verbose=False)
        elif not epoch % 20:
            torch.save(state, model_file_name)
            #torch.onnx.export(model, onnx_img_image, model_file_nameonnx, verbose=False)

    torch.save(state, args['save_dir'] + 'final' +  '.pth')
    wandb.save(args['save_dir'] + 'final' +  '.pth')
    #torch.onnx.export(model, onnx_img_image, args['save_dir'] + 'final' +  '.onnx', verbose=False)

    # Calculation of end time end elapsed time 
    # 計算結束時間與花費時間     
    time_end = time.time()
    spend_time = int(time_end-time_start)
    time_dict =time_processing(spend_time)
    print('totally cost:',f"{time_dict['time_day']}d {time_dict['time_hour']}h {time_dict['time_min']}m {time_dict['time_sec']}s")

if __name__=="__main__":

    ap = argparse.ArgumentParser()
    dsd
    ap.add_argument('-ti', '--train_images',default="/home/yaocong/Experimental/pytorch_model/dataset/train/images/" , help="path to hazy training images")
    ap.add_argument('-tm', '--train_masks',default= "/home/yaocong/Experimental/pytorch_model/dataset/train/masks/",  help="path to mask")

    # ap.add_argument('-ti', '--train_images',default="C:/Users/user/OneDrive/桌面/speed_smoke_segmentation/dataset/train/images/" , help="path to hazy training images")
    # ap.add_argument('-tm', '--train_masks',default= "C:/Users/user/OneDrive/桌面/speed_smoke_segmentation/dataset/train/masks/",  help="path to mask")

    # ap.add_argument('-ti', '--train_images',default="/home/yaocong/Experimental/Dataset/Smoke-Segmentation/Dataset/Train/Additional/Imag/" , help="path to hazy training images")
    # ap.add_argument('-tm', '--train_masks',default= "/home/yaocong/Experimental/Dataset/Smoke-Segmentation/Dataset/Train/Additional/Mask/",  help="path to mask")
    
    # ap.add_argument('-ti', '--train_images',default="/home/yaocong/Experimental/Dataset/SMOKE5K_dataset/SMOKE5K/SMOKE5K/train/img/" , help="path to hazy training images")
    # ap.add_argument('-tm', '--train_masks',default= "/home/yaocong/Experimental/Dataset/SMOKE5K_dataset/SMOKE5K/SMOKE5K/train/gt/",  help="path to mask")

    # ap.add_argument('-ti', '--train_images',default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/training_data/blendall/" , help="path to hazy training images")
    # ap.add_argument('-tm', '--train_masks',default= "/home/yaocong/Experimental/Dataset/SYN70K_dataset/training_data/gt_blendall/",  help="path to mask")
    
    ap.add_argument('-bs','--batch_size',type=int, default = 2, help="set batch_size")
    ap.add_argument('-nw','--num_workers' ,type=int,default = 1 , help="set num_workers")
    ap.add_argument('-e', '--epochs', type = int , default=150,  help="number of epochs for training")
    ap.add_argument('-lr', '--learning_rate', type = float ,default=0.0001, help="learning rate for training")
    ap.add_argument('-savedir','--save_dir', default= "./checkpoint/", help = "directory to save the model snapshot")
    ap.add_argument('-device' ,default='GPU' , help =  "running on CPU or GPU")
    ap.add_argument('-gpus', type= str ,default = "0" , help = "defualt GPU devices(0,1)")
    ap.add_argument('-resume',type= str ,default= "/home/yaocong/Experimental/My_pytorch_model/checkpoint/model_1.pth", 
                        help = "use this file to load last checkpoint for continuing training")    #Use this flag to load last checkpoint for training
    ap.add_argument('-wn','--wandb_name',type = str ,default = "no" ,help = "wandb test name,but 'no' is not use wandb")

    args = vars(ap.parse_args())  #Use vars() to access the value of ap.parse_args() like a dictionary 使用vars()是為了能像字典一樣訪問ap.parse_args()的值

    train()
    
    wandb.finish()
    # if args["wandb_name"]!="no":
    #     # Define sweep config
    #     sweep_configuration = {
    #         'method': 'random',
    #         'name' : 'sweep',
    #         'metric' : {'goad': 'maximize' ,'name':'val_acc'},
    #         'parameters':
    #         {
    #             'batch_size' : {'values' : [16,32,64]},
    #             'epochs' : {'values' : [5,10,15,150]},
    #             'lr' : {'max':0.1,'min':0.0001}
    #         }
    #     }
        
    #     #Initialize sweep by passing in config. (Optional) Provide a name of the project.
    #     sweep_id = wandb.sweep(sweep= sweep_configuration,project='lightssd-project')
    #     wandb.agent(sweep_id, function=train(args), count=10)
    #     wandb.finish()
    # else:
    #     train(args)