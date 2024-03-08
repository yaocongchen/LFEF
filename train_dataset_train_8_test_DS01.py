###########################################################################
# Created by: Yao-Cong,Chen
# Email: yaocongchen@outlook.com
# Copyright (c) 2024 Yao-Cong,Chen. All rights reserved.
###########################################################################
# %%
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import configparser
import argparse
import time
from tqdm import tqdm
from torch.autograd import Variable
import wandb
import torch.onnx
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CyclicLR

# import self-written modules
import models.CGNet_2_erfnet31_13_3113_oneloss_inv_attention as network_model  # import self-written models 引入自行寫的模型
import utils

CONFIG_FILE = "import_dataset_path.cfg"

onnx_img_image = []

model_name = str(network_model)
print("model_name:", model_name)

def check_have_GPU():
    # Check have GPU device 確認是否有GPU裝置
    if args["device"] == "GPU":
        print("====> Use gpu id:'{}'".format(args["gpus"]))
        os.environ["CUDA_VISIBLE_DEVICES"] = args["gpus"]
        if not torch.cuda.is_available():
            raise Exception(
                "No GPU found or Wrong gpu id, please run without --device"
            )  # 例外事件跳出


def check_number_of_GPUs(model):
    if args["device"] == "GPU":
        # args.gpu_nums = 1
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count()=", torch.cuda.device_count())
            # args.gpu_nums = torch.cuda.device_count()
            model = torch.nn.DataParallel(model).cuda()  # multi-card data parallel
            device = torch.device("cuda")
        else:
            print("Single GPU for training")
            model = model.cuda()  # 1-card data parallel
            device = torch.device("cuda")
    else:
        print("CPU for training")
        model = model.cpu()  # use cpu data parallel
        device = torch.device("cpu")

    return model, device


def set_save_dir_names():
    # args["save_dir"] = f'{args["save_dir"]}/bs{args["batch_size"]}e{args["epochs"]}/'
    args["save_dir"] = f"{args['save_dir']}/"
    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])


def wandb_information(model_size, flops, params, model, train_images, train_masks,args):
    wandb.init(
        # Initialize wandb 初始化wandb
        # set the wandb project where this run will be logged
        project="lightssd-project-train",
        name=args["wandb_name"],
        id = args["wandb_id"],
        resume="allow",
        # track hyperparameters and run metadata
        config={
            "Model_name": model_name,
            "Model_size": model_size,
            "FLOPs": flops,
            "Parameters": params,
            "train_images": train_images,
            "train_masks": train_masks,
            "device": args["device"],
            "gpus": args["gpus"],
            "batch_size": args["batch_size"],
            "num_workers": args["num_workers"],
            "epochs": args["epochs"],
            "learning_rate": args["learning_rate"],
            "weight_decay": args["weight_decay"],
            "save_dir": args["save_dir"],
            "resume": args["resume"],
        },
    )

    # wandb.config.architecture = "resnet"

    # Log gradients and model parameters
    wandb.watch(model)


def time_processing(spend_time):
    time_dict = {}
    time_dict["time_day"], spend_time = divmod(spend_time, 86400)
    time_dict["time_hour"], spend_time = divmod(spend_time, 3600)
    time_dict["time_min"], time_dict["time_sec"] = divmod(spend_time, 60)

    return time_dict


def train_epoch(model, training_data_loader, device, optimizer, epoch):
    model.train()
    cudnn.benchmark = True
    count = 0
    n_element = 0
    mean_loss = 0
    mean_miou = 0
    # mean_dice_coef = 0
    # mean_miou_s = 0

    # Training loop 訓練迴圈
    pbar = tqdm((training_data_loader), total=len(training_data_loader))
    # for iteration,(img_image, mask_image) in enumerate(training_data_loader):
    for RGB_image, mask_image in pbar:
        img_image = RGB_image.to(device)
        mask_image = mask_image.to(device)
        # onnx_img_image = img_image

        img_image = Variable(
            img_image, requires_grad=True
        )  # Variable storage data supports almost all tensor operations, requires_grad=True: Derivatives can be obtained, and the backwards method can be used to calculate and accumulate gradients
        mask_image = Variable(
            mask_image, requires_grad=True
        )  # Variable存放資料支援幾乎所有的tensor操作,requires_grad=True:可求導數，方可使用backwards的方法計算並累積梯度

        output = model(img_image)

        # torchvision.utils.save_image(
        #     img_image, "./training_data_captures/" + "img_image" + ".jpg"
        # )
        # torchvision.utils.save_image(
        #     output, "./training_data_captures/" + "output" + ".jpg"
        # )
        # torchvision.utils.save_image(
        #     mask_image, "./training_data_captures/" + "mask_image" + ".jpg"
        # )
        

        optimizer.zero_grad()  # Clear before loss.backward() to avoid gradient residue 在loss.backward()前先清除，避免梯度殘留

        loss = utils.loss.CustomLoss(output, mask_image)
        iou = utils.metrics.IoU(output, mask_image)
        # iou_s = utils.metrics.Sigmoid_IoU(output, mask_image)
        # dice_coef = utils.metrics.dice_coef(output, mask_image, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 0.1
        )  # 梯度裁減(避免梯度爆炸或消失) 0.1為閥值
        optimizer.step()
            
        output = (output > 0.5).float()
        
        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element
        mean_miou += (iou.item() - mean_miou) / n_element
        # mean_miou_s += (iou_s.item() - mean_miou_s) / n_element
        # mean_dice_coef += (dice_coef.item() - mean_dice_coef) / n_element

        pbar.set_description(f"trian_epoch [{epoch}/{args['epochs']}]")
        pbar.set_postfix(
            train_loss=mean_loss,
            train_miou=mean_miou,
            # train_miou_s=mean_miou_s,
            # train_dice_coef=mean_dice_coef,
        )
        if args["wandb_name"] != "no":
            wandb.log(
                {
                    "train_loss": mean_loss,
                    "train_miou": mean_miou,
                    # "train_miou_s": mean_miou_s,
                    # "train_dice_coef": mean_dice_coef,
                }
            )

        # Graphical archive of the epoch test set
        # epoch 測試集中的圖示化存檔
        count += 1
        # if not epoch % 5:
        #     torchvision.utils.save_image(torch.cat((mask_image,output),0), "./training_data_captures/" +str(count)+".jpg")
    return RGB_image, mask_image, output


def valid_epoch(model, validation_data_loader, device, epoch):
    n_element = 0
    mean_loss = 0
    mean_miou = 0
    # mean_dice_coef = 0
    # mean_miou_s = 0

    model.eval()
    pbar = tqdm((validation_data_loader), total=len(validation_data_loader))
    for RGB_image, mask_image in pbar:
        img_image = RGB_image.to(device)
        mask_image = mask_image.to(device)
        onnx_img_image = img_image

        with torch.no_grad():
            output = model(img_image)

        loss = utils.loss.CustomLoss(output, mask_image)
        iou = utils.metrics.IoU(output, mask_image)
        # iou_s = utils.metrics.Sigmoid_IoU(output, mask_image)
        # dice_coef = utils.metrics.dice_coef(output, mask_image, device)

        output = (output > 0.5).float()
        
        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element
        mean_miou += (iou.item() - mean_miou) / n_element  # 別人研究出的算平均的方法
        # mean_miou_s += (iou_s.item() - mean_miou_s) / n_element  # 別人研究出的算平均的方法
        # mean_dice_coef += (dice_coef.item() - mean_dice_coef) / n_element

        pbar.set_description(f"val_epoch [{epoch}/{args['epochs']}]")
        pbar.set_postfix(
            val_loss=mean_loss,
            val_miou=mean_miou,
            # val_miou_s=mean_miou_s,
            # val_dice_coef=mean_dice_coef,
        )

        if args["wandb_name"] != "no":
            wandb.log(
                {
                    "val_loss": mean_loss,
                    "val_miou": mean_miou,
                    # "val_miou_s": mean_miou_s,
                    # "val_dice_coef": mean_dice_coef,
                }
            )

    return (
        mean_loss,
        mean_miou,
        # mean_miou_s,
        # mean_dice_coef,
        RGB_image,
        mask_image,
        output,
        onnx_img_image,
    )


def main():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
        
    train_images = args.get(args["train_images"], config.get(args["train_dataset_path"], "train_images"))
    train_masks = args.get(args["train_masks"], config.get(args["train_dataset_path"], "train_masks"))

    validation_images = args.get(args["validation_images"], config.get(args["validation_dataset_path"], "validation_images"))
    validation_masks = args.get(args["validation_masks"], config.get(args["validation_dataset_path"], "validation_masks"))

    save_mean_miou = 0
    # save_mean_miou_s = 0
    check_have_GPU()
    # The cudnn function library assists in acceleration(if you encounter a problem with the architecture, please turn it off)
    # Cudnn函式庫輔助加速(如遇到架構上無法配合請予以關閉)
    cudnn.enabled = True

    # Model import 模型導入
    model = network_model.Net()

    c = utils.metrics.Calculate(model)
    model_size = c.get_model_size()
    flops, params = c.get_params()

    model, device = check_number_of_GPUs(model)

    set_save_dir_names()

    training_data = utils.dataset.DatasetSegmentation(train_images, train_masks)
    validation_data = utils.dataset.DatasetSegmentation(
        validation_images, validation_masks, mode="all"
    )

    training_data_loader = DataLoader(
        training_data,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    validation_data_loader = DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    # 先用Adam測試模型能力
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(args["learning_rate"]), weight_decay=float(args["weight_decay"])
    )

    # scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    # 創建一個學習率排程
    # scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=100, step_size_down=100, cycle_momentum=False)    
    
    # 用SGD微調到最佳
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=float(args["learning_rate"]),
    #     momentum=0.9,
    #     weight_decay=1e-5,
    # )
    # scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=100, step_size_down=100)

    # model = torch.compile(model)  #pytorch2.0編譯功能(舊GPU無法使用)

    start_epoch = 1

    if args["wandb_name"] != "no":
        wandb_information(model_size, flops, params, model, train_images, train_masks,args)

    if args["resume"]:
        if os.path.isfile(
            args["resume"]
        ):  # There is a specified file in the path 路徑中有指定檔案
            if args["wandb_name"] != "no":
                checkpoint = torch.load(wandb.restore(args["resume"]).name)
            else:
                checkpoint = torch.load(args["resume"])
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            mean_loss = checkpoint["loss"]
            mean_miou = checkpoint["miou"]
            # mean_miou_s = checkpoint["miou_s"]
            save_mean_miou = checkpoint["best_miou"]
            # save_mean_miou_s = checkpoint["best_miou_s"]
            print(
                "=====> load checkpoint '{}' (epoch {})".format(
                    args["resume"], checkpoint["epoch"]
                )
            )
        else:
            print("=====> no checkpoint found at '{}'".format(args["resume"]))

    if not os.path.exists("./training_data_captures/"):
        os.makedirs("./training_data_captures/")
    if not os.path.exists("./validation_data_captures/"):
        os.makedirs("./validation_data_captures/")

    time_start = time.time()  # Training start time 訓練開始時間

    for epoch in range(start_epoch, args["epochs"] + 1):
        train_RGB_image, train_mask_image, train_output = train_epoch(
            model, training_data_loader, device, optimizer, epoch
        )
        torch.cuda.empty_cache()  # 刪除不需要的變數
        (
            mean_loss,
            mean_miou,
            # mean_miou_s,
            # mean_dice_coef,
            RGB_image,
            mask_image,
            output,
            onnx_img_image,
        ) = valid_epoch(model, validation_data_loader, device, epoch)

        # Save model 模型存檔

        # model_file_name = args['save_dir'] + 'model_' + str(epoch) + '.pth'
        # model_file_nameonnx = args['save_dir'] + 'onnxmodel_' + str(epoch) + '.onnx'
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": mean_loss,
            "miou": mean_miou,
            # "miou_s": mean_miou_s,
            # "dice_coef": mean_dice_coef,
            "best_miou": save_mean_miou,
            # "best_miou_s": save_mean_miou_s,
        }
        
        def save_model_and_state(model, state, mean_loss, mean_miou, onnx_img_image, path, filename):
            torch.save(state, f"{path}{filename}_checkpoint.pth")
            torch.save(model.state_dict(), f"{path}{filename}.pth")
            # torch.onnx.export(model, onnx_img_image, f"{path}{filename}.onnx", verbose=False)
            if args["wandb_name"] != "no":
                wandb.log({f"{filename}_loss": mean_loss, f"{filename}_miou": mean_miou})
                wandb.save(f"{path}{filename}_checkpoint.pth", base_path="./")
                wandb.save(f"{path}{filename}.pth", base_path="./")
                # wandb.save(f"{path}{filename}.onnx", base_path="./")

        def save_and_log_image(image, path, filename):
            full_path = f"{path}/{filename}.jpg"
            torchvision.utils.save_image(image, full_path)
            if args["wandb_name"] != "no":
                wandb.log({filename: wandb.Image(full_path)})

        save_model_and_state(model, state,  mean_loss, mean_miou, onnx_img_image,args["save_dir"], "last")

        with open(f"{args['save_dir']}/log.txt", "w") as f:
            f.write(f"{model_name}\n"
                    f"train_images: {train_images}\n"
                    f"train_masks: {train_masks}\n"
                    f"batchsize: {args['batch_size']}\n"
                    f"num_workers: {args['num_workers']}\n"
                    f"epochs: {args['epochs']}\n"
                    f"learning_rate: {args['learning_rate']}\n"
                    f"weight_decay: {args['weight_decay']}\n"
                    f"update time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")

        if args["save_train_image"] != "no":
            save_and_log_image(train_RGB_image, "./training_data_captures", "train_RGB_image")
            save_and_log_image(train_mask_image, "./training_data_captures", "train_mask_image")
            save_and_log_image(train_output, "./training_data_captures", "train_output")

        if args["save_validation_image_last"] != "no":
            save_and_log_image(RGB_image, "./validation_data_captures", "last_RGB_image")
            save_and_log_image(mask_image, "./validation_data_captures", "last_mask_image")
            save_and_log_image(output, "./validation_data_captures", "last_output")


        if mean_miou > save_mean_miou:
            print("best_loss: %.3f , best_miou: %.3f" % (mean_loss, mean_miou))
            save_model_and_state(model, state,  mean_loss, mean_miou, onnx_img_image,args["save_dir"], "best")
            
            if args["save_validation_image_bast"] != "no":
                save_and_log_image(RGB_image, "./validation_data_captures", "best_RGB_image")
                save_and_log_image(mask_image, "./validation_data_captures", "best_mask_image")
                save_and_log_image(output, "./validation_data_captures", "best_output")

            save_mean_miou = mean_miou

        scheduler.step()

    time_end = time.time()
    spend_time = int(time_end - time_start)
    time_dict = time_processing(spend_time)
    print(
        "totally cost:",
        f"{time_dict['time_day']}d {time_dict['time_hour']}h {time_dict['time_min']}m {time_dict['time_sec']}s",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-train_dataset",
        "--train_dataset_path",
        default="Host_SYN70K",
        help="use dataset path",
    )
    ap.add_argument(
        "-validation_dataset",
        "--validation_dataset_path",
        default="Host_DS01",
        help="use test dataset path",
    )

    ap.add_argument(
        "-ti",
        "--train_images",
        help="path to hazy training images",
    )
    ap.add_argument(
        "-tm",
        "--train_masks",
        help="path to mask",
    )
    ap.add_argument(
        "-vi",
        "--validation_images",
        help="path to hazy training images",
    )
    ap.add_argument(
        "-vm",
        "--validation_masks",
        help="path to mask",
    )

    ap.add_argument("-bs", "--batch_size", type=int, default=8, help="set batch_size")
    ap.add_argument("-nw", "--num_workers", type=int, default=1, help="set num_workers")
    ap.add_argument(
        "-e", "--epochs", type=int, default=150, help="number of epochs for training"
    )
    ap.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate for training",
    )    
    ap.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=0.00001,
        help="weight decay for training",
    )
    ap.add_argument(
        "-savedir",
        "--save_dir",
        default="./trained_models/",
        help="directory to save the model snapshot",
    )
    ap.add_argument("-device", default="GPU", help="running on CPU or GPU")
    ap.add_argument("-gpus", type=str, default="0", help="defualt GPU devices(0,1)")
    ap.add_argument(
        "-resume",
        type=str,
        default="/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/last_checkpoint_sample.pth",
        help="use this file to load last checkpoint for continuing training",
    )  # Use this flag to load last checkpoint for training
    ap.add_argument(
        "-wn",
        "--wandb_name",
        type=str,
        default="no",
        help="wandb test name,but 'no' is not use wandb",
    )
    ap.add_argument(
        "-wid",
        "--wandb_id",
        type=str,
        default=None,
        help="wandb id",
    )
    ap.add_argument(
        "-sti",
        "--save_train_image",
        type=str,
        default="no",
        help="wandb test name,but 'no' is not use wandb",
    )
    ap.add_argument(
        "-svi",
        "--save_validation_image_last",
        type=str,
        default="no",
        help="wandb test name,but 'no' is not use wandb",
    )
    ap.add_argument(
        "-svib",
        "--save_validation_image_bast",
        type=str,
        default="no",
        help="wandb test name,but 'no' is not use wandb",
    )
    args = vars(
        ap.parse_args()
    )  # Use vars() to access the value of ap.parse_args() like a dictionary 使用vars()是為了能像字典一樣訪問ap.parse_args()的值

    main()

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
