# %%
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import configparser
import argparse
import time
import random
from tqdm import tqdm
from torch.autograd import Variable
import wandb
import torch.onnx
from torch.utils.data import DataLoader

# import self-written modules
import models.CGNet_add_sem_cam_2loss as network_model  # import self-written models 引入自行寫的模型
import utils

CONFIG_FILE = "import_dataset_path.cfg"

onnx_img_image = []


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
    args["save_dir"] = f'{args["save_dir"]}/bs{args["batch_size"]}e{args["epochs"]}/'
    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])


def wandb_information(model_size, flops, params, model,train_images,train_masks):
    wandb.init(
        # set the wandb project where this run will be logged
        project="lightssd-project-train",
        name=args["wandb_name"],
        # track hyperparameters and run metadata
        config={
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
            "save_dir": args["save_dir"],
            "resume": args["resume"],
        },
    )

    wandb.config.epochs = args["epochs"]
    wandb.config.batch_size = args["batch_size"]
    wandb.config.learning_rate = args["learning_rate"]
    # wandb.config.architecture = "resnet"

    # Log gradients and model parameters
    wandb.watch(model)


def time_processing(spend_time):
    time_day = spend_time // 86400
    spend_time = spend_time % 86400
    time_hour = spend_time // 3600
    spend_time = spend_time % 3600
    time_min = spend_time // 60
    time_sec = spend_time % 60

    time_dict = {}
    time_dict["time_day"] = time_day
    time_dict["time_hour"] = time_hour
    time_dict["time_min"] = time_min
    time_dict["time_sec"] = time_sec

    return time_dict


def train_epoch(model, training_data_loader, device, optimizer, epoch):
    model.train()
    cudnn.benchmark = True
    count = 0
    n_element = 0
    mean_loss = 0
    mean_miou = 0
    mean_dice_coef = 0
    mean_miou_s = 0
    mean_loss_1 = 0
    mean_loss_2 = 0

    # Training loop 訓練迴圈
    pbar = tqdm((training_data_loader), total=len(training_data_loader))
    # for iteration,(img_image, mask_image) in enumerate(training_data_loader):
    for RGB_image, mask_image in pbar:
        img_image = RGB_image.to(device)
        mask_image = mask_image.to(device)
        onnx_img_image = img_image

        img_image = Variable(
            img_image, requires_grad=True
        )  # Variable storage data supports almost all tensor operations, requires_grad=True: Derivatives can be obtained, and the backwards method can be used to calculate and accumulate gradients
        mask_image = Variable(
            mask_image, requires_grad=True
        )  # Variable存放資料支援幾乎所有的tensor操作,requires_grad=True:可求導數，方可使用backwards的方法計算並累積梯度

        output, output_auxiliary = model(img_image)

        optimizer.zero_grad()  # Clear before loss.backward() to avoid gradient residue 在loss.backward()前先清除，避免梯度殘留

        loss, loss_1, loss_2 = utils.two_loss.CustomLoss(
            output, output_auxiliary, mask_image, mode="train"
        )
        iou = utils.metrics.IoU(output, mask_image, device)
        iou_s = utils.metrics.Sigmoid_IoU(output, mask_image)
        dice_coef = utils.metrics.dice_coef(output, mask_image, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 0.1
        )  # 梯度裁減(避免梯度爆炸或消失) 0.1為閥值
        optimizer.step()

        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element
        mean_loss_1 += (loss_1.item() - mean_loss_1) / n_element
        mean_loss_2 += (loss_2.item() - mean_loss_2) / n_element
        mean_miou += (iou.item() - mean_miou) / n_element
        mean_miou_s += (iou_s.item() - mean_miou_s) / n_element
        mean_dice_coef += (dice_coef.item() - mean_dice_coef) / n_element

        pbar.set_description(f"trian_epoch [{epoch}/{args['epochs']}]")
        pbar.set_postfix(
            train_loss=mean_loss,
            train_loss_1=mean_loss_1,
            train_loss_2=mean_loss_2,
            train_miou=mean_miou,
            train_miou_s=mean_miou_s,
            train_dice_coef=mean_dice_coef,
        )
        if args["wandb_name"] != "no":
            wandb.log(
                {
                    "train_loss": mean_loss,
                    "train_loss_1": mean_loss_1,
                    "train_loss_2": mean_loss_2,
                    "train_miou": mean_miou,
                    "train_miou_s": mean_miou_s,
                    "train_dice_coef": mean_dice_coef,
                }
            )

        # Graphical archive of the epoch test set
        # epoch 測試集中的圖示化存檔
        count += 1
        # if not epoch % 5:
        #     torchvision.utils.save_image(torch.cat((mask_image,output),0), "./training_data_captures/" +str(count)+".jpg")
    return RGB_image, mask_image, output


def valid_epoch(model, validation_data_loader, device, epoch):
    # Validation loop 驗證迴圈
    n_element = 0
    mean_loss = 0
    mean_miou = 0
    mean_dice_coef = 0
    mean_miou_s = 0
    mean_loss_1 = 0
    mean_loss_2 = 0

    model.eval()
    pbar = tqdm((validation_data_loader), total=len(validation_data_loader))
    for RGB_image, mask_image in pbar:
        img_image = RGB_image.to(device)
        mask_image = mask_image.to(device)

        with torch.no_grad():
            output, output_auxiliary = model(img_image)

        loss = utils.two_loss.CustomLoss(
            output, output_auxiliary, mask_image, mode="val"
        )
        iou = utils.metrics.IoU(output, mask_image, device)
        iou_s = utils.metrics.Sigmoid_IoU(output, mask_image)
        dice_coef = utils.metrics.dice_coef(output, mask_image, device)

        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element
        mean_miou += (iou.item() - mean_miou) / n_element  # 別人研究出的算平均的方法
        mean_miou_s += (iou_s.item() - mean_miou_s) / n_element  # 別人研究出的算平均的方法
        mean_dice_coef += (dice_coef.item() - mean_dice_coef) / n_element

        pbar.set_description(f"val_epoch [{epoch}/{args['epochs']}]")
        pbar.set_postfix(
            val_loss=mean_loss,
            val_miou_s=mean_miou_s,
            val_miou=mean_miou,
            val_dice_coef=mean_dice_coef,
        )

        if args["wandb_name"] != "no":
            wandb.log(
                {
                    "val_loss": mean_loss,
                    "val_miou": mean_miou,
                    "val_miou_s": mean_miou_s,
                    "val_dice_coef": mean_dice_coef,
                }
            )

    return (
        mean_loss,
        mean_miou_s,
        mean_miou,
        mean_dice_coef,
        RGB_image,
        mask_image,
        output,
    )


def main():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    if (args["train_images"] != None) and (args["train_masks"] != None):
        train_images = args["train_images"]
        train_masks = args["train_masks"] 
    else:
        train_images = config.get(args["dataset_path"],"train_images")
        train_masks = config.get(args["dataset_path"],"train_masks")

    save_mean_miou = 0
    save_mean_miou_s = 0
    check_have_GPU()
    # The cudnn function library assists in acceleration(if you encounter a problem with the architecture, please turn it off)
    # Cudnn函式庫輔助加速(如遇到架構上無法配合請予以關閉)
    cudnn.enabled = True

    # Model import 模型導入
    model = network_model.Net()

    # Calculation model size parameter amount and calculation amount
    # 計算模型大小、參數量與計算量
    c = utils.metrics.Calculate(model)
    model_size = c.get_model_size()
    flops, params = c.get_params()

    # Set up the device for training
    # 設定用於訓練之裝置
    model, device = check_number_of_GPUs(model)

    set_save_dir_names()

    # Import data導入資料
    seconds = time.time()  # Random number generation 亂數產生
    random.seed(seconds)  # 使用時間秒數當亂數種子

    training_data = utils.dataset.DataLoaderSegmentation(
        train_images, train_masks
    )

    random.seed(seconds)  # 使用時間秒數當亂數種子

    validation_data = utils.dataset.DataLoaderSegmentation(
        train_images, train_masks, mode="val"
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
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # Import optimizer導入優化器

    # 先用Adam測試模型能力
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(args["learning_rate"]), weight_decay=0.0001
    )
    # 用SGD微調到最佳
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=float(args["learning_rate"]),
    #     momentum=0.9,
    #     weight_decay=1e-5,
    # )

    # model = torch.compile(model)  #pytorch2.0編譯功能(舊GPU無法使用)

    start_epoch = 1  # Initial epoch 初始epoch值

    # Checkpoint training 斷點訓練
    if args["resume"]:
        if os.path.isfile(
            args["resume"]
        ):  # There is a specified file in the path 路徑中有指定檔案
            checkpoint = torch.load(args["resume"])
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            mean_loss = checkpoint["loss"]
            mean_miou = checkpoint["miou"]
            mean_miou_s = checkpoint["miou_s"]
            save_mean_miou = checkpoint["best_miou"]
            save_mean_miou_s = checkpoint["best_miou_s"]
            print(
                "=====> load checkpoint '{}' (epoch {})".format(
                    args["resume"], checkpoint["epoch"]
                )
            )
        else:
            print("=====> no checkpoint found at '{}'".format(args["resume"]))

    # wandb.ai
    if args["wandb_name"] != "no":
        wandb_information(model_size, flops, params, model,train_images,train_masks)

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
            mean_miou_s,
            mean_miou,
            mean_dice_coef,
            RGB_image,
            mask_image,
            output,
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
            "miou_s": mean_miou_s,
            "dice_coef": mean_dice_coef,
            "best_miou": save_mean_miou,
            "best_miou_s": save_mean_miou_s,
        }

        torch.save(state, args["save_dir"] + "last_checkpoint" + ".pth")
        torch.save(model.state_dict(), args["save_dir"] + "last" + ".pth")

        if args["save_train_image"] != "no":
            torchvision.utils.save_image(
                train_RGB_image,
                "./training_data_captures/" + "last_RGB_image_" + ".jpg",
            )
            torchvision.utils.save_image(
                train_mask_image,
                "./training_data_captures/" + "last_mask_image_" + ".jpg",
            )
            torchvision.utils.save_image(
                train_output, "./training_data_captures/" + "last_output_" + ".jpg"
            )

        if args["save_validation_image_last"] != "no":
            # torchvision.utils.save_image(torch.cat((mask_image,output),0), "./validation_data_captures/" + "last_" + ".jpg")
            torchvision.utils.save_image(
                RGB_image, "./validation_data_captures/" + "last_RGB_image_" + ".jpg"
            )
            torchvision.utils.save_image(
                mask_image, "./validation_data_captures/" + "last_mask_image_" + ".jpg"
            )
            torchvision.utils.save_image(
                output, "./validation_data_captures/" + "last_output_" + ".jpg"
            )

        # torch.onnx.export(model, onnx_img_image, args['save_dir'] + 'last' +  '.onnx', verbose=False)
        if args["wandb_name"] != "no":
            wandb.save(args["save_dir"] + "last_checkpoint" + ".pth")
            wandb.save(args["save_dir"] + "last" + ".pth")
            # Graphical archive of the epoch test set
            # epoch 測試集中的圖示化存檔
            # wandb.log({"last": wandb.Image("./validation_data_captures/" + "last_" + ".jpg")})

            if args["save_train_image"] != "no":
                wandb.log(
                    {
                        "train_RGB_image": wandb.Image(
                            "./training_data_captures/" + "last_RGB_image_" + ".jpg"
                        )
                    }
                )
                wandb.log(
                    {
                        "train_mask_image": wandb.Image(
                            "./training_data_captures/" + "last_mask_image_" + ".jpg"
                        )
                    }
                )
                wandb.log(
                    {
                        "train_output": wandb.Image(
                            "./training_data_captures/" + "last_output_" + ".jpg"
                        )
                    }
                )

            if args["save_validation_image_last"] != "no":
                wandb.log(
                    {
                        "last_RGB_image": wandb.Image(
                            "./validation_data_captures/" + "last_RGB_image_" + ".jpg"
                        )
                    }
                )
                wandb.log(
                    {
                        "last_mask_image": wandb.Image(
                            "./validation_data_captures/" + "last_mask_image_" + ".jpg"
                        )
                    }
                )
                wandb.log(
                    {
                        "last_output": wandb.Image(
                            "./validation_data_captures/" + "last_output_" + ".jpg"
                        )
                    }
                )

        if mean_miou > save_mean_miou:
            print("best_loss: %.3f , best_miou: %.3f" % (mean_loss, mean_miou))
            torch.save(state, args["save_dir"] + "best_checkpoint" + ".pth")
            torch.save(model.state_dict(), args["save_dir"] + "best" + ".pth")
            # torchvision.utils.save_image(
            #     torch.cat((mask_image, output), 0),
            #     "./validation_data_captures/" + "best" + str(count) + ".jpg",
            # )

            if args["save_validation_image_bast"] != "no":
                torchvision.utils.save_image(
                    RGB_image,
                    "./validation_data_captures/" + "best_RGB_image_" + ".jpg",
                )
                torchvision.utils.save_image(
                    mask_image,
                    "./validation_data_captures/" + "best_mask_image_" + ".jpg",
                )
                torchvision.utils.save_image(
                    output, "./validation_data_captures/" + "best_output_" + ".jpg"
                )

            if args["wandb_name"] != "no":
                wandb.log({"best_loss": mean_loss, "best_miou": mean_miou})
                wandb.save(args["save_dir"] + "best_checkpoint" + ".pth")
                wandb.save(args["save_dir"] + "best" + ".pth")
                # wandb.log({"best": wandb.Image("./validation_data_captures/" + "best" + ".jpg")})

                if args["save_validation_image_bast"] != "no":
                    wandb.log(
                        {
                            "best_RGB_image": wandb.Image(
                                "./validation_data_captures/"
                                + "best_RGB_image_"
                                + ".jpg"
                            )
                        }
                    )
                    wandb.log(
                        {
                            "best_mask_image": wandb.Image(
                                "./validation_data_captures/"
                                + "best_mask_image_"
                                + ".jpg"
                            )
                        }
                    )
                    wandb.log(
                        {
                            "best_output": wandb.Image(
                                "./validation_data_captures/" + "best_output_" + ".jpg"
                            )
                        }
                    )
            save_mean_miou = mean_miou

            if mean_miou_s > save_mean_miou_s:
                print("best_loss: %.3f , best_miou_s: %.3f" % (mean_loss, mean_miou_s))
                torch.save(
                    state, args["save_dir"] + "best_mean_miou_s_checkpoint" + ".pth"
                )
                torch.save(
                    model.state_dict(), args["save_dir"] + "best_mean_miou_s" + ".pth"
                )
                # torchvision.utils.save_image(
                #     torch.cat((mask_image, output), 0),
                #     "./validation_data_captures/" + "best" + str(count) + ".jpg",
                # )
                if args["wandb_name"] != "no":
                    wandb.log({"best_loss": mean_loss, "best_miou_s": mean_miou_s})
                    wandb.save(
                        args["save_dir"] + "best_mean_miou_s_checkpoint" + ".pth"
                    )
                    wandb.save(args["save_dir"] + "best_mean_miou_s" + ".pth")

            save_mean_miou_s = mean_miou_s
    #         torch.onnx.export(
    #             model,
    #             onnx_img_image,
    #             args["save_dir"] + "best" + ".onnx",
    #             verbose=False,
    #         )

    #     if epoch > args["epochs"] - 10:
    #         torch.save(state, model_file_name)
    #         # torch.onnx.export(model, onnx_img_image, model_file_nameonnx, verbose=False)
    #     elif not epoch % 20:
    #         torch.save(state, model_file_name)
    #         # torch.onnx.export(model, onnx_img_image, model_file_nameonnx, verbose=False)

    # torch.save(state, args["save_dir"] + "final" + ".pth")
    # wandb.save(args["save_dir"] + "final" + ".pth")
    # # torch.onnx.export(model, onnx_img_image, args['save_dir'] + 'final' +  '.onnx', verbose=False)

    # Calculation of end time end elapsed time
    # 計算結束時間與花費時間
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
        "-dataset",
        "--dataset_path",
        default="Host_SYN70K",
        help="use dataset path",
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

    ap.add_argument("-bs", "--batch_size", type=int, default=8, help="set batch_size")
    ap.add_argument("-nw", "--num_workers", type=int, default=1, help="set num_workers")
    ap.add_argument(
        "-e", "--epochs", type=int, default=150, help="number of epochs for training"
    )
    ap.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.01,
        help="learning rate for training",
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
        default="/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/last_checkpoint.pth",
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
