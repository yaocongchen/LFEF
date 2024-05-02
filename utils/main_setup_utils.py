import os
import wandb
import argparse

def set_save_dir_names(args):
    # args["save_dir"] = f'{args["save_dir"]}/bs{args["batch_size"]}e{args["epochs"]}/'
    args["save_dir"] = f"{args['save_dir']}/"
    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])

def create_model_state_dict(epoch, model, optimizer, mean_loss, mean_miou, save_mean_miou):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": mean_loss,
        "miou": mean_miou,
        "best_miou": save_mean_miou,
    }
    return state

def time_processing(spend_time):
    time_dict = {}
    time_dict["time_day"], spend_time = divmod(spend_time, 86400)
    time_dict["time_hour"], spend_time = divmod(spend_time, 3600)
    time_dict["time_min"], time_dict["time_sec"] = divmod(spend_time, 60)

    return time_dict

def wandb_information(args, model_name, model_size, flops, params, model, train_images, train_masks):
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

def parse_arguments():
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
        default="Host_DS0123",
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
        help="Name of the W&B run. Use 'no' to disable W&B.",
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
        "-svil",
        "--save_validation_image_last",
        type=str,
        default="no",
        help="Save the last validation image. Use 'no' to disable this feature.",
    )
    ap.add_argument(
        "-svib",
        "--save_validation_image_best",
        type=str,
        default="no",
        help="Save the best validation image. Use 'no' to disable this feature.",
    )
    args = vars(
        ap.parse_args()
    )  # Use vars() to access the value of ap.parse_args() like a dictionary 使用vars()是為了能像字典一樣訪問ap.parse_args()的值

    return args