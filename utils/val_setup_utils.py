import os
import wandb
import argparse
import shutil
from typing import Dict, Any, Union

def folders_and_files_name() -> Dict[str, str]:
    save_smoke_semantic_dir_name = "./results/testing_multiple"
    shutil.rmtree(save_smoke_semantic_dir_name, ignore_errors=True)
    os.makedirs(save_smoke_semantic_dir_name)

    return {
        "smoke_semantic_dir_name": save_smoke_semantic_dir_name,
    }

def wandb_information(args: Dict[str, Any], model_name: str, model_size: Union[int, float], flops: Union[int, float], params: Union[int, float]) -> None:
    wandb.init(
        project="lightssd-project-test",
        name=args["wandb_name"],
        config={
            "Model": model_name,
            "Model_size": model_size,
            "FLOPs": flops,
            "Parameters": params,
            **{k: args[k] for k in ["test_images", "test_masks", "batch_size", "num_workers"]}
        }
    )

def parse_arguments() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-ti",
        "--test_images",
        default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/images/",
        help="path to hazy training images",
    )
    ap.add_argument(
        "-tm",
        "--test_masks",
        default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/masks/",
        help="path to mask",
    )
    # ap.add_argument(
    #     "-ti",
    #     "--test_images",
    #     default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/Real/images/",
    #     help="path to hazy training images",
    # )
    # ap.add_argument(
    #     "-tm",
    #     "--test_masks",
    #     default="/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/Real/masks/",
    #     help="path to mask",
    # )
    # ap.add_argument(
    #     "-ti",
    #     "--test_images",
    #     default="/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/120k/img/",
    #     help="path to hazy training images",
    # )
    # ap.add_argument(
    #     "-tm",
    #     "--test_masks",
    #     default="/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/120k/gt/",
    #     help="path to mask",
    # )
    # ap.add_argument(
    #     "-ti",
    #     "--test_images",
    #     default="/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/t/img/",
    #     help="path to hazy training images",
    # )
    # ap.add_argument(
    #     "-tm",
    #     "--test_masks",
    #     default="/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/t/gt/",
    #     help="path to mask",
    # )
    # ap.add_argument(
    #     "-ti",
    #     "--test_images",
    #     default="/home/yaocong/Experimental/Dataset/SMOKE5K_dataset/SMOKE5K/SMOKE5K/test/img/",
    #     help="path to hazy training images",
    # )
    # ap.add_argument(
    #     "-tm",
    #     "--test_masks",
    #     default="/home/yaocong/Experimental/Dataset/SMOKE5K_dataset/SMOKE5K/SMOKE5K/test/gt_/",
    #     help="path to mask",
    # )
    ap.add_argument("-bs", "--batch_size", type=int, default=1, help="set batch_size")
    ap.add_argument("-nw", "--num_workers", type=int, default=1, help="set num_workers")
    ap.add_argument("-m", "--model_path", required=False, default="/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/last.pth",help="load model path")
    ap.add_argument(
        "-wn",
        "--wandb_name",
        type=str,
        default="no",
        help="wandb test name,but 'no' is not use wandb",
    )
    args = vars(ap.parse_args())

    return args