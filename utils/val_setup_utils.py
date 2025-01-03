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
        required=True,
        help="Path to the directory containing test images.",
    )
    ap.add_argument(
        "-tm",
        "--test_masks",
        required=True,
        help="Path to the directory containing test masks.",
    )
    ap.add_argument("-bs", "--batch_size", type=int, default=1, help="Batch size for testing.")
    ap.add_argument("-nw", "--num_workers", type=int, default=1, help="Number of workers for data loading during testing.")
    ap.add_argument("-m", "--model_path", required=False, default="./trained_models/best.pth",help="Path to the trained model to be used for testing.")
    ap.add_argument(
        "-wn",
        "--wandb_name",
        type=str,
        default="no",
        help="Name of the Weights & Biases run for testing. Use 'no' to disable Weights & Biases.",
    )
    args = vars(ap.parse_args())

    return args