import torch
import torchvision
import wandb
import time
from typing import Dict, Union
from torch import Tensor
from torch.nn import Module

def save_model_and_state(args: Dict[str, Union[str, bool]], model: Module, state: Dict, mean_loss: float, mean_miou: float, onnx_img_image: Tensor, path: str, filename: str) -> None:
    torch.save(state, f"{path}{filename}_checkpoint.pth")
    torch.save(model.state_dict(), f"{path}{filename}.pth")
    # torch.onnx.export(model, onnx_img_image, f"{path}{filename}.onnx", verbose=False)
    if args["wandb_name"] != "no":
        wandb.log({f"{filename}_loss": mean_loss, f"{filename}_miou": mean_miou})
        wandb.save(f"{path}{filename}_checkpoint.pth", base_path="./")
        wandb.save(f"{path}{filename}.pth", base_path="./")
        # wandb.save(f"{path}{filename}.onnx", base_path="./")

def save_and_log_image(args: Dict[str, Union[str, bool]], image: Tensor, path: str, filename: str) -> None:
    full_path = f"{path}/{filename}.jpg"
    torchvision.utils.save_image(image, full_path)
    if args["wandb_name"] != "no":
        wandb.log({filename: wandb.Image(full_path)})

def save_experiment_details(args: Dict[str, Union[str, bool, int, float]], model_name: str, train_images: str, train_masks: str) -> None:
    with open(f"{args['model_save_dir']}/log.txt", "w") as f:
        f.write(f"{model_name}\n"
                f"train_images: {train_images}\n"
                f"train_masks: {train_masks}\n"
                f"batchsize: {args['batch_size']}\n"
                f"num_workers: {args['num_workers']}\n"
                f"epochs: {args['epochs']}\n"
                f"learning_rate: {args['learning_rate']}\n"
                f"weight_decay: {args['weight_decay']}\n"
                f"update time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")