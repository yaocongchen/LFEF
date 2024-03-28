import torch
import torchvision
import wandb
import time

def save_model_and_state(model, state, mean_loss, mean_miou, onnx_img_image, path, filename, args):
    torch.save(state, f"{path}{filename}_checkpoint.pth")
    torch.save(model.state_dict(), f"{path}{filename}.pth")
    # torch.onnx.export(model, onnx_img_image, f"{path}{filename}.onnx", verbose=False)
    if args["wandb_name"] != "no":
        wandb.log({f"{filename}_loss": mean_loss, f"{filename}_miou": mean_miou})
        wandb.save(f"{path}{filename}_checkpoint.pth", base_path="./")
        wandb.save(f"{path}{filename}.pth", base_path="./")
        # wandb.save(f"{path}{filename}.onnx", base_path="./")

def save_and_log_image(image, path, filename, args):
    full_path = f"{path}/{filename}.jpg"
    torchvision.utils.save_image(image, full_path)
    if args["wandb_name"] != "no":
        wandb.log({filename: wandb.Image(full_path)})

def save_experiment_details(args, model_name, train_images, train_masks):
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