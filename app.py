#%%
import numpy as np
import gradio as gr
import argparse
import torch
import time
import os
import PIL.Image as Image
import numpy as np

import test
import smoke_segmentation
from visualization_codes import (
    inference_single_picture,
    inference_multiple_pictures,
    inference_video,
)
import models.CGNet_2_erfnet31_13_3113_oneloss_add_deformable_conv as network_model  # import self-written models 引入自行寫的模型

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def SYN70k_dataset(operation):
    if operation == "DS01":
        names = test.folders_and_files_name()
        args = {
            "batch_size": 1,
            "num_workers": 1,
            "model_path": "/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/bs32e150/last.pth",
            "wandb_name": "no",
            "test_images": "/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/img/",
            "test_masks": "/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/mask/",
        }

        time_start = time.time()
        Avg_loss, Avg_miou, Avg_mSSIM = test.smoke_segmentation(device, names,args)
        time_end = time.time()
        Avg_loss = round(Avg_loss, 4)
        Avg_miou = round(Avg_miou, 1)
        Avg_mSSIM = round(Avg_mSSIM, 1)
        total_image = len(os.listdir(args["test_images"]))

        fps = round(total_image / (time_end - time_start), 1)
        spend_time = int(time_end - time_start)
        time_min = spend_time // 60
        time_sec = spend_time % 60

        # Avg_loss, Avg_miou, Avg_mSSIM = test.smoke_segmentation(device, names, args)
        return Avg_loss, Avg_miou,Avg_mSSIM, f"{fps} fps", f"{time_min}m {time_sec}s"
    
    elif operation == "DS02":
        names = test.folders_and_files_name()
        args = {
            "batch_size": 1,
            "num_workers": 1,
            "model_path": "/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/bs32e150/last.pth",
            "wandb_name": "no",
            "test_images": "/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS02/img/",
            "test_masks": "/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS02/mask/",
        }
        time_start = time.time()
        Avg_loss, Avg_miou, Avg_mSSIM = test.smoke_segmentation(device, names,args)
        time_end = time.time()
        Avg_loss = round(Avg_loss, 4)
        Avg_miou = round(Avg_miou, 1)
        Avg_mSSIM = round(Avg_mSSIM, 1)
        total_image = len(os.listdir(args["test_images"]))

        fps = round(total_image / (time_end - time_start), 1)
        spend_time = int(time_end - time_start)
        time_min = spend_time // 60
        time_sec = spend_time % 60

        # Avg_loss, Avg_miou, Avg_mSSIM = test.smoke_segmentation(device, names, args)
        return Avg_loss, Avg_miou,Avg_mSSIM, f"{fps} fps", f"{time_min}m {time_sec}s"
    
    elif operation == "DS03":
        names = test.folders_and_files_name()
        args = {
            "batch_size": 1,
            "num_workers": 1,
            "model_path": "/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/bs32e150/last.pth",
            "wandb_name": "no",
            "test_images": "/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/img/",
            "test_masks": "/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/mask/",
        }
        time_start = time.time()
        Avg_loss, Avg_miou, Avg_mSSIM = test.smoke_segmentation(device, names,args)
        time_end = time.time()
        Avg_loss = round(Avg_loss, 4)
        Avg_miou = round(Avg_miou, 1)
        Avg_mSSIM = round(Avg_mSSIM, 1)
        total_image = len(os.listdir(args["test_images"]))

        fps = round(total_image / (time_end - time_start), 1)
        spend_time = int(time_end - time_start)
        time_min = spend_time // 60
        time_sec = spend_time % 60

        # Avg_loss, Avg_miou, Avg_mSSIM = test.smoke_segmentation(device, names, args)
        return Avg_loss, Avg_miou,Avg_mSSIM, f"{fps} fps", f"{time_min}m {time_sec}s"
    elif operation == None:
        gr.Warning("Please choice your data source")
        return

def Your_image(image):
    time_train = []
    i = 0

    args = {
        "source" : image,
        "model_path": "/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/bs32e150/last.pth",
        "save_video": False,
        "show_video": False,
    }
    names = inference_single_picture.files_name()

    model = network_model.Net().to(device)
    model.load_state_dict(torch.load(args["model_path"]))

    model.eval()
    inference_single_picture.smoke_segmentation(
        input=args["source"],
        model=model,
        device=device,
        names=names,
        time_train=time_train, 
        i=i
    )
    smoke_semantic_image = np.array(Image.open("/home/yaocong/Experimental/speed_smoke_segmentation/results/smoke_semantic.jpg"))
    binary_image = np.array(Image.open("/home/yaocong/Experimental/speed_smoke_segmentation/results/binary.jpg"))
    image_stitching_image = np.array(Image.open("/home/yaocong/Experimental/speed_smoke_segmentation/results/image_stitching.jpg"))
    image_overlap_image = np.array(Image.open("/home/yaocong/Experimental/speed_smoke_segmentation/results/image_overlap.png"))
    
    return smoke_semantic_image,binary_image,image_stitching_image,image_overlap_image

with gr.Blocks() as demo:
    gr.Markdown("# Speed Smoke Segmentation Demo",)
    gr.Markdown("Choice your data source")
    with gr.Tab("SYN70K_Test_Data"):
        with gr.Row():
            operation_input = gr.Radio(["DS01", "DS02", "DS03"], label="Data Source")
        loss = gr.Textbox(label="Avg_loss")
        with gr.Row():
            with gr.Column():
                mIoU = gr.Textbox(label="Avg_mIoU",info="(%)")
                mSSIM = gr.Textbox(label="Avg_mSSIM",info = "(%)")
            with gr.Column():
                fps = gr.Textbox(label="FPS",info="(frames/s)")
                spend_time = gr.Textbox(label="Spend_Time")
        SYN70K_button = gr.Button("GO!")

    with gr.Tab("Your_Image"):
        with gr.Column():
            image_input = gr.Image(label="Input_Image",type="pil")
            with gr.Row():
                image_smoke_semantic = gr.Image(label="smoke_semantic_image",type="numpy")   
                image_binary = gr.Image(label="binary_image",type="numpy")
                image_overlap = gr.Image(label="image_overlap_image",type="numpy")
            with gr.Column():
                image_stitching = gr.Image(label="image_stitching_image",type="numpy")

        image_button = gr.Button("GO!")

    # with gr.Accordion("Open for More!"):
    #     gr.Markdown("Look at me...")

    # text_button.click(flip_text, inputs=text_input, outputs=text_output)
    # image_button.click(flip_image, inputs=image_input, outputs=image_output)
    SYN70K_button.click(SYN70k_dataset, inputs=operation_input, outputs=[loss, mIoU, mSSIM, fps, spend_time])
    image_button.click(Your_image, inputs=image_input, outputs=[image_smoke_semantic, image_binary, image_stitching, image_overlap])
    
if __name__ == "__main__":
    # x = Your_image(Image.open("/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/img/1_3.jpg"))
    
    demo.launch(share=True)
#%%