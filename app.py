#%%
import numpy as np
import gradio as gr
import argparse
import torch

import test

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
        Avg_loss, Avg_miou, Avg_mSSIM = test.smoke_segmentation(device, names, args)
        return Avg_loss, f"{Avg_miou* 100:.2f}",f"{Avg_mSSIM* 100:.2f}"
    
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
        Avg_loss, Avg_miou, Avg_mSSIM = test.smoke_segmentation(device, names,args)
        return Avg_loss, f"{Avg_miou* 100:.2f}", f"{Avg_mSSIM* 100:.2f}"
    
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
        Avg_loss, Avg_miou, Avg_mSSIM = test.smoke_segmentation(device, names,args)
        return Avg_loss, f"{Avg_miou* 100:.2f}", f"{Avg_mSSIM* 100:.2f}"


with gr.Blocks() as demo:
    gr.Markdown("# Speed Smoke Segmentation Demo",)
    gr.Markdown("Choice your data source")
    with gr.Tab("SYN70K_Test_Data"):
        with gr.Row():
            operation_input = gr.Radio(["DS01", "DS02", "DS03"])
        loss = gr.Textbox(label="Avg_loss")
        mIoU = gr.Textbox(label="Avg_mIoU",info="(%)")
        mSSIM = gr.Textbox(label="Avg_mSSIM",info = "(%)")
        text_button = gr.Button("GO!")
    with gr.Tab("Your_Image"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("GO!")

    # with gr.Accordion("Open for More!"):
    #     gr.Markdown("Look at me...")

    # text_button.click(flip_text, inputs=text_input, outputs=text_output)
    # image_button.click(flip_image, inputs=image_input, outputs=image_output)
    text_button.click(SYN70k_dataset, inputs=operation_input, outputs=[loss, mIoU, mSSIM])
    
if __name__ == "__main__":
    demo.launch(share=True)
#%%