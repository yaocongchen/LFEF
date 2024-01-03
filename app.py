#%%
import numpy as np
import gradio as gr
import torch
import time
import os
import PIL.Image as Image
import numpy as np
import wandb

import test
from visualization_codes import inference_single_picture
import models.CGNet_2_erfnet31_13_3113_oneloss_add_deformable_conv as network_model  # import self-written models 引入自行寫的模型
model_name = str(network_model)

Model_folder = "/home/yaocong/Experimental/speed_smoke_segmentation/trained_models/mynet_70k_data/CGnet_erfnet3_1_1_3_test_dilated/"
def model_choice(model_file):
    if model_file == "last.pth":
        MODEL_PATH = Model_folder + "last.pth"
    elif model_file == "best.pth":
        MODEL_PATH = Model_folder + "best.pth"
    return MODEL_PATH

MODEL_LOG = Model_folder + "log.txt"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def model_load(args):
    model = network_model.Net().to(device)
    model.load_state_dict(torch.load(args["model_path"]))
    model.eval()
    return model

def test_dataset(args):
    names = test.folders_and_files_name()

    model = model_load(args)

    def calculate_and_print_fps(total_image, time_start, time_end, wandb_time_total=0):
        fps = total_image / (time_end - time_start - wandb_time_total)
        fps = round(fps, 1)
        spend_time = int(time_end - time_start - wandb_time_total)
        time_min = spend_time // 60
        time_sec = spend_time % 60

        return fps , time_min ,time_sec

    if args["wandb_name"] != "no":  # 此方式還是會誤差FPS4~5
        time_start = time.time()
        Avg_loss, Avg_miou, Avg_mSSIM, wandb_time_total= test.smoke_segmentation(model,device,names,args)
        time_end = time.time()
        Avg_loss = round(Avg_loss, 4)
        Avg_miou = round(Avg_miou, 1)
        Avg_mSSIM = round(Avg_mSSIM, 1)
        total_image = len(os.listdir(args["test_images"]))
        fps,time_min,time_sec = calculate_and_print_fps(total_image, time_start, time_end, wandb_time_total)
        wandb.log({"FPS": fps})
    else:
        time_start = time.time()
        Avg_loss, Avg_miou, Avg_mSSIM = test.smoke_segmentation(model,device,names,args)
        time_end = time.time()
        Avg_loss = round(Avg_loss, 4)
        Avg_miou = round(Avg_miou, 1)
        Avg_mSSIM = round(Avg_mSSIM, 1)        
        total_image = len(os.listdir(args["test_images"]))
        fps, time_min,time_sec =calculate_and_print_fps(total_image, time_start, time_end)

        # Avg_loss, Avg_miou, Avg_mSSIM = test.smoke_segmentation(device, names, args)
    return Avg_loss, Avg_miou,Avg_mSSIM, fps, time_min ,time_sec

def get_args(model_file,operation_input, WANDB_NAME):
    MODEL_PATH = model_choice(model_file)
    return {
        "batch_size": 1,
        "num_workers": 1,
        "model_path": MODEL_PATH,
        "wandb_name": WANDB_NAME,
        "test_images": f"/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/{operation_input}/img/",
        "test_masks": f"/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/{operation_input}/mask/",
    }


def SYN70k_dataset(model_file,operation_input,wandb_name_input):
    if model_file in ["last.pth", "best.pth"]:
        use_model_file = model_file
        if operation_input in ["DS01", "DS02", "DS03"]:
            use_Data_Source = operation_input
            if wandb_name_input != "":
                WANDB_NAME = wandb_name_input
            else:
                WANDB_NAME = "no"
            
            args = get_args(model_file,operation_input, WANDB_NAME)
            Avg_loss, Avg_miou,Avg_mSSIM, fps, time_min ,time_sec = test_dataset(args)
            return Avg_loss, Avg_miou,Avg_mSSIM, f"{fps} fps", f"{time_min}m {time_sec}s", use_model_file, use_Data_Source
        else:
            gr.Warning("Please choice your data source")
            return
    else:
        gr.Warning("Please choice your model file")
        return

def Your_image(model_file,image):
    if model_file in ["last.pth", "best.pth"]:
        use_model_file = model_file
        MODEL_PATH = model_choice(model_file)

        args = {
            "model_path": MODEL_PATH,
        }

        time_train = []
        i = 0

        names = inference_single_picture.files_name()

        model = model_load(args)

        inference_single_picture.smoke_segmentation(
            input=image,
            model=model,
            device=device,
            names=names,
            time_train=time_train, 
            i=i
        )
        smoke_semantic_image = np.array(Image.open("./results/smoke_semantic.jpg"))
        binary_image = np.array(Image.open("./results/binary.jpg"))
        image_stitching_image = np.array(Image.open("./results/image_stitching.jpg"))
        image_overlap_image = np.array(Image.open("./results/image_overlap.png"))
        
        return smoke_semantic_image,binary_image,image_stitching_image,image_overlap_image,use_model_file
    else:
        gr.Warning("Please choice your model file")
        return

def model_update():
    #執行check_trained_model_update.sh
    os.system("bash ./check_trained_model_update.sh")
    #查看log.txt的更新時間
    log_time = os.path.getmtime(MODEL_LOG)
    #轉換成localtime
    log_time_localtime = time.localtime(log_time)
    #轉換成新的時間格式(2016-05-05 20:28:54)
    log_time_new = time.strftime("%Y-%m-%d %H:%M:%S", log_time_localtime)

    return log_time_new

with gr.Blocks() as demo:
    gr.Markdown("# Speed Smoke Segmentation Demo",)
    gr.Markdown("## Choice your data source")
    update_model_button = gr.Button("Update model !")
    with gr.Row():
        status = gr.Textbox(label="Model update time")
        model_file = gr.Radio(["last.pth", "best.pth"], label="Model_File")
        use_model_file = gr.Textbox(label="Use_Model_File")

    with gr.Tab("SYN70K_Test_Data"):
        with gr.Row():
            operation_input = gr.Radio(["DS01", "DS02", "DS03"], label="Data Source")
            use_Data_Source = gr.Textbox(label="Use_Data_Source")
        loss = gr.Textbox(label="Avg_loss")
        with gr.Row():
            with gr.Column():
                mIoU = gr.Textbox(label="Avg_mIoU",info="(%)")
                mSSIM = gr.Textbox(label="Avg_mSSIM",info = "(%)")
            with gr.Column():
                fps = gr.Textbox(label="FPS",info="(frames/s)")
                spend_time = gr.Textbox(label="Spend_Time")

        with gr.Accordion("Advanced Options"):
            gr.Markdown("## Save to Weight & Biases (Optional)")
            wandb_name_input = gr.Textbox(label="wandb_name")

        SYN70K_button = gr.Button("Start !")

    with gr.Tab("Your_Image"):
        with gr.Column():
            image_input = gr.Image(label="Input_Image",type="numpy")
            with gr.Row():
                image_smoke_semantic = gr.Image(label="smoke_semantic_image",type="numpy")   
                image_binary = gr.Image(label="binary_image",type="numpy")
                image_overlap = gr.Image(label="image_overlap_image",type="numpy")
            with gr.Column():
                image_stitching = gr.Image(label="image_stitching_image",type="numpy")

        image_button = gr.Button("Start !")

    # with gr.Accordion("Open for More!"):
    #     gr.Markdown("Look at me...")

    update_model_button.click(model_update,outputs=status)
    SYN70K_button.click(SYN70k_dataset, inputs=[model_file,operation_input,wandb_name_input], outputs=[loss, mIoU, mSSIM, fps, spend_time, use_model_file, use_Data_Source])
    image_button.click(Your_image, inputs=[model_file,image_input], outputs=[image_smoke_semantic, image_binary, image_stitching, image_overlap, use_model_file])
    
if __name__ == "__main__":
    # x = Your_image(Image.open("/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/img/1_3.jpg"))
    
    demo.launch(share=True)
#%%