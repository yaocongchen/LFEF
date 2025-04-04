#%%
import numpy as np
import gradio as gr
import torch
import time
import os
import PIL.Image as Image
from typing import Any, Dict, Tuple, Union, Optional

#定位到主目錄
import sys
sys.path.append("..")

import val
import models.LFEF as network_model  # import self-written models 引入自行寫的模型
import utils
from visualization_codes import inference_single_picture
from utils.val_setup_utils import folders_and_files_name

model_name = str(network_model)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

Model_folder = "../trained_models/"
MODEL_LOG = Model_folder + "log.txt"


def choose_model_path(model_file: str) -> str:
    if model_file == "last.pth":
        return Model_folder + "last.pth"
    elif model_file == "best.pth":
        return Model_folder + "best.pth"

def load_model(args: Dict[str, Any]) -> Tuple[torch.nn.Module, int, int, int]:
    model = network_model.Net().to(device)
    c = utils.metrics.Calculate(model)
    model_size = c.get_model_size()
    flops, params = c.get_params()
    # model = torch.compile(model)  #pytorch2.0編譯功能(舊GPU無法使用)
    # torch.set_float32_matmul_precision('high')
    model.load_state_dict(torch.load(args["model_path"], map_location=device))
    model.eval()
    return model, model_size, flops, params

def evaluate_dataset(args: Dict[str, Any]) -> Tuple[float, float, float, float, int, int, int, int, int, int]:
    names = folders_and_files_name()
    model, model_size, flops, params = load_model(args)
    time_start = time.time()
    Avg_loss, Avg_miou, Avg_mSSIM, Avg_hd = val.smoke_segmentation(args, names, device, model)
    time_end = time.time()
    Avg_loss = round(Avg_loss, 4)
    Avg_miou = round(Avg_miou, 1)
    Avg_mSSIM = round(Avg_mSSIM, 1)        
    total_image = len(os.listdir(args["test_images"]))
    fps, time_min,time_sec = utils.metrics.report_fps_and_time(total_image, time_start, time_end)
    return Avg_loss, Avg_miou, Avg_mSSIM, Avg_hd, fps, time_min ,time_sec, model_size, flops, params

def prepare_args(model_file: str, operation_input: str) -> Dict[str, Any]:
    MODEL_PATH = choose_model_path(model_file)
    return {
        "batch_size": 1,
        "num_workers": 1,
        "model_path": MODEL_PATH,
        "wandb_name": "no",
        "test_images": f"/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/{operation_input}/images/",
        "test_masks": f"/home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/{operation_input}/masks/",
    }


def evaluate_and_report_dataset(model_file: str, operation_input: str) -> Optional[Tuple[float, float, float, float, str, str, int, int, int, str, str]]:
    if model_file in ["last.pth", "best.pth"]:
        use_model_file = model_file
        if operation_input in ["DS01", "DS02", "DS03", "Real"]:
            use_Data_Source = operation_input
            args = prepare_args(model_file,operation_input)
            Avg_loss, Avg_miou, Avg_mSSIM, Avg_hd, fps, time_min ,time_sec, model_size, flops, params= evaluate_dataset(args)
            return Avg_loss, Avg_miou, Avg_mSSIM, Avg_hd, f"{fps} fps", f"{time_min}m {time_sec}s", model_size, flops, params, use_model_file, use_Data_Source
        else:
            gr.Warning("Please choice your data source")
            return
    else:
        gr.Warning("Please choice your model file")
        return

def process_and_display_image(model_file: str, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    if model_file in ["last.pth", "best.pth"]:
        use_model_file = model_file
        MODEL_PATH = choose_model_path(model_file)
        args = {"model_path": MODEL_PATH,}
        time_train = []
        i = 0
        names = inference_single_picture.files_name()
        model, model_size, flops, params = load_model(args)
        inference_single_picture.smoke_segmentation(
            input=image,
            model=model,
            device=device,
            names=names,
            time_train=time_train, 
            i=i
        )
        smoke_semantic_image = np.array(Image.open("./results/process_image/smoke_semantic.jpg"))
        image_stitching_image = np.array(Image.open("./results/process_image/image_stitching.jpg"))
        image_overlap_image = np.array(Image.open("./results/process_image/image_overlap.png"))
        return smoke_semantic_image,image_stitching_image,image_overlap_image,use_model_file
    else:
        gr.Warning("Please choice your model file")
        return

def update_model_and_report_time() -> str:
    os.system("bash ../scripts/check_trained_model_update.sh")
    log_time = os.path.getmtime(MODEL_LOG)
    log_time_localtime = time.localtime(log_time)
    log_time_new = time.strftime("%Y-%m-%d %H:%M:%S", log_time_localtime)
    return log_time_new

with gr.Blocks() as demo:
    gr.Markdown("# Speed Smoke Segmentation Demo",)
    update_model_button = gr.Button("Update model !")
    
    with gr.Row():
        status = gr.Textbox(label="Model update time")
        model_file = gr.Radio(["last.pth", "best.pth"], label="Model File")
        use_model_file = gr.Textbox(label="Use Model File")    
    gr.Markdown("## Choice your data source")
    
    with gr.Tab("Test_Dataset"):
        with gr.Row():
            operation_input = gr.Radio(["DS01", "DS02", "DS03", "Real"], label="Data Source")
            use_Data_Source = gr.Textbox(label="Use Data Source")
        with gr.Row():
            model_size = gr.Textbox(label="Model Size",info="(MB)")
            flops = gr.Textbox(label="FLOPs",info="(G)")
            params = gr.Textbox(label="Params",info="(M)")
        with gr.Row():
            with gr.Column():
                mIoU = gr.Textbox(label="Avg mIoU",info="(%)")
                mSSIM = gr.Textbox(label="Avg mSSIM",info = "(%)")
                hd = gr.Textbox(label="Avg Hausdorff Distance",info="(pixel)")
            with gr.Column():
                loss = gr.Textbox(label="Avg Loss",info="(%)")
                fps = gr.Textbox(label="FPS",info="(frames/s)")
                spend_time = gr.Textbox(label="Spend Time")
        evaluate_and_report_dataset_button = gr.Button("Start !")

    with gr.Tab("Your_Image"):
        with gr.Column():
            image_input = gr.Image(label="Input Image",type="numpy")
            with gr.Row():
                image_smoke_semantic = gr.Image(label="smoke semantic image",type="numpy")   
                image_overlap = gr.Image(label="image overlap image",type="numpy")
            with gr.Column():
                image_stitching = gr.Image(label="image stitching image",type="numpy")
        image_button = gr.Button("Start !")

    update_model_button.click(update_model_and_report_time,outputs=status)
    evaluate_and_report_dataset_button.click(evaluate_and_report_dataset, inputs=[model_file,operation_input], outputs=[loss, mIoU, mSSIM, hd, fps, spend_time, model_size, flops, params, use_model_file, use_Data_Source])
    image_button.click(process_and_display_image, inputs=[model_file,image_input], outputs=[image_smoke_semantic, image_stitching, image_overlap, use_model_file])
    
if __name__ == "__main__":
    # x = process_and_display_image(Image.open("/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/img/1_3.jpg"))
    
    demo.launch(share=True)
#%%