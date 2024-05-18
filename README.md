# Speed smoke segmentation

<!-- Implementation of paper - [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616) -->

<div align="center">
    <a href="./">
        <img src="./model_structure_diagram/smoke_segment_structure-main.drawio.png" width="79%"/>
    </a>
</div>


## Performance 

SYN70K ( $256 \times 256$ )

| Model | Size | Param. | FLOPs | DS01 | DS02 | DS03 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **My** | **1.35MB** | **0.353M** | **0.873G** | **72.23%** | **71.95%** | **72.95%** |

## Installation

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>

``` shell
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov9 -it -v your_coco_path/:/coco/ -v your_code_path/:/yolov9 --shm-size=64g nvcr.io/nvidia/pytorch:21.11-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# go to code folder
cd /yolov9
```

</details>


## Evaluation

[`best.pt`](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt)

``` shell
# Example
python val.py -ti /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/images/ -tm /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/masks/ -m /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/best.pth

```


You will get the results:

```
model_name: <module 'models.CGNet_2_erfnet31_13_3113_oneloss_inv_attention' from '/home/yaocong/Experimental/speed_smoke_segmentation/models/CGNet_2_erfnet31_13_3113_oneloss_inv_attention.py'>
Testing on device cuda.
model size:                     1.347 MB
Computational complexity(FLOPs):   873.04 MMac
Number of parameters:           353.07 k
model path: /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/best.pth
test_data: /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/images/
Number of test Images: 1000
100%|████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:25<00:00, 39.63it/s, test_hd=12.6, test_loss=0.35, test_mSSIM=81.6, test_miou=72]
FPS:39.6
totally cost: 0m 25s
loss: 0.3498
mIoU: 72.00%
```


## Training

<!-- Data preparation -->

Single GPU training

``` shell
# Example
python main.py -bs 32 -train_dataset Host_SYN70K -e 500 -wn base_use_ConvINReLU_downsample
```
Multiple GPU training

``` shell
# Example
python main.py -bs 32 -train_dataset Host_SYN70K -e 500 -wn base_use_ConvINReLU_downsample -gpus 0,1
```

| Parameter | Description | Default Value |
| --- | --- | --- |
| -train_dataset<br>--train_dataset_path | Path to the training dataset. | Host_SYN70K |
| -validation_dataset<br>--validation_dataset_path | Path to the validation dataset. | Host_DS0123 |
| -ti<br>--train_images | Path to the directory containing training images. | None |
| -tm<br>--train_masks | Path to the directory containing training masks. | None |
| -vi<br>--validation_images | Path to the directory containing validation images. | None |
| -vm<br>--validation_masks | Path to the directory containing validation masks. | None |
| -bs<br>--batch_size | Batch size for training. | 8 |
| -nw<br>--num_workers | Number of workers for data loading. | 1 |
| -e<br>--epochs | Number of epochs for training. | 150 |
| -lr<br>--learning_rate | Learning rate for the optimizer. | 0.001 |
| -wd<br>--weight_decay | Weight decay for the optimizer. | 0.00001 |
| -savedir<br>--model_save_dir | Directory to save the trained models. | ./trained_models/ |
| -device | Device to run the training on. Choose between 'CPU' and 'GPU'. | GPU |
| -gpus | GPU devices to use for training. For multiple GPUs, separate by comma. | 0 |
| -resume | Path to the last checkpoint. Use this to resume training. | None |
| -wn<br>--wandb_name | Name of the Weights & Biases run. Use 'no' to disable Weights & Biases. | no |
| -wid<br>--wandb_id | Weights & Biases run ID. | None |
| -sti<br>--save_train_image | Save the training images. Include this argument to enable this feature. | False |
| -svil<br>--save_validation_image_last | Save the last validation image. Include this argument to enable this feature. | False |
| -svib<br>--save_validation_image_best | Save the best validation image. Include this argument to enable this feature. | False |




## Inference

<div align="center">
    <a href="./">
        <img src="./figure/horses_prediction.jpg" width="49%"/>
    </a>
</div>

### val

``` shell
# Example
python val.py -ti /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/images/ -tm /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS03/masks/ -m /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/last.pth
```

| Parameter | Description | Default Value |
| --- | --- | --- |
| -ti,<br>--test_images | Path to the directory containing test images. | None (required) |
| -tm,<br>--test_masks | Path to the directory containing test masks. | None (required) |
| -bs,<br>--batch_size | Batch size for testing. | 1 |
| -nw,<br>--num_workers | Number of workers for data loading during testing. | 1 |
| -m,<br>--model_path | Path to the trained model to be used for testing. | "./trained_models/best.pth" |
| -wn,<br>--wandb_name | Name of the Weights & Biases run for testing. Use 'no' to disable Weights & Biases. | "no" |

### Analytical model

``` shell
# Example
python inference_multiple_pictures_for_evaluate.py -td /home/yaocong/Experimental/Dataset/SYN70K_dataset/testing_data/DS01/ -m /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/best.pth
```

| Parameter | Description | Default Value |
| --- | --- | --- |
| -td,<br>--test_directory | Path to the directory containing test images. | None (required) |
| -m,<br>--model_path | Path to the trained model to be used for evaluation. | None (required) |

### Demo
``` shell
# Example
python smoke_segmentation.py -s /home/yaocong/Experimental/Dataset/smoke_video_dataset/Black_smoke_517.avi -m /home/yaocong/Experimental/speed_smoke_segmentation/trained_models/best.pth -show
```

| Parameter | Description | Default Value |
| --- | --- | --- |
| -s,<br>--source | Path to the image, video file, or directory to be tested. | None (required) |
| -m,<br>--model_path | Path to the trained model to be used for smoke segmentation. | None (required) |
| -vtf,<br>--video_to_frames | Convert the video to frames. Include this argument to enable this feature. | False |
| -save,<br>--save_video | Save the output video. Include this argument to enable this feature. | False |
| -show,<br>--show_video | Display the output video. Include this argument to enable this feature. | False |


## Citation

<!-- ```
@article{wang2024yolov9,
  title={{YOLOv9}: Learning What You Want to Learn Using Programmable Gradient Information},
  author={Wang, Chien-Yao  and Liao, Hong-Yuan Mark},
  booktitle={arXiv preprint arXiv:2402.13616},
  year={2024}
}
```

```
@article{chang2023yolor,
  title={{YOLOR}-Based Multi-Task Learning},
  author={Chang, Hung-Shuo and Wang, Chien-Yao and Wang, Richard Robert and Chou, Gene and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2309.16921},
  year={2023}
}
``` -->

## Acknowledgements

<!-- <details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/VDIGPKU/DynamicDet](https://github.com/VDIGPKU/DynamicDet)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)

</details> -->
