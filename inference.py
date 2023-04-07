import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import time
from torchvision.io import read_image
from zmq import device
import dataset
import lightssd as lightssd
import numpy
import numpy as np
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt

device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
print(f"Training on device {device}.")

def smoke_semantic(input_image,model_path):
	model = lightssd.Net().to(device)
	model.load_state_dict(torch.load(model_path))     
	#model.eval()
	output_f19, output_f34 = model(input_image)   # Import model 導進模型
	return output_f19, output_f34

if __name__ == "__main__":
	smoke_input_image = read_image('/home/yaocong/Experimental/speed_smoke_segmentation/123.jpg')
	transform = transforms.Resize([256, 256])
	smoke_input_image = transform(smoke_input_image)
	output_f19, output_f34 = smoke_semantic(smoke_input_image,'/home/yaocong/Experimental/speed_smoke_segmentation/checkpoint/bs32e150/final.pth')
	print("output_f34:",output_f34.shape)
	print("output_f19:",output_f19.shape)
	torchvision.utils.save_image(output_f34 ,"inference" + ".jpg")