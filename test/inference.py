import torch
import torchvision
import torch.optim
from torchvision.io import read_image
from zmq import device
import sys
from torchvision import transforms
sys.path.append("..")
from models import erfnet

device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
print(f"Training on device {device}.")

def smoke_semantic(input_image,model_path):
	model = erfnet.Net(1).to(device)
	model.load_state_dict(torch.load(model_path))     
	model.eval()
	output = model(input_image)   # Import model 導進模型
	return output

if __name__ == "__main__":
	
	smoke_input_image = read_image('/home/yaocong/Experimental/speed_smoke_segmentation/123.jpg')
	transform = transforms.Resize([256, 256])
	smoke_input_image = transform(smoke_input_image)
	output = smoke_semantic(smoke_input_image,'/home/yaocong/Experimental/speed_smoke_segmentation/checkpoint/bs32e150/final.pth')
	print("output_f34:",output.shape)
	torchvision.utils.save_image(output,"inference" + ".jpg")