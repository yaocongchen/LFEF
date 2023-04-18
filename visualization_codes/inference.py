import torch
import torchvision
import torch.optim
from torchvision.io import read_image
from zmq import device
import sys
from torchvision import transforms
sys.path.append("..")
from models import erfnet

def smoke_semantic(input_image,model_path,device):
	model = erfnet.Net(1).to(device)
	model.load_state_dict(torch.load(model_path))     
	model.eval()
	output = model(input_image)   # Import model 導進模型
	return output

if __name__ == "__main__":

	device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))
	print(f"Inference on device {device}.")

	smoke_input_image = read_image('/home/yaocong/Experimental/speed_smoke_segmentation/test_files/123.jpg')
	model_path = '/home/yaocong/Experimental/speed_smoke_segmentation/checkpoint/bs8e150/final.pth'
	transform = transforms.Resize([256, 256])
	smoke_input_image = transform(smoke_input_image)
	smoke_input_image = (smoke_input_image)/255.0
	smoke_input_image  = smoke_input_image.unsqueeze(0).to(device)

	output = smoke_semantic(smoke_input_image,model_path,device)
	print("output:",output.shape)
	torchvision.utils.save_image(output,"inference" + ".jpg")