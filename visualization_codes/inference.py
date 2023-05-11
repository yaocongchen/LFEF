import torch
import torchvision
import torch.optim
from torchvision.io import read_image
from zmq import device
import sys
from torchvision import transforms
sys.path.append("..")
from models import erfnet
import time

def smoke_semantic(input_image,model_path,device):
	model = erfnet.Net(1).to(device)
	model.load_state_dict(torch.load(model_path))     
	model.eval()

	time_start = time.time()
	output = model(input_image)   # Import model 導進模型

	torch.cuda.synchronize()    #wait for cuda to finish (cuda is asynchronous!)
	
	time_end = time.time()
	spend_time = int(time_end-time_start) 
	# time_min = spend_time // 60 
	# time_sec = spend_time % 60
	#print('totally cost:',f"{time_min}m {time_sec}s")
	#print(total_image)

	# Calculate FPS
	print("Model_FPS: {:.1f}".format(1/(time_end-time_start)))

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