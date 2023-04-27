import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary

class AttnTrans(nn.Module):
    def __init__(self,in_chs,out_chs):
        super().__init__()
        self.conv7_7 = nn.Conv2d(1,1,(7,7),stride = 1,padding = 3,bias =True)
        self.myrelu = nn.ReLU()

        self.mysigmoid = nn.Sigmoid()
        
        self.conv1_1 = nn.Conv2d(3,3,(1,1),stride = 1,bias =True)
        #TODO: NO use upsample
        #self.upsamp = nn.Upsample(size = (64,64),mode ='bilinear',align_corners = True)

        self.conv1_1f = nn.Conv2d(in_chs,out_chs,(1,1),stride = 1,bias = True)

    def forward(self,input):
        #Channel Avg
        channel_avg = torch.mean(input,dim = 1)
        channel_avg = channel_avg.unsqueeze(1)
        channel_avg = self.conv7_7  (channel_avg)
        channel_avg = self.myrelu(channel_avg)
        channel_avg = self.conv7_7 (channel_avg)
        channel_avg = self.mysigmoid(channel_avg)
        #spatial Avg
        spatial_avg = torch.mean(input,dim = [2,3])
        spatial_avg = spatial_avg.unsqueeze(2)
        spatial_avg = spatial_avg.unsqueeze(3)
        spatial_avg = self.conv1_1(spatial_avg)
        spatial_avg = self.myrelu(spatial_avg)
        spatial_avg = self.conv1_1(spatial_avg)
        spatial_avg = self.mysigmoid(spatial_avg)

        output = input * channel_avg
        output = output *spatial_avg

        output = self.conv1_1f(output)

        return output

# class context_branch(nn.Module):
#     def __init__(self, in_chs,out_chs):
#         super().__init__()

#     def forward(self, input):

# class detail_branch(nn.Module):
#     def __init(self, in_chs,out_chs):
#         super().__init__()
#         self.cin_cout = nn.Sequential(
#             self.conv = nn.Conv2d(in_chs,48,(3,3),stride = 2 ,padding=1,bias = True)
#             self.covn = nn.Conv2d(48,,(3,3),stride = 2 ,padding=1,bias = True)
        
#     def forward(self, input):
        

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.AttnTrans = AttnTrans(3,48)



        

    def forward(self, img):
        out = self.AttnTrans(img)


        return out


if __name__ == "__main__":

    img = Image.open('/home/yaocong/Experimental/speed_smoke_segmentation/test_files/123.jpg')
    transform = transforms.ToTensor()
    imgTensor = transform(img)
    imgTensor = imgTensor.unsqueeze(0)
    
    model = Net()
    x = torch.randn(1,3,10,10)
    print("x")
    print(x)
    print(imgTensor.shape)

    output = model(imgTensor)
    #summary(model, input_size=(16,3,256,256))
    #torchvision.utils.save_image(output, "testjpg.jpg")

    print(output)
    print(output.shape)


    # #Channel Avg
    # channel_avg = torch.mean(x,dim = 1)
    # print("channel_avg")
    
    # print(channel_avg)
    # mysig = F.sigmoid(channel_avg)
    # print(mysig)
    # print(channel_avg.shape)
    # #spatial Avg

    # totol = torch.mean(x,dim = [2,3])
    # print("spatial_avg")
    # print(totol.shape)
    # print(totol)
