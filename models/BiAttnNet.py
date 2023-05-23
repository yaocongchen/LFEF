import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
import time

class AttnTrans(nn.Module):
    def __init__(self,in_chs,out_chs):
        super().__init__()
        self.conv7_7 = nn.Conv2d(1,1,(7,7),stride = 1,padding = 3,bias =True)
        self.myrelu = nn.ReLU()

        self.mysigmoid = nn.Sigmoid()
        
        self.conv1_1 = nn.Conv2d(in_chs,in_chs,(1,1),stride = 1,bias =True)
        #TODO: NO use upsample
        #self.upsamp = nn.Upsample(size = (64,64),mode ='bilinear',align_corners = True)

        #self.upsamp = nn.Upsample(size = (28,28),mode ='bilinear',align_corners = True)



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
        #output = self.upsamp(output)
        output = self.conv1_1f(output)

        return output
    
class detail_branch(nn.Module):
    def __init__(self, in_chs,out_chs):
        super().__init__()
        self.AttnTrans_1 = AttnTrans(in_chs,64)
        self.AttnTrans_2 = AttnTrans(64,64)
        self.AttnTrans_3 = AttnTrans(64,64)

        self.AttnTrans_4 = AttnTrans(64,128)
        self.AttnTrans_5 = AttnTrans(128,128)
        self.AttnTrans_6 = AttnTrans(128,128)

        self.AttnTrans_7 = AttnTrans(128,256)
        self.AttnTrans_8 = AttnTrans(256,256)
        self.AttnTrans_9 = AttnTrans(256,256)

        self.AttnTrans_10 = AttnTrans(256,512)
        self.AttnTrans_11 = AttnTrans(512,512)
        self.AttnTrans_12 = AttnTrans(512,out_chs)

    def forward(self, input):
        out = self.AttnTrans_1(input)
        out = self.AttnTrans_2(out)
        out = self.AttnTrans_3(out)

        out = self.AttnTrans_4(out)
        out = self.AttnTrans_5(out)
        out_1 = self.AttnTrans_6(out)

        out_2 = self.AttnTrans_7(out_1)
        out_2 = self.AttnTrans_8(out_2)
        out_2 = self.AttnTrans_9(out_2)

        out_3 = self.AttnTrans_10(out_2)
        out_3 = self.AttnTrans_11(out_3)
        out_3 = self.AttnTrans_12(out_3)

        return out_1,out_2,out_3

class context_branch(nn.Module):
    def __init__(self, in_chs,out_chs):
        super().__init__()
        self.conv_in_48 = nn.Conv2d(in_chs,48,(3,3),stride=2,bias = True)
        
        self.conv_48_48 = nn.Conv2d(48,48,(3,3),stride=2,bias = True)
        self.conv_48_48_no_stride = nn.Conv2d(48,48,(3,3),stride=1,bias = True)

        self.conv_48_96 = nn.Conv2d(48,96,(3,3),stride=2,bias = True)
        self.conv_96_96 = nn.Conv2d(96,96,(3,3),stride=1,bias = True)
        
        self.conv_96_192= nn.Conv2d(96,192,(3,3),stride=2,bias = True)  
        self.conv_192_192= nn.Conv2d(192,192,(3,3),stride=1,bias = True)
        
        self.conv_192_384 = nn.Conv2d(192,384,(3,3),stride=2,bias = True)
        self.conv_384_384 = nn.Conv2d(384,384,(3,3),stride=1,bias = True)

    def forward(self, input):
        out = self.conv_in_48(input)
        
        out = self.conv_48_48(out)
        out = self.conv_48_48_no_stride(out)
        
        out = self.conv_48_96(out)
        out_1 = self.conv_96_96(out)
        
        out_2 = self.conv_96_192(out_1)
        out_2 = self.conv_192_192(out_2)

        out_3 = self.conv_192_384(out_2)
        out_3 = self.conv_384_384(out_3)

        return out_1,out_2,out_3

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cb = context_branch(3,384)
        self.db = detail_branch(3,512)

        self.conv_192_32 = nn.Conv2d(192,32, (1,1) ,stride =1 ,bias = True)
        self.conv_384_32 = nn.Conv2d(384,32, (1,1) ,stride =1 ,bias = True)
        self.conv_768_32 = nn.Conv2d(768,32, (1,1) ,stride =1 ,bias = True)

        self.conv_768_32 = nn.Conv2d(768,1, (1,1) ,stride =1 ,bias = True)

        self.upsamp = nn.Upsample(size = (256,256),mode ='bilinear',align_corners = True)

    def forward(self, img):
        cb_out_1,cb_out_2,cb_out_3 = self.cb(img)
        db_out_1,db_out_2,db_out_3 = self.db(img)

        cat_out_1 = torch.cat((cb_out_1,db_out_1),dim = 1)
        cat_out_1 = self.conv_192_32(cat_out_1) 

        cat_out_2 = torch.cat((cb_out_2,db_out_2),dim = 1)
        cat_out_2 = self.conv_768_32(cat_out_2)

        cat_out_3 = torch.cat((cb_out_3,db_out_3),dim = 1)
        cat_out_3 = self.conv_768_32(cat_out_3)

        out = torch.cat((cat_out_1,cat_out_2),dim = 1)
        out = torch.cat((out,cat_out_3),dim = 1)

        return out

if __name__ == "__main__":

    img = Image.open('/home/yaocong/Experimental/speed_smoke_segmentation/test_files/123.jpg')
    transform = transforms.ToTensor()
    imgTensor = transform(img)
    imgTensor = imgTensor.unsqueeze(0)
    
    model = Net()
    x = torch.randn(16,3,256,256)
    print("x")
    print(x)
    print(imgTensor.shape)

    start_time = time.time()
    with torch.no_grad():
        output = model(x)
    print("FPS: ",1 / (time.time() - start_time))
    start_time = time.time()

    print(output)
    print(output.shape)

    summary(model, input_size=(16,3,256,256))
    #torchvision.utils.save_image(output, "testjpg.jpg")

    fwt = time.time() - start_time



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
