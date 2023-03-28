#%%
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchinfo import summary

k=2

def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()    #contiguous: the memory location remains unchanged
    x2 = x[:, c1:, :, :].contiguous()    #contiguous：記憶體位置不變

    return x1, x2    

def channel_shuffle(x,groups):
    batchsize, num_channels, height, width = x.data.size()
    
    channels_per_group = num_channels // groups
    
    # reshape (torch)
    x = x.view(batchsize,groups,
        channels_per_group,height,width)
    
    x = torch.transpose(x,1,2).contiguous()   #Transpose 轉置
    
    # flatten
    x = x.view(batchsize,-1,height,width)
    
    return x

class DownUnit(nn.Module):

    def __init__(self, in_chs,out_chs):
        super().__init__()

        self.conv1 = nn.Conv2d(in_chs,out_chs-in_chs,kernel_size = 3 ,stride=2, padding =1, bias=True)
        self.maxpl = nn.MaxPool2d(kernel_size = 2, stride = 2, padding=0)
        self.prelu = nn.PReLU()
        self.batch_norm = nn.BatchNorm2d(out_chs-in_chs,eps=1e-3)
    
    def forward(self, x):
        main = self.conv1(x)
        main = self.batch_norm(main)

        ext = self.maxpl(x)
        # Concatenate branche
        out = torch.cat((main, ext), dim = 1)
        
        # Apply batch normalization
        out = self.prelu(out)
        
        return out
    
class CSSAM(nn.Module):

    def __init__(self, in_ch, out_ch, dilated):
        super().__init__()
        in_ch_2 =in_ch//2

        self.conv3x1 = nn.Conv2d(in_ch_2, in_ch_2, kernel_size = (3,1) ,padding="same",dilation=dilated, bias=True)

        self.conv1x3 = nn.Conv2d(in_ch_2, in_ch_2, kernel_size = (1,3) ,padding="same",dilation=dilated, bias=True)

        self.bn1 = nn.BatchNorm2d(in_ch_2,eps=1e-3)

        self.conv1x1 = nn.Conv2d(in_ch, out_ch,kernel_size = (1,1),padding="same", bias=True)

        self.maxpl = nn.MaxPool2d(kernel_size = 3, stride = 1,padding=1)

        self.bn2 = nn.BatchNorm2d(out_ch,eps=1e-3)

        self.mysigmoid = nn.Sigmoid()


    def forward(self,input):

        x1,x2 = split(input)

        output = self.conv3x1(x1)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv1x3(output)
        output = self.bn1(output)
        out13normre = F.relu(output)

        output = self.conv1x3(x2)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv3x1(output)
        output = self.bn1(output)
        out31normre = F.relu(output)

        out11cat = self.conv1x1(torch.cat((out13normre,out31normre), dim=1))

        outmp = self.maxpl(input)
        outmp11 = self.conv1x1(outmp)
        outmp11norm = self.bn2(outmp11)
        outmp11normsgm = self.mysigmoid(outmp11norm)

        Ewp = out11cat * outmp11normsgm
        Ews = out11cat + Ewp

        return channel_shuffle(Ews,2)

class AEM(nn.Module):

    def __init__(self):
        super().__init__()
        self.du1 = DownUnit(in_chs=3, out_chs=32)
        self.stage1_CSSAM_dt1 = CSSAM(in_ch=32, out_ch=32, dilated=1)
        self.du2 = DownUnit(in_chs=32, out_chs=64)
        self.stage2_CSSAM_dt1 = CSSAM(in_ch=64, out_ch=64, dilated=1)
        self.du3 = DownUnit(in_chs=64, out_chs=128)
        self.stage3_CSSAM_dt1 = CSSAM(in_ch=128 , out_ch=128, dilated=1)
        self.stage3_CSSAM_dt2 = CSSAM(in_ch=128 , out_ch=128, dilated=2)
        self.stage3_CSSAM_dt5 = CSSAM(in_ch=128 , out_ch=128, dilated=5)
        self.stage3_CSSAM_dt9 = CSSAM(in_ch=128 , out_ch=128, dilated=9)
        self.stage3_CSSAM_dt17 = CSSAM(in_ch=128 , out_ch=128, dilated=17)

    def forward(self, x):
        # stage1
        out = self.du1(x)
        out = self.stage1_CSSAM_dt1(out)
        out = self.stage1_CSSAM_dt1(out)
        out = self.stage1_CSSAM_dt1(out)
        # stage2
        out_2 = self.du2(out)
        out_2 = self.stage2_CSSAM_dt1(out_2)
        out_2 = self.stage2_CSSAM_dt1(out_2)
        # stage3
        out_3 = self.du3(out_2)
        out_3 = self.stage3_CSSAM_dt1(out_3)
        out_3 = self.stage3_CSSAM_dt2(out_3)
        out_3 = self.stage3_CSSAM_dt5(out_3)
        out_3 = self.stage3_CSSAM_dt9(out_3)
        out_3 = self.stage3_CSSAM_dt2(out_3)
        out_3 = self.stage3_CSSAM_dt5(out_3)
        out_3 = self.stage3_CSSAM_dt9(out_3)
        out_3 = self.stage3_CSSAM_dt17(out_3)

        return out_2 , out_3
    

class SEM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv11_64out32 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2,kernel_size = (1,1),padding="same", bias=True),     #TODO:有自行除於2
            nn.BatchNorm2d(in_ch//2,eps=1e-3)
        )
        self.avgpl = nn.AvgPool2d(kernel_size = 3, stride = 1,padding=1)
        self.maxpl = nn.MaxPool2d(kernel_size = 3, stride = 1,padding=1)

        self.conv11_64out64 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch,kernel_size = (1,1),padding="same", bias=True),
            nn.BatchNorm2d(in_ch,eps=1e-3)
        )

        self.gavgpl = nn.AdaptiveAvgPool2d(1)

        self.conv11_64out128 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size = (1,1),padding="same", bias=True),
            nn.BatchNorm2d(out_ch,eps=1e-3)
        )
        self.mysigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv11_64out32(x)
        f20 = F.relu(out)
        f21 = self.avgpl(f20)
        f22 = self.maxpl(f20)
        f23 = torch.cat((f21,f22),dim=1)
        out = self.conv11_64out64(f23)
        f27 = F.relu(out)

        f24 = self.gavgpl(x)
        f25 = self.conv11_64out64(f24)
        f26 = self.mysigmoid(f25)

        f28 = f26 * f27
        f29 = f28 + x


        return f29

class CAM(nn.Module):
    def __init__(self, in_ch , out_ch):
        super().__init__()
        ck = in_ch//k
        self.conv33_cin_cout = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size =(3,3),padding="same", bias=True),
            nn.BatchNorm2d(out_ch,eps=1e-3)
        )
        self.conv33_cin_ckout = nn.Sequential(
            nn.Conv2d(in_ch,ck,kernel_size =(3,3),padding="same", bias=True),
            nn.BatchNorm2d(ck,eps=1e-3)
        )

        self.conv11_cin_ckout = nn.Conv1d(in_ch,ck,kernel_size = 3,padding="same", bias=True)
        self.conv11_ckin_cout = nn.Conv1d(ck,in_ch,kernel_size = 3,padding="same", bias=True)
        self.mysoftmax = nn.Softmax(dim = 1)

    def forward(self, x):
        f4 = self.conv33_cin_cout(x)
        batchsize, num_channels, height, width = f4.data.size()

        # reshape (torch版的)
        f5 = f4.view(-1, num_channels, height*width) 
        f6 = self.conv11_cin_ckout(f5)
        f9 = self.mysoftmax(f6)

        f7 = self.conv33_cin_ckout(x)
        batchsize, num_channels, height, width = f7.data.size()
        f8 = f7.view(-1, num_channels, height*width)
        
        f10 = f9 * f8

        f11 = self.conv11_ckin_cout(f10)
        batchsize, num_channels, HW = f11.data.size()
        f11 = f11.view(batchsize, num_channels, 32, 32)
        f12 = self.conv33_cin_cout(f11)
        
        f13 = f11 + f12

        return f13
    
class FFM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size = (1,1),padding="same", bias=True),
            nn.BatchNorm2d(out_ch,eps=1e-3),
            nn.ReLU()
        )
        self.upsamp = nn.Upsample(size = (64,64),mode ='bilinear',align_corners = True)

        self.conv11_in192_out128 = nn.Sequential(
            nn.Conv2d(192, out_ch,kernel_size = (1,1),padding="same", bias=True),
            nn.BatchNorm2d(out_ch,eps=1e-3),
            nn.ReLU()
        )

    def forward(self, f29 , f13 ,f18):
        f30 = self.conv11(f13)
        f30 = self.upsamp(f30)
        f31 = torch.cat((f29,f30),dim = 1)
        f32 = self.conv11_in192_out128(f31)
        f33 = f32 * f18

        return f33

class GCP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gcp = DownUnit(in_ch, 256)     #16 x 16 x 256
        self.conv11_in256_out128 = nn.Sequential(
            nn.Conv2d(256, out_ch,kernel_size = (1,1),padding="same", bias=True),
            nn.BatchNorm2d(out_ch,eps=1e-3)
        )
        self.gavgpl = nn.AdaptiveAvgPool2d(1)

        self.conv11_in128_out128 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size = (1,1),padding="same", bias=True),
            nn.BatchNorm2d(out_ch,eps=1e-3)
        )
        self.mysoftmax = nn.Softmax(dim= 1)

    def forward(self, x):
        f14 = self.gcp(x)
        f15 = self.conv11_in256_out128(f14)
        f16 = self.gavgpl(f15)
        f16 = self.conv11_in128_out128(f16)
        f17 = F.relu(f16)
        f18 = self.mysoftmax(f17)

        return f18
    
class SegHead(nn.Module):
    def __init__(self,in_ch, out_ch):
        super().__init__()
        self.conv33 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size = (3,3),padding="same", bias=True),
            nn.BatchNorm2d(out_ch,eps=1e-3),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch,kernel_size = (1,1),padding="same", bias=True),
            nn.BatchNorm2d(out_ch,eps=1e-3)
        )
        self.upsamp = nn.Upsample(size = (256,256),mode ='bilinear',align_corners = True)
        #self.mysigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.conv33(x)
        out = self.upsamp(out)
        out = self.conv11(out)
        #out = self.mysigmoid(out)

        return out
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.aem = AEM()
        self.sem = SEM(64,128)
        self.cam = CAM(128,128)
        self.gcp = GCP(128,128)
        self.ffm = FFM(128,128)
        self.seghead = SegHead(128,1)


        

    def forward(self, img):
        f2 ,f3 = self.aem(img)
        f29 = self.sem(f2)
        f13 = self.cam(f3)
        f18 = self.gcp(f3)
        f33 = self.ffm(f29,f13,f18)

        f19 = self.seghead(f3)
        f34 = self.seghead(f33)

        return f19,f34


if __name__ == "__main__":


    model = Net()
    x = torch.randn(16,3,256,256)
    f19,f34 = model(x)
    print(f19.shape)
    print(f34.shape)
    summary(model, input_size=(16,3,256,256))