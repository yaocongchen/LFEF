# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchinfo import summary


class Net(nn.Module):
    def __init__(self):  # use encoder to pass pretrained encoder
        super().__init__()

        self.test = nn.Conv2d(3, 32, 3, 2, padding=1)
        self.maxpl = nn.MaxPool2d(2, stride=2)
        self.avgpl = nn.AvgPool2d(2, stride=2)

    def forward(self, input):
        out = self.test(input)
        max_pl = self.maxpl(out)
        avg_pl = self.avgpl(out)
        out = max_pl + avg_pl

        return out


if __name__ == "__main__":
    model = Net()
    x = torch.randn(16, 3, 256, 256)
    output = model(x)
    print(output.shape)
    summary(model, input_size=(16, 3, 256, 256))
