import torch
from torchinfo import summary
import sys
sys.path.append(".")
from utils.metrics import Calculate
from models.LFEF import Net

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    model = Net()
    x = torch.randn(16, 3, 256, 256)
    output,aux = model(x)
    # print(output.shape)
    summary(model,input_data=x,verbose=1)
    c = Calculate(model)
    model_size = c.get_model_size()
    flops, params = c.get_params()
