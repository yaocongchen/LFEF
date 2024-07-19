import torch 
import torchvision 
import models.LFEF as network_model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = network_model.Net()

model.load_state_dict(torch.load("./trained_models/best.pth"))

# print(model)
example = torch.rand(1, 3, 256, 256)

traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("./trained_models/torch_script/model.pt")

loaded_model = torch.jit.load("./trained_models/torch_script/model.pt")

input_tensor = torch.randn(1, 3, 256, 256)
output = loaded_model(input_tensor)
print(output)
