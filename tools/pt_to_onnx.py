# pt_to_onnx
import torch
import torch.onnx
import models.LFEF as network_model

def set_to_eval(model):
    for module in model.modules():
        if isinstance(module, torch.nn.InstanceNorm2d):
            module.eval()

# Load the pretrained model
model = network_model.Net()
model.load_state_dict(torch.load("./trained_models/best.pth"))
set_to_eval(model)  # 確保所有InstanceNorm層都處於評估模式
model.eval()


example_input = torch.randn(1, 3, 256, 256, requires_grad=True)

# Export the model
torch.onnx.export(model,               # model being run
                  example_input,                         # model input (or a tuple for multiple inputs)
                  "./trained_models/best.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

