import torch
import torchvision
import torch.optim
from torchvision.io import read_image
from zmq import device
from torchvision import transforms
import time


def smoke_semantic(input_image, model, device, time_train, i):
    start_time = time.time()

    with torch.no_grad():
        output = model(input_image)  # Import model 導進模型
        output = (output > 0.5).float()

    if device == torch.device("cuda"):
        torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)

    if i != 0:
        fwt = time.time() - start_time
        time_train.append(fwt)
        mean_time = sum(time_train) / len(time_train)
        print("Forward time per img: %.3f (Mean: %.3f)" % (fwt, mean_time))
        print("Forward_Time_FPS: %.1f (Mean:%.1f)" % (1 / fwt, 1 / mean_time))

    # time_end = time.time()
    # spend_time = int(time_end-start_time)
    # time_min = spend_time // 60
    # time_sec = spend_time % 60
    # print('totally cost:',f"{time_min}m {time_sec}s")

    # #Calculate FPS
    # print("Model_FPS: {:.1f}".format(1/(time_end-start_time)))

    return output


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Inference on device {device}.")

    smoke_input_image = read_image(
        "/home/yaocong/Experimental/speed_smoke_segmentation/test_files/123.jpg"
    )
    model_path = "/home/yaocong/Experimental/speed_smoke_segmentation/checkpoint/bs8e150/final.pth"
    transform = transforms.Resize([256, 256])
    smoke_input_image = transform(smoke_input_image)
    smoke_input_image = (smoke_input_image) / 255.0
    smoke_input_image = smoke_input_image.unsqueeze(0).to(device)

    output = smoke_semantic(smoke_input_image, model_path, device)
    print("output:", output.shape)
    torchvision.utils.save_image(output, "inference" + ".jpg")
