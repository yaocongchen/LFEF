from typing import List, Tuple
from torch import Tensor
import torch
import torchvision
import torch.optim
from torchvision.io import read_image
from zmq import device
import time
import onnxruntime as ort
import numpy as np


def smoke_semantic(input_image: np.ndarray , ort_session:ort.InferenceSession, time_train: List[float], i: int) -> Tuple[np.ndarray, np.ndarray]:
    start_time = time.time()

    ort_inputs = {ort_session.get_inputs()[0].name: input_image}
    ort_outputs = ort_session.run(None, ort_inputs)
    output = ort_outputs[0]

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


