# Code to evaluate forward pass time in Pytorch
# Sept 2017
# Eduardo Romera
#######################

import os
import numpy as np
import torch
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable

import sys

sys.path.append("..")
from models import erfnet

import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def main(args):
    model = erfnet.Net(19)
    if not args.cpu:
        model = model.cuda()  # .half()	#HALF seems to be doing slower for some reason
    # model = torch.nn.DataParallel(model).cuda()

    model.eval()

    images = torch.randn(args.batch_size, args.num_channels, args.height, args.width)

    if not args.cpu:
        images = images.cuda()  # .half()

    time_train = []

    i = 0

    while True:
        # for step, (images, labels, filename, filenameGt) in enumerate(loader):

        start_time = time.time()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        # preds = outputs.cpu()
        if not args.cpu:
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)

        if i != 0:  # first run always takes some time for setup
            fwt = time.time() - start_time
            time_train.append(fwt)
            per_img = fwt / args.batch_size
            mean_time = sum(time_train) / len(time_train) / args.batch_size
            print(
                "Forward time per img (b=%d): %.3f (Mean: %.3f)"
                % (args.batch_size, per_img, mean_time)
            )
            print(
                "FPS(b=%d): %.1f (Mean:%.1f)"
                % (args.batch_size, 1 / per_img, 1 / mean_time)
            )

        time.sleep(1)  # to avoid overheating the GPU too much
        i += 1


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")

    main(parser.parse_args())
