import os
import torch

def check_have_GPU(args):
    # Check have GPU device 確認是否有GPU裝置
    if args["device"] == "GPU":
        print("====> Use gpu id:'{}'".format(args["gpus"]))
        os.environ["CUDA_VISIBLE_DEVICES"] = args["gpus"]
        if not torch.cuda.is_available():
            raise Exception(
                "No GPU found or Wrong gpu id, please run without --device"
            )  # 例外事件跳出


def check_number_of_GPUs(args, model):
    if args["device"] == "GPU":
        # args.gpu_nums = 1
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count()=", torch.cuda.device_count())
            # args.gpu_nums = torch.cuda.device_count()
            model = torch.nn.DataParallel(model).cuda()  # multi-card data parallel
            device = torch.device("cuda")
        else:
            print("Single GPU for training")
            model = model.cuda()  # 1-card data parallel
            device = torch.device("cuda")
    else:
        print("CPU for training")
        model = model.cpu()  # use cpu data parallel
        device = torch.device("cpu")

    return model, device

# setting random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random seed is set to:", seed)