import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm
import wandb
import utils.loss
import utils.metrics
import utils
import segmentation_models_pytorch as smp

def train_epoch(args, model, training_data_loader, device, optimizer, epoch):
    model.train()
    cudnn.benchmark = True
    count = 0
    n_element = 0
    mean_loss = 0
    mean_miou = 0
    # mean_dice_coef = 0
    # mean_miou_s = 0

    # Training loop 訓練迴圈
    pbar = tqdm((training_data_loader), total=len(training_data_loader))
    # for iteration,(img_image, mask_image) in enumerate(training_data_loader):
    for RGB_image, mask_image in pbar:
        img_image = RGB_image.to(device)
        mask_image = mask_image.to(device).long()
        # onnx_img_image = img_image

        img_image = img_image.requires_grad_(True)

        output, aux = model(img_image)

        # torchvision.utils.save_image(
        #     img_image, "./training_data_captures/" + "img_image" + ".jpg"
        # )
        # torchvision.utils.save_image(
        #     output, "./training_data_captures/" + "output" + ".jpg"
        # )
        # torchvision.utils.save_image(
        #     mask_image, "./training_data_captures/" + "mask_image" + ".jpg"
        # )
        

        optimizer.zero_grad()  # Clear before loss.backward() to avoid gradient residue 在loss.backward()前先清除，避免梯度殘留
        
        loss = utils.loss.CustomLoss(output,aux, mask_image)
        tp, fp, fn, tn = smp.metrics.get_stats(output, mask_image, mode='binary', threshold=0.5)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        # iou = utils.metrics.IoU(output, mask_image)
        # iou_s = utils.metrics.Sigmoid_IoU(output, mask_image)
        # dice_coef = utils.metrics.dice_coef(output, mask_image, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 0.1
        )  # 梯度裁減(避免梯度爆炸或消失) 0.1為閥值
        optimizer.step()
            
        output = (output > 0.5).float()
        
        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element
        mean_miou += (iou.item() - mean_miou) / n_element
        # mean_miou_s += (iou_s.item() - mean_miou_s) / n_element
        # mean_dice_coef += (dice_coef.item() - mean_dice_coef) / n_element

        pbar.set_description(f"trian_epoch [{epoch}/{args['epochs']}]")
        pbar.set_postfix(
            train_loss=mean_loss,
            train_miou=mean_miou,
            # train_miou_s=mean_miou_s,
            # train_dice_coef=mean_dice_coef,
        )
        if args["wandb_name"] != "no":
            wandb.log(
                {
                    "train_loss": mean_loss,
                    "train_miou": mean_miou,
                    # "train_miou_s": mean_miou_s,
                    # "train_dice_coef": mean_dice_coef,
                }
            )

        # Graphical archive of the epoch test set
        # epoch 測試集中的圖示化存檔
        count += 1
        # if not epoch % 5:
        #     torchvision.utils.save_image(torch.cat((mask_image,output),0), "./training_data_captures/" +str(count)+".jpg")
    return RGB_image, mask_image, output


def valid_epoch(args ,model, validation_data_loader, device, epoch):
    n_element = 0
    mean_loss = 0
    mean_miou = 0
    # mean_dice_coef = 0
    # mean_miou_s = 0

    model.eval()
    pbar = tqdm((validation_data_loader), total=len(validation_data_loader))
    for RGB_image, mask_image in pbar:
        img_image = RGB_image.to(device)
        mask_image = mask_image.to(device).long()
        onnx_img_image = img_image

        with torch.no_grad():
            output, aux = model(img_image)

        loss = utils.loss.CustomLoss(output, mask_image)
        tp, fp, fn, tn = smp.metrics.get_stats(output, mask_image, mode='binary', threshold=0.5)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro')
        # iou = utils.metrics.IoU(output, mask_image)
        # iou_s = utils.metrics.Sigmoid_IoU(output, mask_image)
        # dice_coef = utils.metrics.dice_coef(output, mask_image, device)

        output = (output > 0.5).float()
        
        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element
        mean_miou += (iou.item() - mean_miou) / n_element  # 別人研究出的算平均的方法
        # mean_miou_s += (iou_s.item() - mean_miou_s) / n_element  # 別人研究出的算平均的方法
        # mean_dice_coef += (dice_coef.item() - mean_dice_coef) / n_element

        pbar.set_description(f"val_epoch [{epoch}/{args['epochs']}]")
        pbar.set_postfix(
            val_loss=mean_loss,
            val_miou=mean_miou,
            # val_miou_s=mean_miou_s,
            # val_dice_coef=mean_dice_coef,
        )

        if args["wandb_name"] != "no":
            wandb.log(
                {
                    "val_loss": mean_loss,
                    "val_miou": mean_miou,
                    # "val_miou_s": mean_miou_s,
                    # "val_dice_coef": mean_dice_coef,
                }
            )

    return (
        mean_loss,
        mean_miou,
        # mean_miou_s,
        # mean_dice_coef,
        RGB_image,
        mask_image,
        output,
        onnx_img_image,
    )