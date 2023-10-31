# %%
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from glob import glob
import cv2
import numpy as np
import torch
import random
import os
import time

IMG_SCALING = (1, 1)


# Picture brightness enhancement 圖片亮度增強
def cv2_brightness_augment(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    seed = random.uniform(0.5, 1.2)
    v = ((v / 255.0) * seed) * 255.0
    hsv[:, :, 2] = np.array(np.clip(v, 0, 255), dtype=np.uint8)
    rgb_final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_final


class DatasetSegmentation(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        mode="train",
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.mode = mode

        image_data = os.listdir(
            images_dir
        )  # List the list of files in the folder (without path) ps. Using glob will not be able to read special characters, such as: ()

        image_data_first_one = image_data[
            0
        ]  # Get the first file in the folder 取資料夾中的第一個檔案
        self.image_extension = image_data_first_one.split(".")[1]  # take extension 取副檔名

        mask_data = os.listdir(masks_dir)  # 列出資料夾中檔案清單(不含路徑) ps.用glob會因無法讀取特殊字元，如：（）

        mask_data_first_one = mask_data[
            0
        ]  # Get the first file in the folder 取資料夾中的第一個檔案
        self.mask_extension = mask_data_first_one.split(".")[1]  # take extension 取副檔名

        x = glob(f"{images_dir}/*.{self.image_extension}")

        # random.shuffle(x)

        def split_list(lst, ratio=0.8):
            split_index = int(len(lst) * ratio)
            return lst[:split_index], lst[split_index:]

        self.train_x, self.val_x = split_list(x)
        self.test_x = self.train_x + self.val_x

        if self.mode == "train":
            print("Number of Training Images:", len(self.train_x))
        elif self.mode == "val":
            print("Number of Validation Images:", len(self.val_x))
        elif self.mode == "test":
            print("Number of test Images:", len(self.test_x))

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, index):
        if self.mode == "train":
            filename = self.train_x[index].split("/")[-1].split(".")[0]
            y = f"{self.masks_dir}/{filename}.{self.mask_extension}"
            images_path = self.train_x[index]

        elif self.mode == "val":
            filename = self.val_x[index].split("/")[-1].split(".")[0]
            y = f"{self.masks_dir}/{filename}.{self.mask_extension}"
            images_path = self.val_x[index]

        elif self.mode == "test":
            filename = self.test_x[index].split("/")[-1].split(".")[0]
            y = f"{self.masks_dir}/{filename}.{self.mask_extension}"
            images_path = self.test_x[index]

        masks_path = y

        c_img = imread(images_path)
        c_img = cv2_brightness_augment(c_img)

        c_mask = imread(masks_path)

        if IMG_SCALING is not None:
            c_img = cv2.resize(c_img, (256, 256), interpolation=cv2.INTER_AREA)  # 插值
            c_mask = cv2.resize(c_mask, (256, 256), interpolation=cv2.INTER_AREA)
            c_mask = np.reshape(c_mask, (c_mask.shape[0], c_mask.shape[1], -1))
        c_mask = c_mask > 0
        c_mask = c_mask.astype("float32")

        c_img = c_img.astype("float32")  # Normalized 歸一化
        c_img = c_img / 255.0

        out_rgb = torch.from_numpy(c_img).float()
        out_mask = torch.from_numpy(c_mask).float()

        return (
            out_rgb.permute(2, 0, 1).contiguous(),
            out_mask.permute(2, 0, 1).contiguous(),
        )


if __name__ == "__main__":
    training_data = DatasetSegmentation(
        "/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/img",
        "/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/gt",
    )
    print("train:", training_data.train_x)

    validation_data = DatasetSegmentation(
        "/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/img",
        "/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/gt",
        mode="val",
    )
    print("val:", validation_data.val_x)

    test_data = DatasetSegmentation(
        "/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/img",
        "/home/yaocong/Experimental/speed_smoke_segmentation/test_files/ttt/gt",
        mode="test",
    )
    print("test:", test_data.test_x)

    # testing_data_loader = DataLoader(
    #     training_data,
    #     batch_size=8,
    #     shuffle=True,
    #     num_workers=1,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # ds = DatasetSegmentation()
    # dsl = DataLoader(ds, batch_size=1, shuffle=True)
    # fn, o_rgb, o_mask = next(iter(dsl))
    # print(o_rgb.shape)
    # print(o_mask.shape)
