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


def split_list(lst, ratio=0.8):
    split_index = int(len(lst) * ratio)
    return lst[:split_index], lst[split_index:]


class DatasetSegmentation(Dataset):
    def __init__(
        self,
        dataset_dir,
        mode="train",
    ):
        self.dataset_dir = dataset_dir
        self.mode = mode

        image_data = os.listdir(
            f"{dataset_dir}/smoke100k-H/smoke_image/"
        )  # List the list of files in the folder (without path) ps. Using glob will not be able to read special characters, such as: ()

        image_data_first_one = image_data[
            0
        ]  # Get the first file in the folder 取資料夾中的第一個檔案
        self.image_extension = image_data_first_one.split(".")[1]  # take extension 取副檔名

        mask_data = os.listdir(
            f"{dataset_dir}/smoke100k-H/smoke_mask/"
        )  # 列出資料夾中檔案清單(不含路徑) ps.用glob會因無法讀取特殊字元，如：（）

        mask_data_first_one = mask_data[
            0
        ]  # Get the first file in the folder 取資料夾中的第一個檔案
        self.mask_extension = mask_data_first_one.split(".")[1]  # take extension 取副檔名

        x_1 = glob(f"{dataset_dir}/smoke100k-H/smoke_image/*.{self.image_extension}")
        x_2 = glob(f"{dataset_dir}/smoke100k-L/smoke_image/*.{self.image_extension}")
        x_3 = glob(f"{dataset_dir}/smoke100k-M/smoke_image/*.{self.image_extension}")

        # random.shuffle(x)

        self.train_x_1, self.val_x_1 = split_list(x_1)
        self.train_x_2, self.val_x_2 = split_list(x_2)
        self.train_x_3, self.val_x_3 = split_list(x_3)

        self.test_x_1 = self.train_x_1 + self.val_x_1
        self.test_x_2 = self.train_x_2 + self.val_x_2
        self.test_x_3 = self.train_x_3 + self.val_x_3

        if self.mode == "train":
            self.train_x = self.train_x_1 + self.train_x_2 + self.train_x_3
            print("Number of Training Images:", len(self.train_x))
            self.data_dict = self.train_x
        elif self.mode == "val":
            self.val_x = self.val_x_1 + self.val_x_2 + self.val_x_3
            print("Number of Validation Images:", len(self.val_x))
            self.data_dict = self.val_x

        elif self.mode == "test":
            self.test_x = self.test_x_1 + self.test_x_2 + self.test_x_3
            print("Number of test Images:", len(self.test_x))
            self.data_dict = self.test_x

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        filename = self.data_dict[index].split("/")[-1].split(".")[0]

        if "_L_" in filename:
            y = f"{self.dataset_dir}/smoke100k-L/smoke_free_image/{filename}.{self.mask_extension}"
        elif "_M_" in filename:
            y = f"{self.dataset_dir}/smoke100k-M/smoke_free_image/{filename}.{self.mask_extension}"
        elif "_H_" in filename:
            y = f"{self.dataset_dir}/smoke100k-H/smoke_free_image/{filename}.{self.mask_extension}"

        images_path = self.data_dict[index]
        # print("images_path:", images_path)

        masks_path = y
        # print("masks_path:", masks_path)
        c_img = imread(images_path)
        c_img = cv2_brightness_augment(c_img)

        bg_img = imread(masks_path)

        if IMG_SCALING is not None:
            c_img = cv2.resize(c_img, (256, 256), interpolation=cv2.INTER_AREA)  # 插值
            bg_img = cv2.resize(bg_img, (256, 256), interpolation=cv2.INTER_AREA)
            # bg_img = np.reshape(bg_img, (bg_img.shape[0], bg_img.shape[1], -1))
        # bg_img = bg_img > 0

        c_img = c_img.astype("float32")  # Normalized 歸一化
        c_img = c_img / 255.0

        bg_img = bg_img.astype("float32")
        bg_img = bg_img / 255.0

        out_rgb = torch.from_numpy(c_img).float()
        out_bg = torch.from_numpy(bg_img).float()

        return (
            out_rgb.permute(2, 0, 1).contiguous(),
            out_bg.permute(2, 0, 1).contiguous(),
        )


if __name__ == "__main__":
    training_data = DatasetSegmentation(
        "/home/yaocong/Experimental/Dataset/Smoke100k_dataset/Smoke100k_dataset_H_L_M/",
        mode="train",
    )
    # print("train:", training_data.train_x)
    validation_data = DatasetSegmentation(
        "/home/yaocong/Experimental/Dataset/Smoke100k_dataset/Smoke100k_dataset_H_L_M/",
        mode="val",
    )
    # print("val:", validation_data.val_x)

    training_data_loader = DataLoader(
        training_data,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    validation_data_loader = DataLoader(
        validation_data,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    # for iteration, (img_image, mask_image) in enumerate(training_data_loader):
    #     print(img_image.shape)
    #     print(mask_image.shape)
    #     print(iteration)
    # ds = DatasetSegmentation()
    # dsl = DataLoader(ds, batch_size=1, shuffle=True)
    # fn, o_rgb, o_mask = next(iter(dsl))
    # print(o_rgb.shape)
    # print(o_mask.shape)
