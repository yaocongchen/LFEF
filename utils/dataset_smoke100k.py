# %%
from sklearn.model_selection import train_test_split
import os
import torch
import torch.utils.data as data
import numpy as np
import random
import cv2
import torch.utils.data as data
from skimage.io import imread

# Initial parameters 初始參數
IMG_SCALING = (1, 1)


# Data preprocessing 資料預處理
def preparing_training_data(images_dir, masks_dir):
    train_data = []
    validation_data = []
    test_data = []

    image_data = os.listdir(
        images_dir
    )  # List the list of files in the folder (without path) ps. Using glob will not be able to read special characters, such as: ()
    mask_data = os.listdir(masks_dir)  # 列出資料夾中檔案清單(不含路徑) ps.用glob會因無法讀取特殊字元，如：（）

    random.shuffle(mask_data)

    image_data_first_one = image_data[0]  # Get the first file in the folder 取資料夾中的第一個檔案
    extension = image_data_first_one.split(".")[1]  # take extension 取副檔名

    data_holder = {}

    for m_image in mask_data:
        # m_id_ = m_image.split("/")[-1]
        id_ = m_image.split(".")[0]
        i_id_ = id_ + "." + extension
        if not i_id_ in data_holder.keys():
            # 	data_holder[id_].append(m_image)
            # else:
            # m_image = m_image.split("/")[-1]
            data_holder[i_id_] = []
            data_holder[i_id_].append(m_image)

    train_ids = []
    val_ids = []

    num_of_ids = len(data_holder.keys())
    for i in range(num_of_ids):
        if i < num_of_ids * 4 / 5:
            train_ids.append(list(data_holder.keys())[i])
        else:
            val_ids.append(list(data_holder.keys())[i])

    for id_ in list(data_holder.keys()):
        if id_ in train_ids:
            for hazy_image in data_holder[id_]:
                train_data.append([images_dir + id_, masks_dir + hazy_image])
        else:
            for hazy_image in data_holder[id_]:
                validation_data.append([images_dir + id_, masks_dir + hazy_image])

        # for test.py use
        test_data = train_data + validation_data

    # print("train_data",train_data)
    # print("====================================================")
    # print("validation_data", validation_data)

    # TODO:考慮必要性
    random.shuffle(train_data)
    random.shuffle(validation_data)
    random.shuffle(test_data)
    # print("====================================================")
    # print("validation_data",validation_data)

    return train_data, validation_data, test_data


# %%


# Picture brightness enhancement 圖片亮度增強
def cv2_brightness_augment(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    seed = random.uniform(0.5, 1.2)
    v = ((v / 255.0) * seed) * 255.0
    hsv[:, :, 2] = np.array(np.clip(v, 0, 255), dtype=np.uint8)
    rgb_final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_final


# %%
class DataLoaderSegmentation(data.Dataset):
    def __init__(self, images_dir, mode="train"):
        self.train_data = []
        self.validation_data = []
        self.test_data = []
        (
            self.train_data_H,
            self.validation_data_H,
            self.test_data_H,
        ) = preparing_training_data(
            images_dir + "smoke100k-H/smoke_image/",
            images_dir + "smoke100k-H/smoke_mask/",
        )
        (
            self.train_data_L,
            self.validation_data_L,
            self.test_data_L,
        ) = preparing_training_data(
            images_dir + "smoke100k-L/smoke_image/",
            images_dir + "smoke100k-L/smoke_mask/",
        )
        (
            self.train_data_M,
            self.validation_data_M,
            self.test_data_M,
        ) = preparing_training_data(
            images_dir + "smoke100k-M/smoke_image/",
            images_dir + "smoke100k-M/smoke_mask/",
        )
        self.train_data = self.train_data_H + self.train_data_L + self.train_data_M
        self.validation_data = (
            self.validation_data_H + self.validation_data_L + self.validation_data_M
        )
        self.test_data = self.test_data_H + self.test_data_L + self.test_data_M

        if mode == "train":
            self.data_dict = self.train_data
            # print(self.data_dict)
            print("Number of Training Images:", len(self.train_data))
        elif mode == "val":
            self.data_dict = self.validation_data
            print("Number of Validation Images:", len(self.validation_data))
        elif mode == "test":
            self.data_dict = self.test_data
            print("Number of test Images:", len(self.test_data))

    def __len__(self):
        return len(self.data_dict)

    # Import data by index 依index匯入資料
    def __getitem__(self, index):
        images_path, masks_path = self.data_dict[index]
        c_img = imread(images_path)
        c_img = cv2_brightness_augment(c_img)

        c_mask = imread(masks_path)

        if IMG_SCALING is not None:
            c_img = cv2.resize(c_img, (256, 256), interpolation=cv2.INTER_AREA)  # 插值
            c_mask = cv2.resize(c_mask, (256, 256), interpolation=cv2.INTER_AREA)
            c_mask = np.reshape(c_mask, (c_mask.shape[0], c_mask.shape[1], -1))
        c_mask = c_mask > 0

        c_img = c_img.astype("float32")  # Normalized 歸一化
        c_img = c_img / 255.0

        out_rgb = torch.from_numpy(c_img).float()
        out_mask = torch.from_numpy(c_mask).float()

        return (
            out_rgb.permute(2, 0, 1).contiguous(),
            out_mask.permute(2, 0, 1).contiguous(),
        )


if __name__ == "__main__":
    import time

    seconds = time.time()
    print("s", seconds)

    random.seed(seconds)

    print("s", seconds)
    testing_data = DataLoaderSegmentation(
        "/home/yaocong/Experimental/Dataset/smoke100k_dataset/",
        mode="train",
    )
    random.seed(seconds)
    print("s", seconds)
    testing_data = DataLoaderSegmentation(
        "/home/yaocong/Experimental/Dataset/smoke100k_dataset/",
        mode="val",
    )

    testing_data_loader = torch.utils.data.DataLoader(
        testing_data,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
