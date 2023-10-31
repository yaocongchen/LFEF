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

# Random number generation 亂數產生

random.seed(1143)

# Initial parameters 初始參數
IMG_SCALING = (1, 1)


# Data preprocessing 資料預處理
def preparing_training_data(all_file_name):
    train_data = []
    validation_data = []
    test_data = []

    train_ids = []
    val_ids = []
    num_of_ids = len(all_file_name)
    for i in range(num_of_ids):
        if i < num_of_ids * 9 / 10:
            train_ids.append(list(all_file_name)[i])
        else:
            val_ids.append(list(all_file_name)[i])

    # for test.py use
    train_data = train_ids
    validation_data = val_ids
    test_data = train_data + validation_data

    random.shuffle(train_data)
    random.shuffle(validation_data)
    random.shuffle(test_data)

    return train_data, validation_data, test_data


class DatasetSegmentation(data.Dataset):
    def __init__(self, images_dir, masks_dir, mode="train"):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        dir_path = os.path.dirname(images_dir)
        self.all_file_name = os.listdir(dir_path)

        self.train_data, self.validation_data, self.test_data = preparing_training_data(
            self.all_file_name
        )

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
        image_name = self.data_dict[index]
        out_rgb = np.load(self.images_dir + image_name)
        out_mask = np.load(self.masks_dir + image_name)
        return out_rgb, out_mask


if __name__ == "__main__":
    from tqdm import tqdm

    testing_data = DatasetSegmentation(
        "/home/yaocong/Experimental/Dataset/smoke100k_dataset/smoke_image/",
        "/home/yaocong/Experimental/Dataset/smoke100k_dataset/smoke_mask/",
        mode="train",
    )

    # dir_path = os.path.dirname("/home/yaocong/Experimental/Dataset/smoke100k_dataset/smoke_image_npy/")
    # all_file_name = os.listdir(dir_path)
    # print(all_file_name)
    # print(len(all_file_name))

    testing_data_loader = torch.utils.data.DataLoader(
        testing_data,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    print(testing_data_loader)
