import configparser
from torch.utils.data import DataLoader
import utils

CONFIG_FILE = "import_dataset_path.cfg"

def data_processing_train_8_val_DS01(args):
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
        
    train_images = args.get(args["train_images"], config.get(args["train_dataset_path"], "train_images"))
    train_masks = args.get(args["train_masks"], config.get(args["train_dataset_path"], "train_masks"))

    validation_images = args.get(args["validation_images"], config.get(args["validation_dataset_path"], "validation_images"))
    validation_masks = args.get(args["validation_masks"], config.get(args["validation_dataset_path"], "validation_masks"))

    training_data = utils.dataset.DatasetSegmentation(train_images, train_masks)
    validation_data = utils.dataset.DatasetSegmentation(
        validation_images, validation_masks, mode="all"
    )

    training_data_loader = DataLoader(
        training_data,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    validation_data_loader = DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    return train_images, train_masks, training_data_loader, validation_data_loader

def data_processing_8_2(args):
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    train_images = args.get(args["train_images"], config.get(args["train_dataset_path"], "train_images"))
    train_masks = args.get(args["train_masks"], config.get(args["train_dataset_path"], "train_masks"))

    training_data = utils.dataset.DatasetSegmentation(train_images, train_masks)
    validation_data = utils.dataset.DatasetSegmentation(
        train_images, train_masks, mode="val"
    )
    training_data_loader = DataLoader(
        training_data,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    validation_data_loader = DataLoader(
        validation_data,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    return train_images, train_masks, training_data_loader, validation_data_loader