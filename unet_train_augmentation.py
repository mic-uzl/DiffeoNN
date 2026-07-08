import torch
import torch.optim as optim
from pathlib import Path
from torch import nn
from torchvision import transforms, utils, datasets
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import functional as TF
import torch
from torch.utils.data import ConcatDataset

from unet_dataset import UNetDataset
from unet import UNet
from random_transforms_rbf import rand_mask_trans2d
from unet_train import train


saving_interval = 10
shuffle_data_loader = False
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 2
lr = 0.0005
epochs = 20

def binary_mask_transform(pil_img):
    pil_img = pil_img.convert("L").resize((128, 128))
    np_mask = np.array(pil_img)
    bin_mask = (np_mask > 127).astype(np.uint8)  
    return torch.from_numpy(bin_mask).long()     

remote = True  # set to True if running on remote server, False if local
noisy = True
#setting up paths
data_folder = "/data"


#train data
train_image_dir = data_folder + "/empi_pairs/train/images"
train_mask_dir = data_folder + "/empi_pairs/train/masks"
train_pred_mask_dir = data_folder + "/empi_pairs/train/pred_masks"
train_trans_image_dir = data_folder + "/trans_data/train/images"
train_trans_mask_dir = data_folder + "/trans_data/train/masks"
train_trans_pred_mask_dir = data_folder + "/trans_data/train/pred_masks"
train_trans_compare_mask_dir = data_folder + "trans_data/train/compare_masks"
train_trans_pred_orig_mask_dir = data_folder + "trans_data/train/pred_orig_masks"
train_trans_pred_orig_image_dir = data_folder + "trans_data/train/pred_orig_images"

#validation data
val_image_dir = data_folder + "/empi_pairs/val/images"
val_mask_dir = data_folder + "/empi_pairs/val/masks"
val_pred_mask_dir = data_folder + "/empi_pairs/val/pred_masks"
val_trans_image_dir = data_folder + "/trans_data/val/images"
val_trans_mask_dir = data_folder + "/trans_data/val/masks"
val_trans_pred_mask_dir = data_folder + "/trans_data/val/pred_masks"
val_trans_compare_mask_dir = data_folder + "trans_data/val/compare_masks"
val_trans_pred_orig_mask_dir = data_folder + "trans_data/val/pred_orig_masks"
val_trans_pred_orig_image_dir = data_folder + "trans_data/val/pred_orig_images"


os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(train_pred_mask_dir, exist_ok=True)
os.makedirs(train_trans_image_dir, exist_ok=True)
os.makedirs(train_trans_mask_dir, exist_ok=True)
os.makedirs(train_trans_pred_mask_dir, exist_ok=True)
os.makedirs(train_trans_compare_mask_dir, exist_ok=True)
os.makedirs(train_trans_pred_orig_mask_dir, exist_ok=True)
os.makedirs(train_trans_pred_orig_image_dir, exist_ok=True)

os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)
os.makedirs(val_pred_mask_dir, exist_ok=True)
os.makedirs(val_trans_image_dir, exist_ok=True)
os.makedirs(val_trans_mask_dir, exist_ok=True)
os.makedirs(val_trans_pred_mask_dir, exist_ok=True)
os.makedirs(val_trans_compare_mask_dir, exist_ok=True)
os.makedirs(val_trans_pred_orig_mask_dir, exist_ok=True)
os.makedirs(val_trans_pred_orig_image_dir, exist_ok=True)

#training data
train_dataset = UNetDataset(
    image_dir=train_image_dir,
    mask_dir=train_mask_dir,
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()]),
    mask_transform = binary_mask_transform
    )

#validation data
val_dataset = UNetDataset(
    image_dir=val_image_dir,
    mask_dir=val_mask_dir,
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()]),
    mask_transform = binary_mask_transform
    )

#train dataset transformed
train_trans_dataset = UNetDataset(
    image_dir=train_trans_image_dir,
    mask_dir=train_trans_mask_dir,
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()]),
    mask_transform = binary_mask_transform
    )

#validation dataset transformed
val_trans_dataset = UNetDataset(
    image_dir=val_trans_image_dir,
    mask_dir=val_trans_mask_dir,
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()]),
    mask_transform = binary_mask_transform
    )

# Combined training and validation datasets
combined_train_dataset = ConcatDataset([train_dataset, train_trans_dataset])
combined_val_dataset = ConcatDataset([val_dataset, val_trans_dataset])


def train_aug():
    epoch_train_loss, epoch_val_loss = train(train_dataset=combined_train_dataset, val_dataset=combined_val_dataset, batch_size=batch_size, lr=lr, epoch_number=epochs, index=0)
    return epoch_train_loss, epoch_val_loss

if __name__ == "__main__":
    # train normal unet
    train(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=batch_size, lr=lr, epoch_number=epochs, index=1)
    # train augmented unet
    train_aug()
