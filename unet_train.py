import torch
import torch.optim as optim
from pathlib import Path
from torch import nn
from torchvision import transforms, utils, datasets
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
import torch
import numpy as np
import time

from unet_dataset import UNetDataset
from unet import UNet

model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/unet-voc.pt"
saving_interval = 10
epoch_number = 2
shuffle_data_loader = False
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batch_size = 2


def binary_mask_transform(pil_img):
    pil_img = pil_img.convert("L").resize((128, 128))
    np_mask = np.array(pil_img)
    bin_mask = (np_mask > 127).astype(np.uint8)  
    return torch.from_numpy(bin_mask).long()     


data_folder = "/data/"

#test data
test_image_dir = data_folder + "/empi_pairs/test/images"
test_mask_dir = data_folder + "/empi_pairs/test/masks"
test_pred_mask_dir = data_folder + "/empi_pairs/test/pred_masks"
test_trans_image_dir = data_folder + "/trans_data/test/images"
test_trans_mask_dir = data_folder + "/trans_data/test/masks"
test_trans_pred_mask_dir = data_folder + "/trans_data/test/pred_masks"
#train data
train_image_dir = data_folder + "/empi_pairs/train/images"
train_mask_dir = data_folder + "/empi_pairs/train/masks"
train_pred_mask_dir = data_folder + "/empi_pairs/train/pred_masks"
train_trans_image_dir = data_folder + "/trans_data/train/images"
train_trans_mask_dir = data_folder + "/trans_data/train/masks"
train_trans_pred_mask_dir = data_folder + "/trans_data/train/pred_masks"
#validation data
val_image_dir = data_folder + "/empi_pairs/val/images"
val_mask_dir = data_folder + "/empi_pairs/val/masks"
val_pred_mask_dir = data_folder + "/empi_pairs/val/pred_masks"
val_trans_image_dir = data_folder + "/trans_data/val/images"
val_trans_mask_dir = data_folder + "/trans_data/val/masks"
val_trans_pred_mask_dir = data_folder + "/trans_data/val/pred_masks"

model_path = "/results/model"
os.makedirs(model_path, exist_ok=True)
results_path="/results/unet_data"
os.makedirs(results_path, exist_ok=True)

#create paths if they do not exist
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_mask_dir, exist_ok=True)
os.makedirs(test_pred_mask_dir, exist_ok=True)
os.makedirs(test_trans_image_dir, exist_ok=True)
os.makedirs(test_trans_mask_dir, exist_ok=True)
os.makedirs(test_trans_pred_mask_dir, exist_ok=True)

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(train_pred_mask_dir, exist_ok=True)
os.makedirs(train_trans_image_dir, exist_ok=True)
os.makedirs(train_trans_mask_dir, exist_ok=True)
os.makedirs(train_trans_pred_mask_dir, exist_ok=True)

os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)
os.makedirs(val_pred_mask_dir, exist_ok=True)
os.makedirs(val_trans_image_dir, exist_ok=True)
os.makedirs(val_trans_mask_dir, exist_ok=True)
os.makedirs(val_trans_pred_mask_dir, exist_ok=True)


#for training the unet/segmentation network
#test data
test_image_dir = test_image_dir
test_pred_mask_dir = test_pred_mask_dir
test_mask_dir = test_mask_dir
#train data
train_image_dir = train_image_dir
train_mask_dir = train_mask_dir
train_pred_mask_dir = train_pred_mask_dir
#validation data
val_image_dir = val_image_dir
val_mask_dir = val_mask_dir
val_pred_mask_dir = val_pred_mask_dir


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

batch_size = 2
lr = 5e-5
epoch_number = 10

def train(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=batch_size, lr=lr, epoch_number=epoch_number, index=0):
    train_load_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_data_loader)
    val_load_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_data_loader)
    model_path = f"/results/model/{index}_seg_model.pt"

    model = UNet(dimensions=2)
    model.to(device)

    if os.path.isfile(model_path):
        print(f"Not loading model from {model_path} due to output dimension mismatch.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # initialize the best loss to a high value
    # this is used to save the model if the validation loss improves
    best_loss = float('inf')
    epochs_without_improvement = 0
    patience = 5
    epoch_train_loss = []
    epoch_val_loss = []

    for epoch in range(epoch_number):
        print(f"Epoch {epoch+1}")
        time1 = time.time()
        model.train()
        train_losses = []
        for input, target in train_load_data:
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        # print the average loss for that epoch.
        avg_train_loss = sum(train_losses) / len(train_losses)
        epoch_train_loss.append(avg_train_loss)
        print(f"Training Loss: {avg_train_loss:.4f}")

        #validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for input, target in val_load_data:
                input = input.to(device)
                target = target.long().to(device)
                if target.ndim == 4 and target.shape[1] == 1:
                    target = target.squeeze(1)
                if input.shape[0] < 2:
                    continue
                output = model(input)
                loss = criterion(output, target)
                val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        epoch_val_loss.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # check if the validation loss improved
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "model/best_model.pt")
            print(f"Best model saved at epoch {epoch+1} with val loss {best_loss:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss for {epochs_without_improvement} epochs")
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break
        # save model
        if (epoch + 1) % saving_interval == 0:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch + 1}")
        time2 = time.time()
        print(f"Epoch {epoch+1}, time {time2 - time1}")

    torch.save(model.state_dict(), model_path)

    # Save training and validation loss
    torch.save({
        'train_loss': epoch_train_loss,
        'val_loss': epoch_val_loss
    }, f"/results/unet_data/unet_{index}_losses.pt")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    epochs = list(range(1, len(epoch_train_loss) + 1))
    plt.plot(epochs, epoch_train_loss, label='Training Loss')
    if epoch_val_loss:
        plt.plot(epochs, epoch_val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('UNet Training and Validation Loss')
    plt.legend()
    plt.savefig(f"/results/unet_data/unet_{index}_energy_loss.png")
    plt.close()
    print(f"Training complete and model saved as unet_{index}_energy_model.pth")

    return epoch_train_loss, epoch_val_loss

def plot_unet_losses(index, folder="/results/unet_data", show=True):
    """
    Load and plot UNet training and validation losses.

    Args:
        index (int): Index used when saving the loss file.
        folder (str): Path to the folder where loss files are stored.
        show (bool): Whether to display the plot (default: True).
    """
    # Load saved losses
    losses = torch.load(f"{folder}/unet_{index}_losses.pt")
    train_loss = losses['train_loss']
    val_loss = losses['val_loss']
    epochs = list(range(1, len(train_loss) + 1))

    # Plot with grid and larger text
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Training Loss', linewidth=2)
    if val_loss:
        plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)

    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('UNet Training and Validation Loss', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = f"{folder}/unet_{index}_energy_loss_grid.png"
    plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    train()
    #plot_unet_losses("021", folder="/results/unet_data", show=False)
