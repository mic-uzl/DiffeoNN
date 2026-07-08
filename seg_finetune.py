from pathlib import Path
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from torchvision import transforms, utils, datasets
import os
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import torch
from matplotlib.pyplot import cm

from unet_dataset import UNetDataset
from unet import UNet, UNet_small
from unet_train import train
import gc

model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "/results/model/reg_energy.pt"
shuffle_data_loader = False
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

remote = True  # set to True if running on remote server, False if local

model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "/results/model/unet-voc.pt"
saving_interval = 10
epoch_number = 20 
shuffle_data_loader = False
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batch_size = 1

remote = True  # set to True if running on remote server, False if local
noisy = True

data_folder = "/data"
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

def binary_mask_transform(pil_img):
    pil_img = pil_img.convert("L").resize((128, 128))
    np_mask = np.array(pil_img)
    bin_mask = (np_mask > 127).astype(np.uint8)  
    return torch.from_numpy(bin_mask).long()     

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

# test data
test_dataset = UNetDataset(
    image_dir=test_image_dir,
    mask_dir=test_mask_dir,
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()]),
    mask_transform = binary_mask_transform
    )

# predict segmentation masks with UNet
def predict(model_path=model_path, test_dataset=test_dataset, test_pred_mask_dir=test_pred_mask_dir):
    model = UNet(dimensions=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    os.makedirs(test_pred_mask_dir, exist_ok=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=shuffle_data_loader)
        
    for i in range(len(test_dataset)):
        img, _ = test_dataset[i]  
        img_tensor = img.unsqueeze(0).to(device)  

        with torch.no_grad():
            output = model(img_tensor)  
            pred_mask = output.argmax(dim=1).squeeze(0)  

        # Save predicted mask
        save_path = os.path.join(test_pred_mask_dir, f"{i+1}.png")
        to_pil_image(pred_mask.byte() * 255).save(save_path)

    print(f"\nPrediction completed. Best model used: No")
    return

# metrics for unet segmentation
def metrics_unet(test_dataset = test_dataset, test_seg_output_dir=None, results_dir=None):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    iou = []
    dice = []
    accuracy = []
    total_pixels = 128 * 128
    num_masks = 0

    for i in range(len(test_dataset)):
        img, mask = test_dataset[i]  
        mask = mask.squeeze()  
        if mask.max() > 1:
            mask = (mask > 127).to(torch.uint8)  
        else:
            mask = mask.to(torch.uint8)

        mask_array = mask.numpy()  

        # Load predicted mask and binarize
        pred_mask_path = os.path.join(test_seg_output_dir, f"{i+1}.png")
        pred_mask = Image.open(pred_mask_path).convert("L").resize((128, 128))
        pred_mask_array = (np.array(pred_mask) > 127).astype(np.uint8)

        # Optionally save some masks for visual inspection
        if i % 1 == 0 and results_dir:
            Image.fromarray(mask_array * 255).save(os.path.join(results_dir, f"{int((i / 1)+1)}_mask.png"))
            Image.fromarray(pred_mask_array * 255).save(os.path.join(results_dir, f"{int((i / 1)+1)}_pred_mask.png"))

        # Metrics
        intersection = np.logical_and(mask_array, pred_mask_array).sum()
        union = np.logical_or(mask_array, pred_mask_array).sum()
        mask_sum = mask_array.sum() + pred_mask_array.sum()

        iou_item = intersection / union if union != 0 else 0.0
        dice_item = 2 * intersection / mask_sum if mask_sum != 0 else 0.0
        acc_item = (pred_mask_array == mask_array).sum() / total_pixels

        iou.append(iou_item)
        dice.append(dice_item)
        accuracy.append(acc_item)
        num_masks += 1

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(iou, label='IOU')
    plt.plot(dice, label='Dice')
    plt.plot(accuracy, label='Accuracy')
    plt.xlabel('Testdataset')
    plt.ylabel('Losses')
    plt.title('UNet Testing Results')
    plt.legend()
    plt.savefig(results_dir + f"/unet_test_losses.png")
    plt.close()
    print(f"Training complete and model saved as unet_test_losses.png")
    return {
        "iou": iou ,
        "dice": dice ,
        "accuracy": accuracy,
        "num_masks": num_masks
    }

# get results for unet
def results_unet(unet_path=None, results_dir=None, test_dataset=test_dataset, test_pred_mask_dir=test_pred_mask_dir):
    predict(model_path=unet_path, test_dataset=test_dataset, test_pred_mask_dir=test_pred_mask_dir)
    metrics = metrics_unet(test_dataset = test_dataset, test_seg_output_dir=test_pred_mask_dir, results_dir=results_dir)
    mean_iou = sum(metrics['iou'])/metrics['num_masks']
    mean_dice = sum(metrics['dice'])/metrics['num_masks']
    mean_accuracy = sum(metrics['accuracy'])/metrics['num_masks']
    print(f"Number of masks evaluated: {metrics['num_masks']}")
    print("Evaluation metrics:")
    print(f"IOU: {mean_iou:.4f}, DICE: {mean_dice:.4f}, Accuracy: {mean_accuracy:.4f}")
    # Save metrics to a file
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        f.write(f"Mean IOU: {mean_iou:.4f}\n")
        f.write(f"Mean DICE: {mean_dice:.4f}\n")
        f.write(f"Mean Accuracy: {mean_accuracy:.4f}\n")
        f.write("IOU:\n")
        f.write("\n".join(map(str, metrics['iou'])) + "\n")
        f.write("DICE:\n")
        f.write("\n".join(map(str, metrics['dice'])) + "\n")
        f.write("Accuracy:\n")
        f.write("\n".join(map(str, metrics['accuracy'])) + "\n")
    return metrics

# hyperparameter finetuning for segmentation unet
def finetune_seg():
    lr = np.array([5e-4, 1e-4, 5e-5, 1e-5])
    batch_size = np.array([2,4,8])
    epochs = np.array([10]) 
    colour = list(cm.rainbow(np.linspace(0, 1, len(lr) * len(batch_size))))

    iou = []
    dice = []
    accuracy = []

    for i, e in enumerate(epochs):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        for j, b in enumerate(batch_size):
            for k, l in enumerate(lr):
                print(f"Training with batch size: {b}, epochs: {e}, learning rate: {l}, index: {i}{j}{k}")

                train_loss, val_loss = train(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=int(b), lr=l, epoch_number=int(e), index=f"{i}{j}{k}")
                
                unet_path = f"/results/model/{i}{j}{k}_seg_model.pt"
                #make folder:
                results_dir = f"/results/unet_data/results/{i}{j}{k}"
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                
                metrics = results_unet(unet_path=unet_path, results_dir=results_dir, test_dataset=test_dataset, test_pred_mask_dir=test_pred_mask_dir)
        
                iou.append(metrics['iou'])
                dice.append(metrics['dice'])
                accuracy.append(metrics['accuracy'])

                torch.cuda.empty_cache()
                gc.collect()
                #save settings
                Path(results_dir).mkdir(parents=True, exist_ok=True)
                with open(results_dir + "/settings.txt", "w") as f:
                    f.write("Settings:\n")
                    f.write(f"Batch size: {b}\n")
                    f.write(f"Number of epochs: {e}\n")
                    f.write(f"learning rate: {l}\n")
        for ax in (ax1, ax2):
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Losses')
            ax.grid(True)
        
        ax1.set_title('Training loss')
        ax2.set_title('Validation loss')
        plt.tight_layout()
        plt.savefig(f"/results/unet_data/unet_losses.png")
        #plt.show()
        plt.close()

        # test data plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    for i, b in enumerate(batch_size):
        for j, l in enumerate(lr):
            k= lr.size * i + j
            ax1.plot(iou[k],'.', label=f'batch size: {b}, learning rate: {l}, index: 0{i}{j}', color=colour[k])
            ax2.plot(dice[k], '.', label=f'batch size: {b}, learning rate: {l}, index: 0{i}{j}', color=colour[k])
            ax3.plot(accuracy[k], '.', label=f'batch size: {b}, learning rate: {l}, index: 0{i}{j}', color=colour[k])

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel('Samples')
        ax.set_ylabel('Losses')
        ax.legend()
        ax.grid(True)
    
    ax1.set_title('IOU loss')
    ax2.set_title('Dice loss')
    ax3.set_title('Accuracy loss')
    plt.tight_layout()
    plt.savefig(f"/results/unet_data/test_unet_losses.png")
    #plt.show()
    plt.close()

    # test data plot colourbar
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    data_dice = np.zeros((len(batch_size), len(lr)))
    data_iou = np.zeros((len(batch_size), len(lr)))
    data_accuracy = np.zeros((len(batch_size), len(lr)))
    for i, b in enumerate(batch_size):
        for j, l in enumerate(lr):
            k= lr.size * i + j
            data_dice[i,j] = np.mean(dice[k])
            data_iou[i,j] = np.mean(iou[k])
            data_accuracy[i,j] = np.mean(accuracy[k])

    img_iou = ax1.imshow(data_iou, cmap='viridis', aspect='auto')
    img_dice = ax2.imshow(data_dice, cmap='viridis', aspect='auto')
    img_accuracy = ax3.imshow(data_accuracy, cmap='viridis', aspect='auto')

    for i in range(data_iou.shape[0]):
        for j in range(data_iou.shape[1]):
            ax1.text(j, i, f"{data_iou[i, j]:.4f}", ha='center', va='center',
                    color='white' if data_iou[i, j] < np.mean(data_iou) else 'black')
            ax2.text(j, i, f"{data_dice[i, j]:.4f}", ha='center', va='center',
                    color='white' if data_dice[i, j] < np.mean(data_dice)  else 'black')
            ax3.text(j, i, f"{data_accuracy[i, j]:.4f}", ha='center', va='center',
                    color='white' if data_accuracy[i, j] < np.mean(data_accuracy) else 'black')

    for ax in (ax1, ax2, ax3):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(np.arange(data_iou.shape[1]), minor=False)
        ax.set_yticks(np.arange(data_iou.shape[0]), minor=False)
        ax.set_xticklabels(lr, rotation=45)
        ax.set_yticklabels(batch_size)
        ax.invert_yaxis()
        ax.set_xlabel('learning rate')
        ax.set_ylabel('barch size')
    
    colorbar_elbo = fig.colorbar(img_iou, ax=ax1, orientation='vertical')
    colorbar_elbo.set_label('IoU loss')
    colorbar_l2 = fig.colorbar(img_dice, ax=ax2, orientation='vertical')
    colorbar_l2.set_label('Dice loss')
    colorbar_l2 = fig.colorbar(img_accuracy, ax=ax3, orientation='vertical')
    colorbar_l2.set_label('Accuracy loss')
    
    
    ax1.set_title('IoU loss')
    ax2.set_title('Dice loss')
    ax3.set_title('Accuracy loss')
    plt.tight_layout()
    plt.savefig(f"/results/unet_data/test_colorbar_unet.png")
    #plt.show()
    plt.close()

# adjust plots with metrics 
def fix_plot():
    plt.rcParams.update({'font.size': 22})

    results_path = "/results/unet_data/results"
    lr = np.array([5e-4, 1e-4, 5e-5, 1e-5])
    batch_size = np.array([2, 4, 8])
    save_path1 = "/results/unet_data/test_colorbar_unet_squares1.png"

    data_iou = np.zeros((len(batch_size), len(lr)))
    data_dice = np.zeros((len(batch_size), len(lr)))
    data_accuracy = np.zeros((len(batch_size), len(lr)))

    for i, b in enumerate(batch_size):
        for j, l in enumerate(lr):
            metrics_file = os.path.join(results_path, f"0{i}{j}", "metrics.txt")
            if not os.path.exists(metrics_file):
                raise FileNotFoundError(f"Missing file: {metrics_file}")
            with open(metrics_file, "r") as f:
                lines = f.readlines()
                data_iou[i, j] = float(lines[0].split(":")[1])
                data_dice[i, j] = float(lines[1].split(":")[1])
                data_accuracy[i, j] = float(lines[2].split(":")[1])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

    img_iou = ax1.imshow(data_iou, cmap='viridis', aspect='auto')
    img_dice = ax2.imshow(data_dice, cmap='viridis', aspect='auto')
    img_accuracy = ax3.imshow(data_accuracy, cmap='viridis', aspect='auto')

    for i in range(data_iou.shape[0]):
        for j in range(data_iou.shape[1]):
            ax1.text(j, i, f"{data_iou[i, j]:.4f}", ha='center', va='center',
                     fontsize=16,
                     color='white' if data_iou[i, j] < np.mean(data_iou) else 'black')
            ax2.text(j, i, f"{data_dice[i, j]:.4f}", ha='center', va='center',
                     fontsize=16,
                     color='white' if data_dice[i, j] < np.mean(data_dice) else 'black')
            ax3.text(j, i, f"{data_accuracy[i, j]:.4f}", ha='center', va='center',
                     fontsize=16,
                     color='white' if data_accuracy[i, j] < np.mean(data_accuracy) else 'black')

    for ax in (ax1, ax2, ax3):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(np.arange(data_iou.shape[1]), minor=False)
        ax.set_yticks(np.arange(data_iou.shape[0]), minor=False)
        ax.set_xticklabels(lr, rotation=45, fontsize=18)
        ax.set_yticklabels(batch_size, fontsize=18)
        ax.invert_yaxis()
        ax.set_xlabel('learning rate', fontsize=20)
        ax.set_ylabel('batch size', fontsize=20)
    
    colorbar_elbo = fig.colorbar(img_iou, ax=ax1, orientation='vertical')
    colorbar_l2 = fig.colorbar(img_dice, ax=ax2, orientation='vertical')
    colorbar_l2 = fig.colorbar(img_accuracy, ax=ax3, orientation='vertical')
    
    ax1.set_title('IoU', fontsize=24)
    ax2.set_title('Dice Coefficient', fontsize=24)
    ax3.set_title('Accuracy', fontsize=24)
    plt.tight_layout()
    plt.savefig(save_path1, dpi=300)
    plt.close()
    print(f"Saved updated plot with square cells to {save_path1}")

    save_path2 = "/results/unet_data/test_colorbar_unet_squares2.png"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 6))

    img_iou = ax1.imshow(data_iou, cmap='viridis', aspect='auto')
    img_dice = ax2.imshow(data_dice, cmap='viridis', aspect='auto')
    img_accuracy = ax3.imshow(data_accuracy, cmap='viridis', aspect='auto')

    for i in range(data_iou.shape[0]):
        for j in range(data_iou.shape[1]):
            ax1.text(j, i, f"{data_iou[i, j]:.4f}", ha='center', va='center',
                     fontsize=16,
                     color='white' if data_iou[i, j] < np.mean(data_iou) else 'black')
            ax2.text(j, i, f"{data_dice[i, j]:.4f}", ha='center', va='center',
                     fontsize=16,
                     color='white' if data_dice[i, j] < np.mean(data_dice) else 'black')
            ax3.text(j, i, f"{data_accuracy[i, j]:.4f}", ha='center', va='center',
                     fontsize=16,
                     color='white' if data_accuracy[i, j] < np.mean(data_accuracy) else 'black')

    for ax in (ax1, ax2, ax3):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(np.arange(data_iou.shape[1]), minor=False)
        ax.set_yticks(np.arange(data_iou.shape[0]), minor=False)
        ax.set_xticklabels(lr, rotation=45, fontsize=18)
        ax.set_yticklabels(batch_size, fontsize=18)
        ax.invert_yaxis()
        ax.set_xlabel('learning rate', fontsize=20)
        ax.set_ylabel('batch size', fontsize=20)
    
    colorbar_elbo = fig.colorbar(img_iou, ax=ax1, orientation='vertical')
    colorbar_l2 = fig.colorbar(img_dice, ax=ax2, orientation='vertical')
    colorbar_l2 = fig.colorbar(img_accuracy, ax=ax3, orientation='vertical')
    
    ax1.set_title('IoU', fontsize=24)
    ax2.set_title('Dice Coefficient', fontsize=24)
    ax3.set_title('Accuracy', fontsize=24)
    plt.tight_layout()
    plt.savefig(save_path2, dpi=300)
    plt.close()
    print(f"Saved updated plot with square cells to {save_path2}")

if __name__ == "__main__":
    finetune_seg()
    fix_plot()
