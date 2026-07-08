import numpy as np
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import torch


from unet_dataset import UNetDataset
from seg_finetune import results_unet


shuffle_data_loader = False
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


#setting up paths
data_folder = "/data/"
#test data
test_image_dir = data_folder + "/empi_pairs/test/images"
test_mask_dir = data_folder + "/empi_pairs/test/masks"
test_pred_mask_dir = data_folder + "/empi_pairs/test/pred_masks"
test_trans_image_dir = data_folder + "/trans_data/test/images"
test_trans_mask_dir = data_folder + "/trans_data/test/masks"
test_trans_pred_mask_dir = data_folder + "/trans_data/test/pred_masks"
test_trans_compare_mask_dir = data_folder + "trans_data/test/compare_masks"
test_trans_pred_orig_mask_dir = data_folder + "trans_data/test/pred_orig_masks"
test_trans_pred_orig_image_dir = data_folder + "trans_data/test/pred_orig_images"
test_pred_mask_aug_dir = data_folder + "/empi_pairs/test/pred_masks_aug"
test_trans_pred_mask_aug_dir = data_folder + "/trans_data/test/pred_masks_aug"

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
train_pred_mask_aug_dir = data_folder + "/empi_pairs/train/pred_masks_aug"
train_trans_pred_mask_aug_dir = data_folder + "/trans_data/train/pred_masks_aug"

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
val_pred_mask_aug_dir = data_folder + "/empi_pairs/val/pred_masks_aug"
val_trans_pred_mask_aug_dir = data_folder + "/trans_data/val/pred_masks_aug"

#create paths if they do not exist
#if not os.path.exists(test_pred_mask_dir):
os.makedirs(test_pred_mask_aug_dir, exist_ok=True)
os.makedirs(train_pred_mask_aug_dir, exist_ok=True)
os.makedirs(val_pred_mask_aug_dir, exist_ok=True)
os.makedirs(val_trans_pred_mask_aug_dir, exist_ok=True)
os.makedirs(train_trans_pred_mask_aug_dir, exist_ok=True)
os.makedirs(test_trans_pred_mask_aug_dir, exist_ok=True)


def binary_mask_transform(pil_img):
    pil_img = pil_img.convert("L").resize((128, 128))
    np_mask = np.array(pil_img)
    bin_mask = (np_mask > 127).astype(np.uint8)  
    return torch.from_numpy(bin_mask).long()     

#training data
train_trans_dataset = UNetDataset(
    image_dir=train_trans_image_dir,
    mask_dir=train_trans_mask_dir,
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()]),
    mask_transform = binary_mask_transform
    )

#validation data
val_trans_dataset = UNetDataset(
    image_dir=val_trans_image_dir,
    mask_dir=val_trans_mask_dir,
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()]),
    mask_transform = binary_mask_transform
    )

# test data
test_trans_dataset = UNetDataset(
    image_dir=test_trans_image_dir,
    mask_dir=test_trans_mask_dir,
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()]),
    mask_transform = binary_mask_transform
    )

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

# segmentation results without augmentated unet on original data
def seg_results_all_data():
    # unet path
    unet_model_path = "/results/model/0_seg_model.pt"
    results_dir = "/results/unet_data/results_all_data"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    iou = []
    dice = []
    accuracy = []

    # train data
    metrics_train = results_unet(unet_path=unet_model_path, results_dir=results_dir + "/train", test_dataset=train_dataset, test_pred_mask_dir=test_pred_mask_dir)
    iou.append(np.mean(metrics_train['iou']))
    dice.append(np.mean(metrics_train['dice']))
    accuracy.append(np.mean(metrics_train['accuracy']))

    # validation data
    metrics_val = results_unet(unet_path=unet_model_path, results_dir=results_dir + "/val", test_dataset=val_dataset, test_pred_mask_dir=val_pred_mask_dir)
    iou.append(np.mean(metrics_val['iou']))
    dice.append(np.mean(metrics_val['dice']))
    accuracy.append(np.mean(metrics_val['accuracy']))
    
    # test data
    metrics_test = results_unet(unet_path=unet_model_path, results_dir=results_dir + "/test", test_dataset=test_dataset, test_pred_mask_dir=test_pred_mask_dir)
    iou.append(np.mean(metrics_test['iou']))
    dice.append(np.mean(metrics_test['dice']))
    accuracy.append(np.mean(metrics_test['accuracy']))

    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    ax1.plot(iou, 'o')
    ax2.plot(dice, 'o')
    ax3.plot(accuracy, 'o')
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel('Dataset')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks(np.arange(0.9, 1.1, 0.02))
        ax.set_xticklabels(['Train', 'Validation', 'Test'])
        ax.set_ylabel('Losses')
        ax.grid(True)
    
    ax1.set_title('IOU loss')
    ax2.set_title('Dice loss')
    ax3.set_title('Accuracy loss')
    plt.tight_layout()
    plt.savefig(results_dir + "/seg_results_all_data.png")
    #plt.show()
    plt.close()

    return metrics_train, metrics_val, metrics_test


# segmentation results with augmentated unet on transformed data
def seg_trans_results_with_aug():
    unet_model_path = "/results/model/1_seg_model.pt"
    unet_aug_model_path = "/results/model/0_seg_model.pt"

    results_dir = "/results/unet_data/results_all_trans_data"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    iou = []
    dice = []
    accuracy = []

    iou_aug = []
    dice_aug = []
    accuracy_aug = []

    # train data
    metrics_train = results_unet(unet_path=unet_model_path, results_dir=results_dir + "/train",
                                 test_dataset=train_trans_dataset, test_pred_mask_dir=test_trans_pred_mask_dir)
    iou.append(np.mean(metrics_train['iou']))
    dice.append(np.mean(metrics_train['dice']))
    accuracy.append(np.mean(metrics_train['accuracy']))

    # train data with augmentation
    metrics_train_aug = results_unet(unet_path=unet_aug_model_path, results_dir=results_dir + "/train_aug",
                                     test_dataset=train_trans_dataset, test_pred_mask_dir=train_trans_pred_mask_aug_dir)
    iou_aug.append(np.mean(metrics_train_aug['iou']))
    dice_aug.append(np.mean(metrics_train_aug['dice']))
    accuracy_aug.append(np.mean(metrics_train_aug['accuracy']))

    # validation data
    metrics_val = results_unet(unet_path=unet_model_path, results_dir=results_dir + "/val",
                               test_dataset=val_trans_dataset, test_pred_mask_dir=val_trans_pred_mask_dir)
    iou.append(np.mean(metrics_val['iou']))
    dice.append(np.mean(metrics_val['dice']))
    accuracy.append(np.mean(metrics_val['accuracy']))

    # validation data with augmentation
    metrics_val_aug = results_unet(unet_path=unet_aug_model_path, results_dir=results_dir + "/val_aug",
                                   test_dataset=val_trans_dataset, test_pred_mask_dir=val_trans_pred_mask_aug_dir)
    iou_aug.append(np.mean(metrics_val_aug['iou']))
    dice_aug.append(np.mean(metrics_val_aug['dice']))
    accuracy_aug.append(np.mean(metrics_val_aug['accuracy']))

    # test data
    metrics_test = results_unet(unet_path=unet_model_path, results_dir=results_dir + "/test",
                                test_dataset=test_trans_dataset, test_pred_mask_dir=test_trans_pred_mask_dir)
    iou.append(np.mean(metrics_test['iou']))
    dice.append(np.mean(metrics_test['dice']))
    accuracy.append(np.mean(metrics_test['accuracy']))

    # test data with augmentation
    metrics_test_aug = results_unet(unet_path=unet_aug_model_path, results_dir=results_dir + "/test_aug",
                                    test_dataset=test_trans_dataset, test_pred_mask_dir=test_trans_pred_mask_aug_dir)
    iou_aug.append(np.mean(metrics_test_aug['iou']))
    dice_aug.append(np.mean(metrics_test_aug['dice']))
    accuracy_aug.append(np.mean(metrics_test_aug['accuracy']))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    ax1.plot(iou, 'o', label='Without Augmentation', color='blue')
    ax2.plot(dice, 'o', label='Without Augmentation', color='blue')
    ax3.plot(accuracy, 'o', label='Without Augmentation', color='blue')

    ax1.plot(iou_aug, 'o', label='With Augmentation', color='orange')
    ax2.plot(dice_aug, 'o', label='With Augmentation', color='orange')
    ax3.plot(accuracy_aug, 'o', label='With Augmentation', color='orange')

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel('Dataset')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks(np.arange(0.95, 1.02, 0.02))
        ax.set_xticklabels(['Train', 'Validation', 'Test'])
        ax.set_ylabel('Losses')
        ax.legend()
        ax.grid(True)

    ax1.set_title('IOU loss')
    ax2.set_title('Dice loss')
    ax3.set_title('Accuracy loss')
    plt.tight_layout()
    plt.savefig(results_dir + "/seg_results_with_aug.png")
    # plt.show()
    plt.close()

    # saving the metrics in txt file
    with open(results_dir + "/metrics.txt", "w") as f:
        f.write("Metrics without augmentation:\n")
        f.write(f"Train: IOU: {iou[0]}, Dice: {dice[0]}, Accuracy: {accuracy[0]}\n")
        f.write(f"Validation: IOU: {iou[1]}, Dice: {dice[1]}, Accuracy: {accuracy[1]}\n")
        f.write(f"Test: IOU: {iou[2]}, Dice: {dice[2]}, Accuracy: {accuracy[2]}\n\n")

        f.write("Metrics with augmentation:\n")
        f.write(f"Train: IOU: {iou_aug[0]}, Dice: {dice_aug[0]}, Accuracy: {accuracy_aug[0]}\n")
        f.write(f"Validation: IOU: {iou_aug[1]}, Dice: {dice_aug[1]}, Accuracy: {accuracy_aug[1]}\n")
        f.write(f"Test: IOU: {iou_aug[2]}, Dice: {dice_aug[2]}, Accuracy: {accuracy_aug[2]}\n")

    return metrics_train, metrics_val, metrics_test, metrics_train_aug, metrics_val_aug, metrics_test_aug

if __name__ == "__main__":
    #metrics_train, metrics_val, metrics_test = seg_results_all_data()
    seg_trans_results_with_aug()