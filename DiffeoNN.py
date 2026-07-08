import json
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from torchvision.transforms.functional import to_pil_image

from PIL import Image
import os
import shutil
import re
from adversarial_loss_train import convexnet  
import torch.nn.functional as F
from can_reg import get_trans
from seg_finetune import predict
from unet_dataset import UNetDataset
from unet import UNet
from seg_finetune import results_unet
from torch.utils.data import Subset


os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
from torchvision import transforms
from can_layers import SpatialTransformer
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

data_folder = "/data"
#test data
test_image_dir = data_folder + "/empi_pairs/test/images"
test_mask_dir = data_folder + "/empi_pairs/test/masks"
test_pred_mask_dir = data_folder + "/empi_pairs/test/pred_masks"
test_trans_image_dir = data_folder + "/trans_data/test/images"
test_trans_mask_dir = data_folder + "/trans_data/test/masks"
test_trans_pred_mask_dir = data_folder + "/trans_data/test/pred_masks"
test_trans_compare_mask_dir = data_folder + "/trans_data/test/compare_masks"
test_trans_pred_orig_mask_dir = data_folder + "/trans_data/test/pred_orig_masks"
test_trans_pred_orig_image_dir = data_folder + "/trans_data/test/pred_orig_images"
test_trans_pred_mask_aug_dir = data_folder + "/trans_data/test/pred_masks_aug"

#train data
train_image_dir = data_folder + "/empi_pairs/train/images"
train_mask_dir = data_folder + "/empi_pairs/train/masks"
train_pred_mask_dir = data_folder + "/empi_pairs/train/pred_masks"
train_trans_image_dir = data_folder + "/trans_data/train/images"
train_trans_mask_dir = data_folder + "/trans_data/train/masks"
train_trans_pred_mask_dir = data_folder + "/trans_data/train/pred_masks"
train_trans_compare_mask_dir = data_folder + "/trans_data/train/compare_masks"
train_trans_pred_orig_mask_dir = data_folder + "/trans_data/train/pred_orig_masks"
train_trans_pred_orig_image_dir = data_folder + "/trans_data/train/pred_orig_images"
train_trans_pred_mask_aug_dir = data_folder + "/trans_data/train/pred_masks_aug"

#validation data
val_image_dir = data_folder + "/empi_pairs/val/images"
val_mask_dir = data_folder + "/empi_pairs/val/masks"
val_pred_mask_dir = data_folder + "/empi_pairs/val/pred_masks"
val_trans_image_dir = data_folder + "/trans_data/val/images"
val_trans_mask_dir = data_folder + "/trans_data/val/masks"
val_trans_pred_mask_dir = data_folder + "/trans_data/val/pred_masks"
val_trans_compare_mask_dir = data_folder + "/trans_data/val/compare_masks"
val_trans_pred_orig_mask_dir = data_folder + "/trans_data/val/pred_orig_masks"
val_trans_pred_orig_image_dir = data_folder + "/trans_data/val/pred_orig_images"
val_trans_pred_mask_aug_dir = data_folder + "/trans_data/val/pred_masks_aug"

#create paths if they do not exist
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_mask_dir, exist_ok=True)
os.makedirs(test_pred_mask_dir, exist_ok=True)
os.makedirs(test_trans_image_dir, exist_ok=True)
os.makedirs(test_trans_mask_dir, exist_ok=True)
os.makedirs(test_trans_pred_mask_dir, exist_ok=True)
os.makedirs(test_trans_compare_mask_dir, exist_ok=True)
os.makedirs(test_trans_pred_orig_mask_dir, exist_ok=True)
os.makedirs(test_trans_pred_orig_image_dir, exist_ok=True)
os.makedirs(test_trans_pred_mask_aug_dir, exist_ok=True)


os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(train_pred_mask_dir, exist_ok=True)
os.makedirs(train_trans_image_dir, exist_ok=True)
os.makedirs(train_trans_mask_dir, exist_ok=True)
os.makedirs(train_trans_pred_mask_dir, exist_ok=True)
os.makedirs(train_trans_compare_mask_dir, exist_ok=True)
os.makedirs(train_trans_pred_orig_mask_dir, exist_ok=True)
os.makedirs(train_trans_pred_orig_image_dir, exist_ok=True)
os.makedirs(train_trans_pred_mask_aug_dir, exist_ok=True)

os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)
os.makedirs(val_pred_mask_dir, exist_ok=True)
os.makedirs(val_trans_image_dir, exist_ok=True)
os.makedirs(val_trans_mask_dir, exist_ok=True)
os.makedirs(val_trans_pred_mask_dir, exist_ok=True)
os.makedirs(val_trans_compare_mask_dir, exist_ok=True)
os.makedirs(val_trans_pred_orig_mask_dir, exist_ok=True)
os.makedirs(val_trans_pred_orig_image_dir, exist_ok=True)
os.makedirs(val_trans_pred_mask_aug_dir, exist_ok=True)



# datasets for testing augmented segmentation
def binary_mask_transform(pil_img):
    pil_img = pil_img.convert("L").resize((128, 128))
    np_mask = np.array(pil_img)
    bin_mask = (np_mask > 127).astype(np.uint8)  
    return torch.from_numpy(bin_mask).long()     

#training data
train_trans_dataset_full = UNetDataset(
    image_dir=train_trans_image_dir,
    mask_dir=train_trans_mask_dir,
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()]),
    mask_transform = binary_mask_transform
    )

#validation data
val_trans_dataset_full = UNetDataset(
    image_dir=val_trans_image_dir,
    mask_dir=val_trans_mask_dir,
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()]),
    mask_transform = binary_mask_transform
    )

# test data
test_trans_dataset_full = UNetDataset(
    image_dir=test_trans_image_dir,
    mask_dir=test_trans_mask_dir,
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor()]),
    mask_transform = binary_mask_transform
    )

# plotting grid
def plot_grid(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()

# save data to json
def data_file(path, predice, postdice, postdice_b, negdet, ssim_post, ssim_pre, args, i, j, pre_arr, post_arr,
              post_arr_b):
    data = {}
    data['preDiceScore'] = predice
    data['postDiceScore'] = postdice
    data['postDiceScore_b'] = postdice_b
    data['preDiceMean'] = np.mean(pre_arr)
    data['postDiceMean'] = np.mean(post_arr)
    data['postDiceMean_b'] = np.mean(post_arr_b)
    data['foldings'] = negdet.item()
    data['ssim_pre'] = ssim_pre.item()
    data['ssim_post'] = ssim_post.item()
    data['args'] = vars(args)
    datapath = os.path.join(path, 'fix%04dmov%04d.json' % (i, j))
    with open(datapath, 'w+') as fp:
        json.dump(data, fp)
    return 0

# create grid for warping
def create_grid(size, device=device): 
    # create sampling grid
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor)

    if len(size) == 2:
        grids = torch.meshgrid(*vectors, indexing='ij')

    return grid.to(device)

# prune images to overlapping region
def prune_image(fix_img, mov_img):
    fix_z = fix_img.mean(axis=0).mean(axis=0)
    mov_z = mov_img.mean(axis=0).mean(axis=0)
    arg_f = np.argwhere(fix_z > 0.01)
    arg_m = np.argwhere(mov_z > 0.01)
    lower, upper = max(min(arg_f), min(arg_m))[0], min(max(arg_f), max(arg_m))[0]
    return fix_img[:, :, lower:upper + 1], mov_img[:, :, lower:upper + 1], lower, upper

np.random.seed(0)
torch.set_default_dtype(torch.float64)
# warp mask function
def warp_mask(mask_M2_np, norm_grid, device):
    if mask_M2_np.ndim == 3:
        mask_M2_np = mask_M2_np[..., 0]
    H, W = mask_M2_np.shape
    mask_M2 = torch.from_numpy(mask_M2_np).float().unsqueeze(0).unsqueeze(0).to(device)

    # use norm_grid directly
    grid = norm_grid.permute(0, 2, 3, 1)  # [1, H, W, 2], already in [-1, 1]
    warped = F.grid_sample(mask_M2, grid, mode='nearest', padding_mode='zeros', align_corners=True)
    return warped

# get warped mask
def get_mask(mask_path, save_path, new_locs_backward, save=True):
    mask_M2_np = np.array(Image.open(mask_path).convert("L").resize((128, 128))) / 255.0
    mask_M2_np = (mask_M2_np >= 0.5).astype(np.float32)

    # convert to tensor if needed
    if isinstance(new_locs_backward, np.ndarray):
        new_locs_backward = torch.from_numpy(new_locs_backward).float().to(device)

    M1_tensor = warp_mask(mask_M2_np, new_locs_backward, device)
    M1_np = M1_tensor.squeeze().cpu().numpy()

    if save:
        Image.fromarray((M1_np >= 0.5).astype(np.uint8) * 255).save(save_path)
    return M1_np

# visualisation DiffeoNN
def visualise(fixed_img, moved_img, warped_img, new_locs, new_locs_backward, inshape, M1_np, M2_np, gt_M, index=0, output_dir=None):
    """
    Visualize the results of the registration.
    """
    
    H, W = [dim for dim in inshape]

    # transformation grid
    fig, ax = plt.subplots()
    plt.axis('off')
    plt.tight_layout()
    plot_grid(new_locs[0, 1], -new_locs[0, 0], ax=ax, color="C0")
    plt.savefig(output_dir+ f'{index}_forward.pdf', bbox_inches='tight')
    plt.show()
    fig, ax = plt.subplots()
    plt.axis('off')
    plt.tight_layout()
    plt.title('warped Img')
    plt.imshow(warped_img)
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    plt.axis('off')
    plt.tight_layout()
    plt.title('Difference between picture and predicted mask')
    plt.imshow(moved_img - M1_np)
    plt.show()

    plt.imsave(output_dir+ f'{index}_pred_mask_diff.png', moved_img - M1_np)
    plt.close()

    fig, ax = plt.subplots()
    plt.axis('off')
    plt.tight_layout()
    plt.title('Predicted input mask')
    plt.imshow(M1_np)
    plt.show()

    plt.imsave(output_dir+ f'{index}_pred_orig_mask.png', M1_np)
    plt.close()

    fig, ax = plt.subplots()
    plt.axis('off')
    plt.tight_layout()
    plt.title('Difference between picture and predicted mask')
    plt.imshow(moved_img - M2_np)
    plt.show()

    plt.imsave(output_dir+ f'{index}_warped_mask_diff.png', warped_img - M2_np)
    plt.close()

    fig, ax = plt.subplots()
    plt.axis('off')
    plt.tight_layout()
    plt.title('Predicted canonicalised mask')
    plt.imshow(M2_np)
    plt.show()

    plt.imsave(output_dir+ f'{index}_warped_mask.png', M2_np)
    plt.close()


    fig, ax = plt.subplots()

    colors_d = np.zeros((H, W, 3))
    colors_d[..., 0] = moved_img[:, :]
    colors_d[..., 1] = 0.5 * (moved_img[:, :] + warped_img[:, :])
    colors_d[..., 2] = warped_img[:, :]
    plt.imshow(colors_d)
    plt.title('Overlay in the end')
    plt.axis('off')
    plt.show()
    plt.imsave(output_dir+ f'{index}_transformation_overlay.png', colors_d)
    plt.close()


    fig, ax = plt.subplots()
    colors_m = np.zeros((H, W, 3))
    colors_m[..., 0] = gt_M[:, :]
    colors_m[..., 1] = 0.5 * (gt_M[:, :] + M1_np[:, :])
    colors_m[..., 2] = M1_np[:, :]
    plt.imshow(colors_m)
    plt.title('Difference between ground truth and predicted mask')
    plt.axis('off')
    plt.show()
    colors_m_norm = (colors_m - colors_m.min()) / (colors_m.max() - colors_m.min() + 1e-8)
    plt.imsave(output_dir + f'{index}_mask_diff.png', colors_m_norm)
    #plt.imsave(output_dir+ f'{index}_mask_diff.png', colors_m)
    plt.close()


    fig, ax = plt.subplots()
    plt.axis('off')
    plt.tight_layout()
    plt.title('Ground truth mask')
    plt.imshow(gt_M)
    plt.show()

    plt.imsave(output_dir+ f'{index}_gt_mask.png', gt_M) # YlGnBn
    plt.close()

    plt.imsave(output_dir+ f'{index}_fixed.png',
               np.stack((fixed_img[:, :], fixed_img[:,  :], fixed_img[:, :]), axis=-1), vmin=0, vmax=255, cmap='PuBu')
    plt.imsave(output_dir+ f'{index}_moving.png',
               np.stack((moved_img[:, :], moved_img[:, :], moved_img[:,  :]), axis=-1), vmin=0, vmax=255, cmap='PuBu')
    plt.imsave(output_dir+ f'{index}_warped.png',
               np.stack((warped_img[:, :], warped_img[:,  :], warped_img[:,  :]), axis=-1), vmin=0, vmax=255, cmap='PuBu')

# metrics - iou, dice, accuracy
def metrics_trans(n=0, mask_dir=None, pred_mask_dir=None, compare_mask_dir=None, results_dir=None):
    iou = []
    dice = []
    accuracy = []
    total_pixels = 128 * 128
    num_masks = 0
    iou_comp = []
    dice_comp = []
    accuracy_comp = []

    for i in range(n):

        # load mask and binarize
        mask_path = os.path.join(mask_dir, f"{i+1}.png")
        mask_image = Image.open(mask_path).convert("L").resize((128, 128))
        mask_array = (np.array(mask_image) > 127).astype(np.uint8)

        # Load predicted mask and binarize
        pred_mask_path = os.path.join(pred_mask_dir, f"{i+1}.png")
        pred_mask = Image.open(pred_mask_path).convert("L").resize((128, 128))
        pred_mask_array = (np.array(pred_mask) > 127).astype(np.uint8)

        # load compare mask and binarize
        compare_mask_path = os.path.join(compare_mask_dir, f"{i+1}.png")
        compare_mask = Image.open(compare_mask_path).convert("L").resize((128, 128))
        compare_mask_array = (np.array(compare_mask) > 127).astype(np.uint8)

        # Optionally save some masks for visual inspection
        z=1
        if i % 1 == 0:
            Image.fromarray(mask_array * 255).save(os.path.join(results_dir, f"{int((i / 1)+1)}_mask.png"))
            Image.fromarray(pred_mask_array * 255).save(os.path.join(results_dir, f"{int((i / 1)+1)}_pred_mask.png"))
            Image.fromarray(compare_mask_array * 255).save(os.path.join(results_dir, f"{int((i / 1)+1)}_compare_mask.png"))
            # save difference image
            diff_image = np.abs(mask_array - pred_mask_array) * 255
            Image.fromarray(diff_image).save(os.path.join(results_dir, f"{int((i / 1)+1)}_diff_mask.png"))
            # difference with compare mask
            diff_compare_image = np.abs(mask_array - compare_mask_array) * 255
            Image.fromarray(diff_compare_image).save(os.path.join(results_dir, f"{int((i / 1)+1)}_diff_compare_mask.png"))

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

        # metrics for compare mask
        intersection_comp = np.logical_and(mask_array, compare_mask_array).sum()
        union_comp = np.logical_or(mask_array, compare_mask_array).sum()
        mask_sum_comp = mask_array.sum() + compare_mask_array.sum()
        iou_item_comp = intersection_comp / union_comp if union_comp != 0 else 0.0
        dice_item_comp = 2 * intersection_comp / mask_sum_comp if mask_sum_comp != 0 else 0.0
        acc_item_comp = (compare_mask_array == mask_array).sum() / total_pixels
        iou_comp.append(iou_item_comp)
        dice_comp.append(dice_item_comp)
        accuracy_comp.append(acc_item_comp)

        num_masks += 1

    # Plot iou loss
    plt.figure(figsize=(10, 5))
    plt.plot(iou, label='IOU')
    plt.plot(iou_comp, label='IOU without Canonicalisation', linestyle='--')
    plt.xlabel('Testdataset')
    plt.ylabel('Losses')
    plt.title('Masks Results')
    plt.legend()
    plt.savefig(results_dir + f"/iou_losses.png")
    plt.close()

    mean_iou = np.mean(iou)
    mean_iou_comp = np.mean(iou_comp)

    # Plot dice loss
    plt.figure(figsize=(10, 5))
    plt.plot(dice, label='Dice')
    plt.plot(dice_comp, label='Dice without Canonicalisation', linestyle='--')
    plt.xlabel('Testdataset')
    plt.ylabel('Losses')
    plt.title('Masks Results')
    plt.legend()
    plt.savefig(results_dir + f"/dice_losses.png")
    plt.close()

    mean_dice = np.mean(dice)
    mean_dice_comp = np.mean(dice_comp)

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy, label='Accuracy')
    plt.plot(accuracy_comp, label='Accuracy without Canonicalisation', linestyle='--')
    plt.xlabel('Testdataset')
    plt.ylabel('Losses')
    plt.title('Masks Results')
    plt.legend()
    plt.savefig(results_dir + f"/accuracy_losses.png")
    plt.close()

    mean_accuracy = np.mean(accuracy)
    mean_accuracy_comp = np.mean(accuracy_comp)

    return {
        "iou": iou ,
        "dice": dice ,
        "accuracy": accuracy,
        "iou_comp": iou_comp,
        "dice_comp": dice_comp,
        "accuracy_comp": accuracy_comp,
        "num_masks": num_masks,
        "mean_iou": mean_iou,
        "mean_iou_comp": mean_iou_comp,
        "mean_dice": mean_dice,
        "mean_dice_comp": mean_dice_comp,
        "mean_accuracy": mean_accuracy,
        "mean_accuracy_comp": mean_accuracy_comp
    }


# get non-faulty indices
def get_unfaulty_indices(path=None, n=0, faulty_filename="faulty_transformed_images.txt"):
    # Load faulty filenames
    faulty_path = os.path.join(path, faulty_filename)
    if os.path.exists(faulty_path):
        with open(faulty_path, "r") as f:
            faulty_filenames = set(line.strip() for line in f)
    else:
        faulty_filenames = set()

    # Get all valid .png filenames, sorted numerically
    image_filenames = sorted(
        [f for f in os.listdir(path) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    # Select first `n` non-faulty filenames
    selected_filenames = []
    for fname in image_filenames:
        if fname not in faulty_filenames:
            selected_filenames.append(fname)
        if len(selected_filenames) == n:
            break

    if len(selected_filenames) < n:
        raise ValueError(f"Only {len(selected_filenames)} non-faulty images found in {path}, but {n} requested.")

    # Return indices assuming filenames are like '1.png', '2.png', etc.
    selected_indices = [int(os.path.splitext(f)[0]) - 1 for f in selected_filenames]
    return selected_indices

# apply canonicalisation and segmentation to transformed squares
def test_trans(n=10, dataset=None, output_dir=None, adv_weight = 0.001, vae_weight=0.001):
    # dataset: None, 'test', 'train', 'val'
    unet_model_path = "/results/model/1_seg_model.pt"
    unet_model = UNet(dimensions=2)
    checkpoint = torch.load(unet_model_path, map_location=device)
    unet_model.load_state_dict(checkpoint)
    unet_model.eval()
    unet_model.to(device)
    print(f"Testing network with {n} images (transformed squares from {dataset} dataset)")

    if dataset == 'test':
        orig_image_dir = test_image_dir
        trans_image_dir = test_trans_image_dir
        pred_orig_mask_dir = test_trans_pred_orig_mask_dir
        pred_trans_mask_dir = test_trans_pred_mask_dir
        trans_mask_dir = test_trans_mask_dir
        compare_mask_dir = test_trans_compare_mask_dir
        output_dir = output_dir + "test/"
    elif dataset == 'train':
        orig_image_dir = train_image_dir
        trans_image_dir = train_trans_image_dir
        pred_orig_mask_dir = train_trans_pred_orig_mask_dir
        pred_trans_mask_dir = train_trans_pred_mask_dir
        trans_mask_dir = train_trans_mask_dir
        compare_mask_dir = train_trans_compare_mask_dir
        output_dir = output_dir + "train/"
    elif dataset == 'val':
        orig_image_dir = val_image_dir
        trans_image_dir = val_trans_image_dir
        pred_orig_mask_dir = val_trans_pred_orig_mask_dir
        pred_trans_mask_dir = val_trans_pred_mask_dir
        trans_mask_dir = val_trans_mask_dir
        compare_mask_dir = val_trans_compare_mask_dir
        output_dir = output_dir + "val/"
    else:
        raise ValueError("Dataset must be 'test', 'train', or 'val'.")

    image_filenames = sorted(
        [f for f in os.listdir(trans_image_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    os.makedirs(output_dir, exist_ok=True)
    loss_list = []

    # process each image
    for i in range(n):
        fname = image_filenames[i]

        fix_path = os.path.join(orig_image_dir, fname)
        mov_path = os.path.join(trans_image_dir, fname)
        pred_mask_path = os.path.join(pred_orig_mask_dir, fname)
        pred_trans_mask_path = os.path.join(pred_trans_mask_dir, fname)
        trans_mask_path = os.path.join(trans_mask_dir, fname)
        compare_mask_path = os.path.join(compare_mask_dir, fname)

        discriminator = convexnet().to(device)
        discriminator.load_state_dict(torch.load("/results/model/adversarial_discriminator_otf.pth", map_location=device)) #for transformed squares
        discriminator.eval()

        # get fixed and moving images, warped image, new locations, and backward locations
        fixed_img, moved_img, warped_img, new_locs, new_locs_backward, neg_flow, inshape, first_loss, final_loss = get_trans(fix_path=fix_path, mov_path=mov_path, adv_weight = adv_weight, vae_weight_var=vae_weight)

        loss_list.append(final_loss)

        img = warped_img
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = unet_model(img_tensor)
            pred_mask = output.argmax(dim=1).squeeze(0)

        # Save and reload it (or just keep in memory)
        to_pil_image(pred_mask.byte() * 255).save(pred_mask_path)

        # 3. Warp predicted square mask back using neg_flow
        square_mask_np = (pred_mask > 0).float().cpu().numpy()
        square_mask_tensor = torch.from_numpy(square_mask_np).unsqueeze(0).unsqueeze(0).to(device)

        square_mask_tensor = square_mask_tensor.float()
        neg_flow = neg_flow.float()

        transformer = SpatialTransformer(size=(128, 128)).to(device)
        warped_mask, _ = transformer(square_mask_tensor, neg_flow, mode='bilinear')
        M1_np = warped_mask.detach().cpu().squeeze().numpy()
        # Save new mask
        Image.fromarray((M1_np >= 0.5).astype(np.uint8) * 255).save(pred_trans_mask_path)

        # predict mask without Canonicalisation
        img2 = moved_img
        img_tensor2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = unet_model(img_tensor2)
            pred_mask2 = output.argmax(dim=1).squeeze(0)

        # Save and reload it (or just keep in memory)
        to_pil_image(pred_mask2.byte() * 255).save(compare_mask_path)

        square_mask2_np = (pred_mask2 > 0).float().cpu().numpy()

        if i% 1 == 0:
            print(f"Visualizing {fname} ({i+1}/{n})")
            # visualize the results
            gt_mask = np.array(Image.open(trans_mask_path).convert("L").resize((128, 128))) / 255.0
            gt_mask = (gt_mask >= 0.5).astype(np.float32)
            visualise(fixed_img=fixed_img, moved_img=moved_img, warped_img=warped_img, new_locs=new_locs, new_locs_backward=new_locs_backward, inshape=inshape, M1_np=M1_np, M2_np=square_mask_np, gt_M= gt_mask, index=i+1, output_dir=output_dir)

    # compute metrics
    metrics = metrics_trans(n=n, mask_dir=trans_mask_dir, pred_mask_dir= pred_trans_mask_dir, compare_mask_dir=compare_mask_dir, results_dir=output_dir)
    print(f"Testing network with {n} images (transformed squares)")
    print(f"With Canonicalisation: mean iou: {metrics['mean_iou']}, mean dice: {metrics['mean_dice']}, mean accuracy: {metrics['mean_accuracy']}")
    print(f"Without Canonicalisation: mean iou: {metrics['mean_iou_comp']}, mean dice: {metrics['mean_dice_comp']}, mean accuracy: {metrics['mean_accuracy_comp']}")
    # save the metrics to a file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"With Canonicalisation: mean iou: {metrics['mean_iou']}, mean dice: {metrics['mean_dice']}, mean accuracy: {metrics['mean_accuracy']}\n")
        f.write(f"Without Canonicalisation: mean iou: {metrics['mean_iou_comp']}, mean dice: {metrics['mean_dice_comp']}, mean accuracy: {metrics['mean_accuracy_comp']}\n")
        f.write(f"Number of masks: {metrics['num_masks']}\n")
        f.write(f"IOU: {metrics['iou']}\n")
        f.write(f"Dice: {metrics['dice']}\n")
        f.write(f"Accuracy: {metrics['accuracy']}\n")
        f.write(f"IOU without Canonicalisation: {metrics['iou_comp']}\n")
        f.write(f"Dice without Canonicalisation: {metrics['dice_comp']}\n")
        f.write(f"Accuracy without Canonicalisation: {metrics['accuracy_comp']}\n")

    return metrics, loss_list

# code for benchmarking with only test dataset presented in boxplot
def benchmarking_boxplot_only_test():

    n = 100
    adv_weight = 0.001
    vae_weight = 0.01
    output_dir = '/results/test_can_data/trans_data/boxplot_test'

    unet_aug_model_path = "/results/model/0_seg_model.pt"
    test_selected_indices = get_unfaulty_indices(path=test_trans_image_dir, n=n)
    test_trans_dataset = Subset(test_trans_dataset_full, test_selected_indices)


    # test data with augmentation
    metrics_test_aug = results_unet(unet_path=unet_aug_model_path, results_dir=output_dir + "/test_aug",
                                    test_dataset=test_trans_dataset, test_pred_mask_dir=test_trans_pred_mask_aug_dir)
    iou_aug = metrics_test_aug['iou']
    dice_aug = metrics_test_aug['dice']
    accuracy_aug = metrics_test_aug['accuracy']

    test_metrics, test_loss_list = test_trans(n=n, dataset='test', output_dir=output_dir, adv_weight=adv_weight, vae_weight=vae_weight)
    iou = test_metrics['iou']
    dice = test_metrics['dice']
    accuracy = test_metrics['accuracy']
    iou_comp = test_metrics['iou_comp']
    dice_comp = test_metrics['dice_comp']
    accuracy_comp = test_metrics['accuracy_comp']

    # plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # Labels
    labels = ['UNet with Augmentation', 'U-Net without Augmentation', 'DiffeoNN']

    # --- IOU ---
    ax1.boxplot([iou_aug, iou_comp, iou], labels=labels, patch_artist=True)
    means = [np.mean(iou_aug), np.mean(iou_comp), np.mean(iou)]
    ax1.scatter([1, 2, 3], means, color='red', marker='o', s=60, label='Mean')
    ax1.set_title('IOU')
    ax1.set_ylabel('Scores')
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True)
    ax1.legend()

    # --- Dice ---
    ax2.boxplot([dice_aug, dice_comp, dice], labels=labels, patch_artist=True)
    means = [np.mean(dice_aug), np.mean(dice_comp), np.mean(dice)]
    ax2.scatter([1, 2, 3], means, color='red', marker='o', s=60, label='Mean')
    ax2.set_title('Dice Score')
    ax2.set_ylabel('Scores')
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True)
    ax2.legend()

    # --- Accuracy ---
    ax3.boxplot([accuracy_aug, accuracy_comp, accuracy], labels=labels, patch_artist=True)
    means = [np.mean(accuracy_aug), np.mean(accuracy_comp), np.mean(accuracy)]
    ax3.scatter([1, 2, 3], means, color='red', marker='o', s=60, label='Mean')
    ax3.set_title('Accuracy')
    ax3.set_ylabel('Scores')
    ax3.set_ylim(0.0, 1.05)
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_dir + "/benchmarking_results_test_boxplot.png")
    plt.close()

    # save all metrics arrays for later use
    metrics_data = {
        "iou": iou,
        "dice": dice,
        "accuracy": accuracy,
        "iou_comp": iou_comp,
        "dice_comp": dice_comp,
        "accuracy_comp": accuracy_comp,
        "iou_aug": iou_aug,
        "dice_aug": dice_aug,
        "accuracy_aug": accuracy_aug,
    }
    with open(os.path.join(output_dir, "benchmarking_metrics_test.pkl"), "wb") as f:
        pickle.dump(metrics_data, f)

    return iou, dice, accuracy, iou_comp, dice_comp, accuracy_comp, iou_aug, dice_aug, accuracy_aug


# code for benchmarking boxplot from saved metrics
def bp_iou_from_metrics(metrics_path, output_dir, filter_threshold=0.0):
    """
    Loads saved metrics from .npz, filters out cases where DiffeoNN IOU < threshold,
    and re-plots the benchmarking boxplots. Also plots per-sample differences
    (DiffeoNN - U-Net without Augmentation) before filtering.

    Args:
        metrics_path (str): Path to saved metrics file (.npz expected).
        output_dir (str): Directory where plots will be saved.
        filter_threshold (float): Minimum IOU to keep for DiffeoNN.
    """


    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

    iou = np.array(metrics['iou'])
    dice = np.array(metrics['dice'])
    accuracy = np.array(metrics['accuracy'])
    iou_comp = np.array(metrics['iou_comp'])
    dice_comp = np.array(metrics['dice_comp'])
    accuracy_comp = np.array(metrics['accuracy_comp'])
    iou_aug = np.array(metrics['iou_aug'])
    dice_aug = np.array(metrics['dice_aug'])
    accuracy_aug = np.array(metrics['accuracy_aug'])

    print(
        f"Augmented U-Net - mean IOU: {np.mean(iou_aug)}, mean Dice: {np.mean(dice_aug)}, mean Accuracy: {np.mean(accuracy_aug)}")
    print(
        f"U-Net w/o Augmentation - mean IOU: {np.mean(iou_comp)}, mean Dice: {np.mean(dice_comp)}, mean Accuracy: {np.mean(accuracy_comp)}")
    print(f"DiffeoNN - mean IOU: {np.mean(iou)}, mean Dice: {np.mean(dice)}, mean Accuracy: {np.mean(accuracy)}")

    # save in .txt
    metrics_file = os.path.join(output_dir, 'benchmarking_metrics_summary.txt')
    with open(metrics_file, 'w') as f:
        f.write(
            f"Augmented U-Net - mean IOU: {np.mean(iou_aug)}, mean Dice: {np.mean(dice_aug)}, mean Accuracy: {np.mean(accuracy_aug)}\n")
        f.write(
            f"U-Net w/o Augmentation - mean IOU: {np.mean(iou_comp)}, mean Dice: {np.mean(dice_comp)}, mean Accuracy: {np.mean(accuracy_comp)}\n")
        f.write(
            f"DiffeoNN - mean IOU: {np.mean(iou)}, mean Dice: {np.mean(dice)}, mean Accuracy: {np.mean(accuracy)}\n")


    # --- Filter: keep only entries where DiffeoNN IOU >= threshold ---
    mask = iou >= filter_threshold
    iou_filt, dice_filt, accuracy_filt = iou[mask], dice[mask], accuracy[mask]
    iou_comp_filt, dice_comp_filt, accuracy_comp_filt = iou_comp[mask], dice_comp[mask], accuracy_comp[mask]
    iou_aug_filt, dice_aug_filt, accuracy_aug_filt = iou_aug[mask], dice_aug[mask], accuracy_aug[mask]

    # --- Boxplots after filtering ---
    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 7))
    labels = ['Naïve', 'DiffeoNN', 'Aug.']

    # IOU
    ax1.boxplot([iou_comp_filt, iou_filt, iou_aug_filt], labels=labels, patch_artist=True)
    means_iou = [np.mean(iou_comp_filt), np.mean(iou_filt), np.mean(iou_aug_filt)]
    ax1.set_title('IoU')
    #ax1.set_ylabel('Scores', fontsize=14)

    # make x and y tick labels bigger
    ax1.tick_params(axis='both', labelsize=12)

    ax1.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iou_boxplot.png"))
    plt.close()

    fig, (ax2) = plt.subplots(1, 1, figsize=(6, 6))
    labels = ['Naïve', 'DiffeoNN', 'Augmented']

    # Dice
    ax2.boxplot([dice_comp_filt, dice_filt, dice_aug_filt], labels=labels, patch_artist=True)
    means_dice = [np.mean(dice_comp_filt), np.mean(dice_filt), np.mean(dice_aug_filt)]
    #ax2.set_title('Dice Coefficient')
    ax2.set_ylabel('Dice Coefficient', fontsize=14)

    # make x and y tick labels bigger
    ax2.tick_params(axis='both', labelsize=12)

    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dice_boxplot.png"))
    plt.close()

    return (iou_filt, dice_filt, accuracy_filt,
            iou_comp_filt, dice_comp_filt, accuracy_comp_filt,
            iou_aug_filt, dice_aug_filt, accuracy_aug_filt)

def bp_iou_from_metrics2(metrics_path, output_dir, filter_threshold=0.0):
    """
    Loads saved metrics from .npz, filters out cases where DiffeoNN IOU < threshold,
    and re-plots the benchmarking boxplots. Also saves mean ± std (var) like in papers.

    Args:
        metrics_path (str): Path to saved metrics file (.npz expected).
        output_dir (str): Directory where plots will be saved.
        filter_threshold (float): Minimum IOU to keep for DiffeoNN.
    """

    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

    iou = np.array(metrics['iou'])
    dice = np.array(metrics['dice'])
    accuracy = np.array(metrics['accuracy'])
    iou_comp = np.array(metrics['iou_comp'])
    dice_comp = np.array(metrics['dice_comp'])
    accuracy_comp = np.array(metrics['accuracy_comp'])
    iou_aug = np.array(metrics['iou_aug'])
    dice_aug = np.array(metrics['dice_aug'])
    accuracy_aug = np.array(metrics['accuracy_aug'])


    def stats_str(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        var = np.var(arr)
        return f"{mean:.4f} ± {std:.4f} (var={var:.4f})"


    # ---------------- BEFORE FILTERING ----------------
    print("\n=== BEFORE FILTERING ===")
    print(f"Augmented U-Net  IOU: {stats_str(iou_aug)}")
    print(f"U-Net (Naïve)    IOU: {stats_str(iou_comp)}")
    print(f"DiffeoNN         IOU: {stats_str(iou)}")

    print(f"Augmented U-Net  Dice: {stats_str(dice_aug)}")
    print(f"U-Net (Naïve)    Dice: {stats_str(dice_comp)}")
    print(f"DiffeoNN         Dice: {stats_str(dice)}")

    print(f"Augmented U-Net  Acc: {stats_str(accuracy_aug)}")
    print(f"U-Net (Naïve)    Acc: {stats_str(accuracy_comp)}")
    print(f"DiffeoNN         Acc: {stats_str(accuracy)}")


    # save in .txt
    metrics_file = os.path.join(output_dir, 'benchmarking_metrics_summary.txt')
    with open(metrics_file, 'w') as f:
        f.write("=== BEFORE FILTERING ===\n")

        f.write(f"Augmented U-Net  IOU: {stats_str(iou_aug)}\n")
        f.write(f"U-Net (Naïve)    IOU: {stats_str(iou_comp)}\n")
        f.write(f"DiffeoNN         IOU: {stats_str(iou)}\n\n")

        f.write(f"Augmented U-Net  Dice: {stats_str(dice_aug)}\n")
        f.write(f"U-Net (Naïve)    Dice: {stats_str(dice_comp)}\n")
        f.write(f"DiffeoNN         Dice: {stats_str(dice)}\n\n")

        f.write(f"Augmented U-Net  Acc: {stats_str(accuracy_aug)}\n")
        f.write(f"U-Net (Naïve)    Acc: {stats_str(accuracy_comp)}\n")
        f.write(f"DiffeoNN         Acc: {stats_str(accuracy)}\n\n")


    # --- Filter ---
    mask = iou >= filter_threshold

    iou_filt, dice_filt, accuracy_filt = iou[mask], dice[mask], accuracy[mask]
    iou_comp_filt, dice_comp_filt, accuracy_comp_filt = iou_comp[mask], dice_comp[mask], accuracy_comp[mask]
    iou_aug_filt, dice_aug_filt, accuracy_aug_filt = iou_aug[mask], dice_aug[mask], accuracy_aug[mask]


    # ---------------- AFTER FILTERING ----------------
    print("\n=== AFTER FILTERING ===")
    print(f"Augmented U-Net  IOU: {stats_str(iou_aug_filt)}")
    print(f"U-Net (Naïve)    IOU: {stats_str(iou_comp_filt)}")
    print(f"DiffeoNN         IOU: {stats_str(iou_filt)}")

    print(f"Augmented U-Net  Dice: {stats_str(dice_aug_filt)}")
    print(f"U-Net (Naïve)    Dice: {stats_str(dice_comp_filt)}")
    print(f"DiffeoNN         Dice: {stats_str(dice_filt)}")

    print(f"Augmented U-Net  Acc: {stats_str(accuracy_aug_filt)}")
    print(f"U-Net (Naïve)    Acc: {stats_str(accuracy_comp_filt)}")
    print(f"DiffeoNN         Acc: {stats_str(accuracy_filt)}")


    with open(metrics_file, 'a') as f:
        f.write("=== AFTER FILTERING ===\n")

        f.write(f"Augmented U-Net  IOU: {stats_str(iou_aug_filt)}\n")
        f.write(f"U-Net (Naïve)    IOU: {stats_str(iou_comp_filt)}\n")
        f.write(f"DiffeoNN         IOU: {stats_str(iou_filt)}\n\n")

        f.write(f"Augmented U-Net  Dice: {stats_str(dice_aug_filt)}\n")
        f.write(f"U-Net (Naïve)    Dice: {stats_str(dice_comp_filt)}\n")
        f.write(f"DiffeoNN         Dice: {stats_str(dice_filt)}\n\n")

        f.write(f"Augmented U-Net  Acc: {stats_str(accuracy_aug_filt)}\n")
        f.write(f"U-Net (Naïve)    Acc: {stats_str(accuracy_comp_filt)}\n")
        f.write(f"DiffeoNN         Acc: {stats_str(accuracy_filt)}\n\n")


    # --- Boxplots ---
    fig, (ax1) = plt.subplots(1, 1, figsize=(7, 7))
    labels = ['Naïve', 'DiffeoNN', 'Aug.']

    ax1.boxplot([iou_comp_filt, iou_filt, iou_aug_filt],
                labels=labels,
                patch_artist=True)

    ax1.set_title('IoU')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iou_boxplot.png"))
    plt.close()


    fig, (ax2) = plt.subplots(1, 1, figsize=(6, 6))
    labels = ['Naïve', 'DiffeoNN', 'Augmented']

    ax2.boxplot([dice_comp_filt, dice_filt, dice_aug_filt],
                labels=labels,
                patch_artist=True)

    ax2.set_ylabel('Dice Coefficient', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dice_boxplot.png"))
    plt.close()


    return (iou_filt, dice_filt, accuracy_filt,
            iou_comp_filt, dice_comp_filt, accuracy_comp_filt,
            iou_aug_filt, dice_aug_filt, accuracy_aug_filt)

# statistics
def full_stats(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    var = np.var(arr)

    median = np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1

    whisk_low = q1 - 1.5 * iqr
    whisk_high = q3 + 1.5 * iqr

    return {
        "mean": mean,
        "std": std,
        "var": var,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "whisk_low": whisk_low,
        "whisk_high": whisk_high
    }

def stats_to_string(name, arr):
    s = full_stats(arr)

    return (
        f"{name}\n"
        f"  mean±std: {s['mean']:.4f} ± {s['std']:.4f}  (var={s['var']:.4f})\n"
        f"  median[IQR]: {s['median']:.4f} [{s['q1']:.4f}–{s['q3']:.4f}]\n"
        f"  whiskers: [{s['whisk_low']:.4f}, {s['whisk_high']:.4f}]\n"
    )


# Extract leading number (works for both mask types)
def extract_number(filename):
    match = re.match(r"(\d+)", filename)
    return int(match.group(1))

def fix_saving_order(input_folder = None, output_folder=None):
    os.makedirs(output_folder, exist_ok=True)

    # Get all relevant files (_mask.png or _pred_mask.png)
    files = [
        f for f in os.listdir(input_folder)
        if f.endswith("_mask.png") or f.endswith("_pred_mask.png")
    ]

    # Get unique numbers (1..100)
    numbers = sorted(set(extract_number(f) for f in files))

    # Numeric order (1,2,3,...,100)
    numeric_order = sorted(numbers)

    # Lexicographic order (1,10,100,11,...)
    lexicographic_order = sorted(numbers, key=lambda x: str(x))

    # Create mapping: numeric -> lexicographic
    mapping = dict(zip(numeric_order, lexicographic_order))

    # Rename and copy files
    for file in files:
        number = extract_number(file)
        new_number = mapping[number]

        # Replace only the leading number
        new_filename = re.sub(r"^\d+", str(new_number), file)

        # Replace _pred_mask with pred_mask_aug
        new_filename = new_filename.replace("_pred_mask", "_pred_mask_aug")

        src_path = os.path.join(input_folder, file)
        dst_path = os.path.join(output_folder, new_filename)

        shutil.copy(src_path, dst_path)

    print("Reordering complete for both mask and pred_mask files!")



if __name__ == '__main__':
    output_dir = '/results/test_can_data/trans_data/boxplot_test'
    metrics_path = os.path.join(output_dir, "benchmarking_metrics_test.pkl")
    benchmarking_boxplot_only_test()

    # Call the function
    filtered_results = bp_iou_from_metrics2(
        metrics_path=metrics_path,
        output_dir=output_dir,
        filter_threshold=0.0
    )

    # set up for benchmarking boxplot only test
    iou = []
    dice = []
    accuracy = []

    iou_comp = []
    dice_comp = []
    accuracy_comp = []

    # set up parameters for unet augmented testing
    n = 100
    adv_weight = 0.1
    output_dir = '/results/test_can_data/trans_data/boxplot_test'

    unet_aug_model_path = "/results/model/0_seg_model.pt"
    test_selected_indices = get_unfaulty_indices(path=test_trans_image_dir, n=n)
    test_trans_dataset = Subset(test_trans_dataset_full, test_selected_indices)

    iou_aug = []
    dice_aug = []
    accuracy_aug = []

    # test data with augmentation
    metrics_test_aug = results_unet(unet_path=unet_aug_model_path, results_dir=output_dir + "/test_aug",
                                    test_dataset=test_trans_dataset, test_pred_mask_dir=test_trans_pred_mask_aug_dir)
    iou_aug = metrics_test_aug['iou']
    dice_aug = metrics_test_aug['dice']
    accuracy_aug = metrics_test_aug['accuracy']

    # fix saving order in augemented testing
    input_folder = '/results/test_can_data/trans_data/boxplot_test/test_aug'
    output_folder = '/results/test_can_data/trans_data/boxplot_test/test_aug_fixed'

    fix_saving_order(input_folder=input_folder, output_folder=output_folder)
