import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
import time
from PIL import Image
import os

print(torch.cuda.is_available())
print(torch.cuda.device_count())
x = torch.ones(5, device=torch.device("cuda:0"))
from adversarial_loss_train import convexnet
from can_reg import get_trans
from unet import UNet
from my_network_alldata import visualise, metrics_trans



os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
from can_layers import SpatialTransformer
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())

data_folder = "/data/"


#train data
train_image_dir = data_folder + "/empi_pairs/train/images"
train_mask_dir = data_folder + "/empi_pairs/train/masks"
train_pred_mask_dir = data_folder + "/equiv/pred_masks"
train_compare_mask_dir = data_folder + "/equiv/compare_masks"
train_pred_orig_mask_dir = data_folder + "/equiv/pred_orig_masks"
train_pred_orig_image_dir = data_folder + "/equiv/pred_orig_images"
train_pred_mask_aug_dir = data_folder + "/equiv/pred_masks_aug"

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(train_pred_mask_dir, exist_ok=True)
os.makedirs(train_pred_orig_mask_dir, exist_ok=True)
os.makedirs(train_pred_orig_image_dir, exist_ok=True)
os.makedirs(train_pred_mask_aug_dir, exist_ok=True)
os.makedirs(train_compare_mask_dir, exist_ok=True)

# apply canonicalisation and segmentation to transformed squares
def test_equiv1(n=10, output_dir = '/results/test_equiv'):
    # dataset: None, 'test', 'train', 'val'
    unet_model_path = "/results/model/1_seg_model.pt"
    unet_model = UNet(dimensions=2)
    checkpoint = torch.load(unet_model_path, map_location=device)
    unet_model.load_state_dict(checkpoint)
    unet_model.eval()
    unet_model.to(device)
    print(f"Testing network with {n} images (from training dataset)")

    adv_weight = 0.001
    vae_weight = 0.01

    orig_image_dir = train_image_dir
    trans_image_dir = train_image_dir
    pred_orig_mask_dir = train_pred_orig_mask_dir
    pred_trans_mask_dir = train_pred_mask_dir
    trans_mask_dir = train_mask_dir
    compare_mask_dir = train_compare_mask_dir

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
        discriminator.load_state_dict(torch.load("/results/model/adversarial_discriminator_otf.pth",
                                                 map_location=device))  # for transformed squares
        discriminator.eval()

        # get fixed and moving images, warped image, new locations, and backward locations
        time1 = time.time()
        fixed_img, moved_img, warped_img, new_locs, new_locs_backward, neg_flow, inshape, first_loss, final_loss = get_trans(
            fix_path=fix_path, mov_path=mov_path, adv_weight=adv_weight, vae_weight_var=vae_weight)
        time2 = time.time()
        print(f"Time of canonicalisation step: {time2 - time1}")
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

        if i % 1 == 0:
            print(f"Visualizing {fname} ({i + 1}/{n})")
            # visualize the results
            gt_mask = np.array(Image.open(trans_mask_path).convert("L").resize((128, 128))) / 255.0
            gt_mask = (gt_mask >= 0.5).astype(np.float32)
            visualise(fixed_img=fixed_img, moved_img=moved_img, warped_img=warped_img, new_locs=new_locs,
                      new_locs_backward=new_locs_backward, inshape=inshape, M1_np=M1_np, M2_np=square_mask_np,
                      gt_M=gt_mask, index=i + 1, output_dir=output_dir)

    # compute metrics
    metrics = metrics_trans(n=n, mask_dir=compare_mask_dir, pred_mask_dir=pred_trans_mask_dir,
                            compare_mask_dir=compare_mask_dir, results_dir=output_dir)
    print(f"Testing network with {n} images (transformed squares)")
    print(
        f"With Canonicalisation: mean iou: {metrics['mean_iou']}, mean dice: {metrics['mean_dice']}, mean accuracy: {metrics['mean_accuracy']}")
    print(
        f"Without Canonicalisation: mean iou: {metrics['mean_iou_comp']}, mean dice: {metrics['mean_dice_comp']}, mean accuracy: {metrics['mean_accuracy_comp']}")
    # save the metrics to a file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(
            f"With Canonicalisation: mean iou: {metrics['mean_iou']}, mean dice: {metrics['mean_dice']}, mean accuracy: {metrics['mean_accuracy']}\n")
        f.write(
            f"Without Canonicalisation: mean iou: {metrics['mean_iou_comp']}, mean dice: {metrics['mean_dice_comp']}, mean accuracy: {metrics['mean_accuracy_comp']}\n")
        f.write(f"Number of masks: {metrics['num_masks']}\n")
        f.write(f"IOU: {metrics['iou']}\n")
        f.write(f"Dice: {metrics['dice']}\n")
        f.write(f"Accuracy: {metrics['accuracy']}\n")
        f.write(f"IOU without Canonicalisation: {metrics['iou_comp']}\n")
        f.write(f"Dice without Canonicalisation: {metrics['dice_comp']}\n")
        f.write(f"Accuracy without Canonicalisation: {metrics['accuracy_comp']}\n")

    return metrics, loss_list

if __name__ == '__main__':
    test_equiv1(n=100)