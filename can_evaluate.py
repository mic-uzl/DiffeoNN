import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import can_generators as generators
import nibabel as nib
from can_layers import SpatialTransformer


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    smooth = 1e-10
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def max_dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.minimum(torch.sum(y_true_f), torch.sum(y_pred_f))
    smooth = 1e-10
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice, ind_sum = [], []
    for index in range(1, numLabels + 1):
        ind_sum.append(torch.sum(y_true == index).item())
        dice.append(dice_coef(torch.where(abs(y_true - index) < 5e-2, 1, 0),
                              torch.where(abs(y_pred - index) < 5e-2, 1, 0)).item())
    return dice, ind_sum  # taking average


def max_dice_coef_multilabel(y_true, y_pred, numLabels):
    dice = []
    for index in range(0, numLabels + 1):
        dice.append(max_dice_coef(torch.where(abs(y_true - index) < 5e-2, 1, 0),
                                  torch.where(abs(y_pred - index) < 5e-2, 1, 0)).item())
    return dice  # taking average


def evaluation(path_mov, path_fixed, dvf, batch_size, downsample_fac,
               old_deform=None, training=True, numLabels=35):
    mov_img = nib.load(path_mov).get_fdata().astype(float)  # r'OASIS_OAS1_0001_MR1/aligned_norm.nii.gz'
    fix_img = nib.load(path_fixed).get_fdata().astype(float)
    mov_img, fix_img = torch.from_numpy(mov_img).float(), torch.from_numpy(fix_img).float()
    '''down = torch.nn.Upsample(scale_factor=1, mode='trilinear')
    # Normalise Images
    mov_img, fix_img = torch.from_numpy(mov_img), torch.from_numpy(fix_img)
    mov_img, fix_img = down(mov_img.reshape((1, 1) + mov_img.shape)), down(
        fix_img.reshape((1, 1) + fix_img.shape))'''
    generator = generators.gencors(
        mov_img.numpy(), fix_img.numpy(), batch_size=batch_size, add_feat_axis=False, downsample_fac=downsample_fac)
    inputs = next(generator)
    y_mov = mov_img.to('cuda')
    y_fix = fix_img.to('cuda')
    Warper = SpatialTransformer(y_mov.shape)
    if training == True:
        # inputs, y_fix, y_mov, dvf = inputs.to('cuda'), y_fix.to('cuda'), y_mov.to('cuda'), dvf.to('cuda')
        warped_img, _ = Warper(y_mov, dvf, mode='nearest')
    else:
        warped_img, _ = Warper(y_mov, dvf, mode='nearest')
    pre_dice, vol_fac = dice_coef_multilabel(y_fix, y_mov, numLabels)
    post_dice, vol_fac = dice_coef_multilabel(y_fix, warped_img, numLabels)
    return pre_dice, post_dice, np.array(vol_fac) / np.sum(vol_fac)


def evaluation(path_mov, path_fixed, dvf, batch_size, downsample_fac,
               old_deform=None, training=True, numLabels=62):
    mov_img = nib.load(path_mov).get_fdata().astype(float)[16:-16,16:-16,16:-16] # r'OASIS_OAS1_0001_MR1/aligned_norm.nii.gz'
    fix_img = nib.load(path_fixed).get_fdata().astype(float)[16:-16,16:-16,16:-16]
    mov_img, fix_img = torch.from_numpy(mov_img).float(), torch.from_numpy(fix_img).float()
    '''down = torch.nn.Upsample(scale_factor=1, mode='trilinear')
    # Normalise Images
    mov_img, fix_img = torch.from_numpy(mov_img), torch.from_numpy(fix_img)
    mov_img, fix_img = down(mov_img.reshape((1, 1) + mov_img.shape)), down(
        fix_img.reshape((1, 1) + fix_img.shape))'''
    generator = generators.gencors(
        mov_img.numpy(), fix_img.numpy(), batch_size=batch_size, add_feat_axis=False, downsample_fac=downsample_fac)
    inputs = next(generator)
    y_mov = mov_img.to('cuda')
    y_fix = fix_img.to('cuda')
    Warper = SpatialTransformer(y_mov.shape)
    if training == True:
        # inputs, y_fix, y_mov, dvf = inputs.to('cuda'), y_fix.to('cuda'), y_mov.to('cuda'), dvf.to('cuda')
        warped_img, _ = Warper(y_mov, dvf, mode='nearest')
    else:
        warped_img, _ = Warper(y_mov, dvf, mode='nearest')
    pre_dice, vol_fac = dice_coef_multilabel(y_fix, y_mov, numLabels)
    post_dice, vol_fac = dice_coef_multilabel(y_fix, warped_img, numLabels)
    return pre_dice, post_dice, np.array(vol_fac) / np.sum(vol_fac)


def evaluation2(fix_seg, moved_seg, dvf, numLabels=62):
    y_mov = torch.from_numpy(moved_seg).to('cuda')
    y_fix = torch.from_numpy(fix_seg).to('cuda')
    pre_dice, post_dice = [], []
    Warper = SpatialTransformer(y_mov.shape[1:])
    for i in range(numLabels):
        warped_img, _ = Warper(y_mov[i].double(), dvf, mode='nearest')
        pre_dice.append(dice_coef(y_fix[i], y_mov[i]).item())
        post_dice.append(dice_coef(y_fix[i], warped_img).item())
    return pre_dice, post_dice
