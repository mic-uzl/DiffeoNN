import os
import sys
import glob
import numpy as np
import torch



def synthmorph(label_maps, batch_size=1, same_subj=False, flip=True):
    """
    Generator for SynthMorph registration.

    Parameters:
        labels_maps: List of pre-loaded ND label maps, each as a NumPy array.
        batch_size: Batch size. Default is 1.
        same_subj: Whether the same label map is returned as the source and target for further
            augmentation. Default is False.
        flip: Whether axes are flipped randomly. Default is True.
    """
    in_shape = label_maps[0].shape
    num_dim = len(in_shape)

    # "True" moved image and warp, that will be ignored by SynthMorph losses.
    void = np.zeros((batch_size, *in_shape, num_dim), dtype='float32')

    rand = np.random.default_rng()
    prop = dict(replace=False, shuffle=False)
    while True:
        ind = rand.integers(len(label_maps), size=2 * batch_size)
        x = [label_maps[i] for i in ind]

        if same_subj:
            x = x[:batch_size] * 2
        x = np.stack(x)[..., None]

        if flip:
            axes = rand.choice(num_dim, size=rand.integers(num_dim + 1), **prop)
            x = np.flip(x, axis=axes + 1)

        src = x[:batch_size, ...]
        trg = x[batch_size:, ...]
        yield [src, trg], [void] * 2


def gencors(
        mov_img,
        fix_img,
        batch_size=1,
        segs=None,
        np_var='vol',
        pad_shape=None,
        resize_factor=1,
        add_feat_axis=True,
        downsample_fac=1,
):
    """
    Generate coordinates from corresponding images to feed into the DNVF

    Parameters:
        mov_img: moved image, which is to align.
        fix_img: fixed image
        segs: Loads corresponding segmentations. Default is None.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """
    assert mov_img.shape == fix_img.shape and 2 <= len(mov_img.shape) <= 5
    xmin, xmax, distance = -1, 1, 2
    epsilon = 1e-7
    if len(mov_img.shape) == 4:
        _, _, H, W = mov_img.shape
        maxDim = max(H, W)
        dH, dW = H / maxDim, W / maxDim
        coords = torch.cartesian_prod(torch.arange(xmin * dH, xmax * dH, distance * downsample_fac / H * dH),
                                      torch.arange(xmin * dW, xmax * dW, distance * downsample_fac / W * dW))
    if len(mov_img.shape) == 2:
        H, W = mov_img.shape
        maxDim = max(H, W)
        dH, dW = H / maxDim, W / maxDim
        coords = torch.cartesian_prod(torch.arange(xmin * dH, xmax * dH, distance * downsample_fac / H * dH),
                                      torch.arange(xmin * dW, xmax * dW, distance * downsample_fac / W * dW))
    elif len(mov_img.shape) == 5:
        _, _, H, W, Z = mov_img.shape
        maxDim = max(H, W, Z)
        dH, dW, dZ = H / maxDim, W / maxDim, Z / maxDim
        coords = torch.cartesian_prod(torch.arange(xmin * dH, xmax * dH, distance * downsample_fac / H * dH),
                                      torch.arange(xmin * dW, xmax * dW, distance * downsample_fac / W * dW),
                                      torch.arange(xmin * dZ, xmax * dZ, distance * downsample_fac / Z * dZ))
    elif len(mov_img.shape) == 3:
        # create sampling grid
        vectors = [torch.arange(0, s/downsample_fac) for s in mov_img.shape]
        for i in range(len(mov_img.shape)):
            vectors[i] = 2 * (vectors[i] / (max(mov_img.shape)/downsample_fac - 1) - 0.5)
        # H, W, Z = mov_img.shape
        # maxDim = max(H, W, Z)
        # dH, dW, dZ = H / maxDim, W / maxDim, Z / maxDim
        coords = torch.cartesian_prod(vectors[0], vectors[1],vectors[2])
    while True:
        yield coords.float()


def mancors(
        mov_img,
        fix_img,
        batch_size=1,
        segs=None,
        np_var='vol',
        pad_shape=None,
        resize_factor=1,
        add_feat_axis=True,
        downsample_fac=1,
):
    """
    Generate coordinates from corresponding images to feed into the DNVF

    Parameters:
        mov_img: moved image, which is to align.
        fix_img: fixed image
        segs: Loads corresponding segmentations. Default is None.
        np_var: Name of the volume variable if loading npz files. Default is 'vol'.
        pad_shape: Zero-pads loaded volumes to a given shape. Default is None.
        resize_factor: Volume resize factor. Default is 1.
        add_feat_axis: Load volume arrays with added feature axis. Default is True.
    """
    assert mov_img.shape == fix_img.shape and 2 <= len(mov_img.shape) <= 5
    xmin, xmax, distance = -1, 1, 2
    if len(mov_img.shape) == 4:
        _, _, H, W = mov_img.shape
        maxDim = max(H, W)
        dH, dW = H / maxDim, W / maxDim
        coords = torch.cartesian_prod(torch.arange(xmin * dH, xmax * dH, distance * downsample_fac / H * dH),
                                      torch.arange(xmin * dW, xmax * dW, distance * downsample_fac / W * dW))
        dist = torch.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2).reshape(-1, 1)
        angle = torch.arctan2(coords[:, 0], coords[:, 1]).reshape(-1, 1)
    if len(mov_img.shape) == 2:
        H, W = mov_img.shape
        maxDim = max(H, W)
        dH, dW = H / maxDim, W / maxDim
        coords = torch.cartesian_prod(torch.arange(xmin * dH, xmax * dH, distance * downsample_fac / H * dH),
                                      torch.arange(xmin * dW, xmax * dW, distance * downsample_fac / W * dW))
        dist = torch.sqrt(coords[:, 1] ** 2 + coords[:, 2] ** 2)
        angle = torch.arctan2(coords[:, 1], coords[:, 2])


    while True:
        yield torch.concatenate([coords, dist, angle], dim=1).float(), torch.from_numpy(
            fix_img).float(), torch.from_numpy(
            mov_img).float()  # torch.concatenate([coords, dist, angle], dim=1).float()
