import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from scipy.interpolate import RBFInterpolator
from matplotlib.collections import LineCollection
from PIL import Image

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def plot_grid(x, y, ax=None, **kwargs):
    ax = ax or plt.gca
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow, padding='zeros',mode='bilinear'):
        # new locations
        # need to normalize grid values to [-1, 1] for resampler
        shape = self.grid.shape[2:]
        newgrid = torch.zeros_like(self.grid)
        newflow = torch.zeros_like(flow)
        newflow[:] = flow[:]
        for i in range(len(shape)):
            newgrid[:, i, ...] = 2 * (self.grid[:, i, ...] / (shape[i] - 1) - 0.5)
            #newflow[:, i, ...] = newflow[:, i, ...] * 2 / (shape[i] - 1)  # Normalize DVF to [-1, 1]
        new_locs = newgrid + newflow

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
            return nnf.grid_sample(src.reshape(1, -1, shape[0], shape[1]), new_locs, align_corners=True, mode=mode,
                                   padding_mode=padding)



def determinant_jacobian(dvf):
    '''
    Determine the Jacobian determinant of a dense vector field (dvf).
    :param dvf: Dense vector field of shape (B, C, H, W) or (B, C, D, H, W) (B,C=1,2)
    :return: Jacobian determinant of the dvf as tensor (H,W))
    '''
    dy = (dvf[:,:,2:, 1:-1]- dvf[:,:, :-2, 1:-1]) / 2.0
    dx = (dvf[:,:,1:-1, 2:] - dvf[:,:,1:-1, :-2]) / 2.0
    #print(f"dy {dy}")
    #print(f"dvf[:,:,2:, 1:-1] {dvf[:,:,2:, 1:-1]}", f"dvf[:,:, :-2, 1:-1] {dvf[:,:, :-2, 1:-1]}")
    #print(f"dx {dx}")
    #print(f"dvf[:,:,1:-1, 2:] {dvf[:,:,1:-1, 2:]}", f"dvf[:,:,1:-1, :-2] {dvf[:,:,1:-1, :-2]}")

    det = (1 + dx[:,0]) * (1 + dy[:,1]) - dx[:,1] * dy[:,0]
    #det = dx[:,0] * dy[:,1] - dx[:,1] * dy[:,0]
    return det

def random_2d_warp(rotate=True, num_points=12, max_disp=0.12):
    pre_scale=0.6 #0.8
    coordinates = np.linspace(-0.5, 0.5, int(np.sqrt(num_points)))
    xx, yy = np.meshgrid(coordinates, coordinates, indexing='ij')
    orig_points = np.stack([xx.flatten(), yy.flatten()], axis=1)

    # Scale everything inward first
    orig_points = orig_points * pre_scale

    if rotate:
        theta = np.random.uniform(-np.pi/6, np.pi/6)
        #scale_mat = np.array([[1.2, -0.5], [0.0, 0.6]]) # works with this
        scale_mat = np.array([[1.5, -1.0], [0.0, 1.2]])
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]])
        rot_mat = 0.9 * rotation
        #translation = 0.05 * (np.random.rand(2) - 0.5)

        # Apply affine
        #deformed_points = 0.2 * (orig_points @ rot_mat @ scale_mat - 0.5) + translation
        deformed_points = orig_points @ rot_mat #+ translation
    else:
        deformed_points = orig_points.copy()

    # Optional: small local displacement
    displacements = max_disp * 2 * (np.random.rand(*deformed_points.shape) - 0.5)
    deformed_points += displacements

    # Clip to frame
    deformed_points = np.clip(deformed_points, -0.5, 0.5)

    rbf = RBFInterpolator(orig_points, deformed_points, kernel='thin_plate_spline')
    return rbf

def get_2d_dvf(rbf, size):
    """
    Get the dense vector field from the RBF model.
    :param rbf: RBF model
    :param size: Size of the output vector field
    :return: Dense vector field
    """
    # create a grid of points in the range [-0.5, 0.5]
    x = np.linspace(-0.5, 0.5, size[0])
    y = np.linspace(-0.5, 0.5, size[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)

    # compute the deformed points
    deformed_points = rbf(grid_points)# - grid_points

    # reshape to match the output size
    dvf = (deformed_points - grid_points).reshape(size[0], size[1], 2)
    return torch.tensor(dvf).permute(2, 0, 1).unsqueeze(0).float()  # (1, 2, H, W)

def rand_trans2d(fix_img, num_points = 12, max_disp=0.12, rotate=True):
    """
    Apply a random 2D transformation to an image using radial basis functions (RBF).
    :param fix_img: Fixed image to be transformed
    :param num_points: Number of control points for the RBF
    :param max_disp: Maximum displacement for the control points
    :param rotate: Whether to apply random rotation and translation
    :return: Warped image and dense vector field (DVF)
    """
    fix_img_tensor = torch.tensor(fix_img).unsqueeze(0).unsqueeze(0)
    #while True:
    #rbf1 = random_2d_warp(rotate=rotate, num_points=num_points, max_disp=max_disp)
    #rbf2 = random_2d_warp(rotate=rotate, num_points=num_points, max_disp=max_disp)
    #rbf = lambda x: rbf2(rbf1(x))
    rbf = random_2d_warp(rotate=rotate, num_points=num_points, max_disp=max_disp)
    dvf = get_2d_dvf(rbf, size= fix_img.shape)

    #test jacobian determinant of the dvf (diffeomorphic: det(J)>0):
    det = determinant_jacobian(dvf)
    #print(f"Jacobian determinant min: {det.min().item()}, max: {det.max().item()}")
    if (det< 0).any():
        print("Warning: Jacobian determinant is negative. The transformation is not diffeomorphic.")
        #break

    transformer = SpatialTransformer(size= fix_img.shape, mode='bilinear')
    warped_img = transformer(fix_img_tensor.float(), dvf.float(), padding='zeros', mode='bilinear')

    return warped_img.squeeze(), dvf.squeeze()

def rand_mask_trans2d(fix_img, mask_img, num_points = 12, max_disp=0.12, rotate=True):
    """
    Apply a random 2D transformation to an image using radial basis functions (RBF).
    :param fix_img: Fixed image to be transformed
    :param num_points: Number of control points for the RBF
    :param max_disp: Maximum displacement for the control points
    :param rotate: Whether to apply random rotation and translation
    :return: Warped image and dense vector field (DVF)
    """
    fix_img_tensor = torch.tensor(fix_img).unsqueeze(0).unsqueeze(0)
    mask_img_tensor = torch.tensor(mask_img).unsqueeze(0).unsqueeze(0)
    #while True:
    rbf = random_2d_warp(rotate=rotate, num_points=num_points, max_disp=max_disp)
    #rbf1 = random_2d_warp(rotate=rotate, num_points=num_points, max_disp=max_disp)
    #rbf2 = random_2d_warp(rotate=rotate, num_points=num_points, max_disp=max_disp)
    #rbf = lambda x: rbf2(rbf1(x))
    dvf = get_2d_dvf(rbf, size= fix_img.shape)

    #test jacobian determinant of the dvf (diffeomorphic: det(J)>0):
    det = determinant_jacobian(dvf)
    #print(f"Jacobian determinant min: {det.min().item()}, max: {det.max().item()}")
    if (det< 0).any():
        print("Warning: Jacobian determinant is negative. The transformation is not diffeomorphic.")
        #break

    transformer = SpatialTransformer(size= fix_img.shape, mode='bilinear')
    warped_img = transformer(fix_img_tensor.float(), dvf.float(), padding='zeros', mode='bilinear')
    warped_mask = transformer(mask_img_tensor.float(), dvf.float(), padding='zeros', mode='bilinear')
    warped_mask = torch.round(warped_mask) # changed from nearest to bilinear and then added round
    return warped_img.squeeze(), warped_mask.squeeze(), dvf.squeeze()


def main2D():
    fix_path = r'MA/Johannes/NeuralVelo/Hands/t1_triangles.jpg'
    t_fix_img = Image.open(fix_path).resize((128, 128))
    t_fix_img = t_fix_img.convert("L")
    fix_img = np.array(t_fix_img, dtype=np.float32)/255.0

    fix_img_tensor = torch.tensor(fix_img).unsqueeze(0).unsqueeze(0)
    #while True:
    rbf = random_2d_warp(rotate=True, num_points=5, max_disp=0.3)
    dvf = get_2d_dvf(rbf, size= fix_img.shape)

    #test jacobian determinant of the dvf (diffeomorphic: det(J)>0):
    det = determinant_jacobian(dvf)
    #print(f"Jacobian determinant min: {det.min().item()}, max: {det.max().item()}")
    if (det< 0).any():
        print("Warning: Jacobian determinant is negative. The transformation is not diffeomorphic.")
        #break

    transformer = SpatialTransformer(size= fix_img.shape, mode='bilinear')
    warped_img = transformer(fix_img_tensor.float(), dvf.float(), padding='zeros', mode='bilinear')


    plt.subplot(1, 2, 1)
    plt.imshow(fix_img, cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(warped_img.squeeze().detach().numpy(), cmap='gray')
    plt.title("Warped")
    plt.show()
    plt.close()

    # Save the results
    torch.save(warped_img, 'MA/Johannes/NeuralVelo/app/RandTrans/warped_img.pt')
    torch.save(dvf, 'MA/Johannes/NeuralVelo/app/RandTrans/dvf.pt')
    torch.save(fix_img_tensor, 'MA/Johannes/NeuralVelo/app/RandTrans/fix_img.pt')

    plt.imsave('MA/Johannes/NeuralVelo/app/RandTrans/warped_img.png', warped_img.squeeze().detach().numpy())
    plt.imsave('MA/Johannes/NeuralVelo/app/RandTrans/fix_img.png', fix_img_tensor.squeeze().detach().numpy())
    plt.imsave('MA/Johannes/NeuralVelo/app/RandTrans/difference.png', fix_img_tensor.squeeze().detach().numpy()-warped_img.squeeze().detach().numpy())
    #plt.imsave('MA/Johannes/NeuralVelo/app/RandTrans/dvf.png', dvf.squeeze().detach().numpy())

    plt.imshow(det[0].cpu().numpy(), cmap='seismic')
    plt.colorbar()
    plt.title("Jacobian Determinant")
    plt.savefig('MA/Johannes/NeuralVelo/app/RandTrans/jacobian_determinant.png')
    plt.tight_layout()
    plt.show()

"""
def test_determinant_jacobian():
    # Test the determinant_jacobian function
    dvf = torch.tensor([[[[1.0, 2.5], [3.0, 4.0]]]])  # Simple DVF
    print(f"DVF shape: {dvf.shape}")
    det = determinant_jacobian(dvf)
    print(f"Jacobian determinant: {det.item()}")

"""  

if __name__ == '__main__':
    main2D()
    #test_determinant_jacobian()
