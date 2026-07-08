import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import gradcheck
torch.set_default_dtype(torch.float32)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size):
        super().__init__()

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).to('cpu') #cuda
        self.shape = grid.shape[2:]
        for i in range(len(self.shape)):
            # need to normalize grid values to [-1, 1] for resampler
            grid[:, i, ...] = 2 * (grid[:, i, ...] / (self.shape[i] - 1) - 0.5)
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow, padding='border', mode='bilinear'): # TODO: I changed padding from 'zeros' to 'zeros'
        new_loc = self.grid + flow
        maxdir = max(self.shape)
        for i in range(len(self.shape)):
            # need to normalize grid values to [-1, 1] for resampler
            new_loc[:, i, ...] = self.grid[:, i, ...] + flow[:, i, ...] * (maxdir - 1) / (self.shape[i] - 1)
        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(self.shape) == 2:
            new_locs = new_loc.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
            return nnf.grid_sample(src.reshape(1, -1, self.shape[0], self.shape[1]), new_locs, align_corners=True,
                                   mode=mode,
                                   padding_mode=padding), new_loc
            # grid_sample(src.reshape(1,-1,shape[0],shape[1]), new_locs)
        elif len(self.shape) == 3:
            new_locs = new_loc.permute(0, 2, 3, 4, 1)
            new_locs = torch.flip(new_locs, dims=(4,))
            return nnf.grid_sample(src.reshape(1, -1, self.shape[0], self.shape[1], self.shape[2]), new_locs,
                                   align_corners=True,
                                   mode=mode,
                                   padding_mode=padding), new_loc
        else:
            return None

def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    '''

    from https://proceedings.mlr.press/v172/wolterink22a.html

    '''

    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
            input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
            + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
            + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
            + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
            + input_array[x1, y0, z1] * x * (1 - y) * z
            + input_array[x0, y1, z1] * (1 - x) * y * z
            + input_array[x1, y1, z0] * x * y * (1 - z)
            + input_array[x1, y1, z1] * x * y * z
    )

    return output


def fast_trilinear_color_interpolation(input_array, x_indices, y_indices, z_indices):
    '''

    from https://proceedings.mlr.press/v172/wolterink22a.html

    '''

    n_dim = input_array.shape[0]
    x_indices = (x_indices + 1) * (input_array.shape[1] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[2] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[3] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[1] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[2] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[3] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[1] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[2] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[3] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    n = len(x)
    output = torch.zeros(input_array.shape).to(input_array.device)

    for i in range(n_dim):
        temp = (input_array[i, x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
                + input_array[i, x1, y0, z0] * x * (1 - y) * (1 - z)
                + input_array[i, x0, y1, z0] * (1 - x) * y * (1 - z)
                + input_array[i, x0, y0, z1] * (1 - x) * (1 - y) * z
                + input_array[i, x1, y0, z1] * x * (1 - y) * z
                + input_array[i, x0, y1, z1] * (1 - x) * y * z
                + input_array[i, x1, y1, z0] * x * y * (1 - z)
                + input_array[i, x1, y1, z1] * x * y * z)

        output[i, :] = temp

    return output


def SO3log(input):
    trR = input[..., 0, 0] + input[..., 1, 1] + input[..., 2, 2]
    cos_theta = torch.clip(0.5 * (trR - 1), max=1, min=-1)
    sin_theta = 0.5 * torch.sqrt(torch.clip((3 - trR) * (1 + trR), min=0))
    theta = torch.arctan2(sin_theta, cos_theta)
    R_minus_R_T_vee = torch.stack(
        [
            input[..., 2, 1] - input[..., 1, 2],
            input[..., 0, 2] - input[..., 2, 0],
            input[..., 1, 0] - input[..., 0, 1],
        ],
        dim=-1,
    )

    c = 0.5 * (1 + theta * theta / 6 + 7 / 360 * (theta ** 4)).unsqueeze(1)
    # it diverges around theta=pi
    R_diag = torch.einsum("...ii->...i", input)
    R_diag = torch.clamp(R_diag, min=-1.0, max=1.0)
    eeT_diag = 0.5 * (R_diag + 1.0)
    signs = R_minus_R_T_vee.sign()
    signs[signs == 0.0] = 1.0
    small_trace = eeT_diag.sqrt() * signs

    v = torch.where((theta - torch.pi).abs() < 1e-4, small_trace,
                    theta.unsqueeze(1) / (2 * sin_theta.unsqueeze(1)) * R_minus_R_T_vee)

    return c * R_minus_R_T_vee


class SO3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        ctx.save_for_backward(input)
        trR = input[..., 0, 0] + input[..., 1, 1] + input[..., 2, 2]
        cos_theta = torch.clip(0.5 * (trR - 1), max=1, min=-1)
        sin_theta = 0.5 * torch.sqrt(torch.clip((3 - trR) * (1 + trR) + 1e-5, min=0))
        theta = torch.arctan2(sin_theta, cos_theta)
        R_minus_R_T_vee = torch.stack(
            [
                input[..., 2, 1] - input[..., 1, 2],
                input[..., 0, 2] - input[..., 2, 0],
                input[..., 1, 0] - input[..., 0, 1],
            ],
            dim=-1,
        )

        c = 0.5 * (1 + theta * theta / 6 + 7 / 360 * (theta ** 4)).unsqueeze(1)
        # it diverges around theta=pi
        R_diag = torch.einsum("...ii->...i", input)
        R_diag = torch.clamp(R_diag, min=-1.0, max=1.0)
        eeT_diag = 0.5 * (R_diag + 1.0)
        signs = R_minus_R_T_vee.sign()
        signs[signs == 0.0] = 1.0
        small_trace = eeT_diag.sqrt() * signs
        v = torch.where((theta.unsqueeze(1) - torch.pi).abs() < 1e-4, small_trace,
                        theta.unsqueeze(1) / (2 * sin_theta.unsqueeze(1)) * R_minus_R_T_vee)

        return torch.where(abs(3 - trR).unsqueeze(1) < 1e-6, c * R_minus_R_T_vee, v)

    @staticmethod
    def backward(ctx, grad_output, float=0.5):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors

        trR = input[..., 0, 0] + input[..., 1, 1] + input[..., 2, 2]
        cos_theta = torch.clip(0.5 * (trR - 1), max=1, min=-1)
        R_minus_R_T_vee = torch.stack(
            [
                input[..., 2, 1] - input[..., 1, 2],
                input[..., 0, 2] - input[..., 2, 0],
                input[..., 1, 0] - input[..., 0, 1],
            ],
            dim=-1,
        )
        theta = torch.arccos(cos_theta)
        sin_theta = torch.sqrt(1 - cos_theta * cos_theta)
        c = (theta * cos_theta - sin_theta) / (4 * sin_theta ** 3 + 1e-6)
        a = c.unsqueeze(1) * R_minus_R_T_vee
        b = theta / (2 * sin_theta + 1e-6)
        near_id = torch.stack(
            [torch.zeros_like(input).flatten(start_dim=1), torch.zeros_like(input).flatten(start_dim=1),
             torch.zeros_like(input).flatten(start_dim=1)], dim=-1)  # .to('cuda')
        grad = torch.zeros_like(near_id)
        neg_inds1, neg_inds2 = (5, 6, 1), (0, 1, 2)
        pos_inds1, pos_inds2 = (7, 2, 3), (0, 1, 2)
        near_id[..., neg_inds1, neg_inds2] = -0.5
        near_id[..., pos_inds1, pos_inds2] = 0.5

        grad[..., neg_inds1, neg_inds2] = -b.unsqueeze(1)
        grad[..., pos_inds1, pos_inds2] = b.unsqueeze(1)
        inds = (0, 4, 8)
        grad[..., inds, :] = a.unsqueeze(1)
        grads = torch.where(cos_theta.unsqueeze(1).unsqueeze(2) > 1 - 1e-6, near_id, grad)
        return (torch.einsum("ijk,ik->ij", grads, grad_output)).reshape(-1, 3, 3)





class MatrixTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear', device='cpu'): # cuda
        super().__init__()
        self.mode = 'bilinear'

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids).float()
        self.shape = grid.shape[1:]
        for i in range(len(self.shape)):
            grid[i, ...] = 2 * (grid[i, ...] / (self.shape[i] - 1) - 0.5) * (self.shape[i] - 1) / (max(self.shape) - 1)
        grid = torch.cat((grid, torch.ones_like(grids[0].unsqueeze(0))))
        grid = torch.unsqueeze(grid, 0)
        self.grid = grid.type(torch.FloatTensor).to(device)

    def forward(self, src, flow, padding='zeros'):
        # new locations
        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(self.shape) == 2:
            new_loc = torch.einsum("ijklm,iklm->ijlm", flow, self.grid)
            new_locs = torch.zeros_like(new_loc)
            for i in range(len(self.shape)):
                new_locs[:, i, ...] = new_loc[:, i, ...] / ((self.shape[i] - 1) / (max(self.shape) - 1))
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
            return nnf.grid_sample(src.reshape(1, -1, self.shape[0], self.shape[1]),
                                   new_locs, align_corners=True, mode=self.mode, padding_mode=padding)
        elif len(self.shape) == 3:
            new_loc = torch.einsum("ilmnjk,iklmn->ijlmn", flow, self.grid)
            new_locs = torch.zeros_like(new_loc)
            for i in range(len(self.shape)):
                new_locs[:, i, ...] = new_loc[:, i, ...] / ((self.shape[i] - 1) / (max(self.shape) - 1))
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
            return nnf.grid_sample(
                src.reshape(1, self.shape[0], self.shape[1], self.shape[2], -1).permute(0, 4, 1, 2, 3),
                new_locs, align_corners=True, mode=self.mode, padding_mode=padding)
        else:
            return None


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            add, _ = self.transformer(vec, vec)
            vec = vec + add
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field.
    """

    def __init__(self, vel_resize, ndims, size):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.size = size
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, size=self.size, align_corners=True, mode=self.mode)
            # x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            # x = self.factor * x
            x = nnf.interpolate(x, size=self.size, align_corners=True, mode=self.mode)

        # don't do anything if resize is 1
        return x


class MatrixInt(nn.Module):
    """
    Integrates a manifold valued field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps, device='cpu'): #cuda
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.inshape = inshape
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = MatrixTransformer(inshape)
        dims = len(inshape)
        if len(inshape) == 2:
            V = torch.zeros(1, inshape[0], inshape[1], dims, dims).to(device)
            self.V = V
            A = torch.zeros(1, inshape[0], inshape[1], dims, dims).to(device)
            self.A = A
        if dims == 3:
            I = torch.zeros(1, 1, 1, 1, dims, dims)
            I[0, ...] = torch.eye(3)
            self.I = I.to(device)
            A = torch.zeros(1, inshape[0], inshape[1], inshape[2], dims, dims)
            self.A = A.to(device)
            self.i, self.j = torch.triu_indices(len(inshape), len(inshape), offset=1)
            self.MatExp = torch.zeros(1, inshape[0], inshape[1], inshape[2], dims + 1, dims + 1).to(device)

    def Matexp(self, input):
        # size = input.size()
        if False:
            log_R_vee = input[:, :3].permute(0, 2, 3, 4, 1)
            log_t_vee = input[:, 3:].permute(0, 2, 3, 4, 1)
            MatExp = 0 * self.MatExp
            MatExp[..., 0, 1] = -log_R_vee[..., 2]
            MatExp[..., 0, 2] = log_R_vee[..., 1]
            MatExp[..., 1, 2] = -log_R_vee[..., 0]
            MatExp[..., 1, 0] = log_R_vee[..., 2]
            MatExp[..., 2, 0] = -log_R_vee[..., 1]
            MatExp[..., 2, 1] = log_R_vee[..., 0]
            MatExp[..., :3, 3] = log_t_vee
            return torch.matrix_exp(MatExp)
        elif True:
            rot_dim = 3 * (3 - 1) // 2  # rotational degrees of freedom
            A = 0 * self.A
            norm = torch.linalg.vector_norm(input[:, :rot_dim], dim=1)  # + 1e-7
            A[..., self.i, self.j] = input[:, :rot_dim, ...].permute(0, 2, 3, 4, 1)
            A = (A - A.transpose(4, 5))  # create antisymmetric matrix
            MatExp = 0 * self.MatExp
            A_sq = torch.linalg.matrix_power(A, 2)
            norm_sq = norm * norm
            a = torch.where(norm < 0.001, 1 - norm_sq / 6 * (1 - norm_sq / 20 * (1 - norm_sq / 42)),
                            torch.sin(norm) / norm)
            b = torch.where(norm < 0.001, (1 - norm_sq / 12 * (1 - norm_sq / 30 * (1 - norm_sq / 56))) / 2,
                            (1 - torch.cos(norm)) / norm_sq)
            c = torch.where(norm < 0.001, (1 - norm_sq / 20 * (1 - norm_sq / 42 * (1 - norm_sq / 72))) / 6,
                            (norm - torch.sin(norm)) / (norm_sq * norm))  # Taylorentwicklung'''

            a, b, c = a.unsqueeze(4).unsqueeze(5), b.unsqueeze(4).unsqueeze(5), c.unsqueeze(4).unsqueeze(5)
            rot = self.I + a * A + b * A_sq
            V = self.I + b * A + c * A_sq
            MatExp[..., :3, :3] = rot
            MatExp[..., :3, 3] = torch.einsum("ijkmno,iojkm->ijkmn", V, input[:, rot_dim:])
            MatExp[..., 3, 3] = 1
            # mult = torch.exp(input[:, -1]).unsqueeze(1).permute(0, 2, 3, 4, 1).unsqueeze(5)
            return MatExp  # rodriguez formular
        else:
            return None

    def forward(self, vec, device='cuda'):
        if len(self.inshape) == 3:
            vec = vec * self.scale
            mat_field = self.Matexp(vec)
            for _ in range(self.nsteps):
                interp = self.transformer(mat_field, mat_field).reshape(1, 4, 4, self.inshape[0], self.inshape[1],
                                                                        self.inshape[2])
                mat_field = torch.einsum("ijkmno,imnokl->imnojl", interp, mat_field)
        else:
            vec = vec * self.scale
            mat_field = self.Matexp(vec).permute(0, 3, 4, 1, 2)
            for _ in range(self.nsteps):
                interp = self.transformer(mat_field, mat_field)
                reshaped_interp = interp.reshape(1, 3, 3, self.inshape[0], self.inshape[1])
                mat_field = torch.einsum("ijkmn,iklmn->ijlmn", reshaped_interp, mat_field)
        return mat_field


class DriftTransformer(nn.Module):
    """
    Lie Group Transformer
    """

    def __init__(self, size, ndims=3, device='cpu'): # cuda
        super().__init__()
        self.mode = 'bilinear'
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids).float()
        self.shape = grid.shape[1:]
        for i in range(len(self.shape)):
            grid[i, ...] = 2 * (grid[i, ...] / (self.shape[i] - 1) - 0.5) * (self.shape[i] - 1) / (max(self.shape) - 1)
        grid = torch.cat((grid, torch.ones_like(grids[0].unsqueeze(0))))
        grid = torch.unsqueeze(grid, 0)
        self.grid = grid.type(torch.FloatTensor).to(device)
        if ndims == 3:
            # I = torch.zeros(1, 1, 1, 1, ndims, ndims)
            # I[0, ...] = torch.eye(3)
            # self.I = I.to(device)
            # A = torch.zeros(1, size[0], size[1], size[2], ndims, ndims)
            # self.A = A.to(device)
            # self.i, self.j = torch.triu_indices(len(size), len(size), offset=1)
            self.MatExp = torch.zeros(1, size[0], size[1], size[2], ndims + 1, ndims + 1).to(device)
            self.SO3log = SO3.apply
        elif ndims == 2:
            V = torch.zeros(1, size[0], size[1], ndims, ndims).to(device)
            self.V = V
            A = torch.zeros(1, size[0], size[1], ndims, ndims)
            self.A = A.to(device)
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        # self.register_buffer('grid', self.grid, persistent=False)
        self.grid = grid.to(device)

    def Matexp(self, input):
        shape = input.shape[2:]
        if True:
            input = input.permute(0, 2, 3, 4, 1)
            log_R_vee = input[..., :3]
            log_t_vee = input[..., 3:]
            MatExp = 0 * self.MatExp
            MatExp[..., 0, 1] = -log_R_vee[..., 2]
            MatExp[..., 0, 2] = log_R_vee[..., 1]
            MatExp[..., 1, 2] = -log_R_vee[..., 0]
            MatExp[..., 1, 0] = log_R_vee[..., 2]
            MatExp[..., 2, 0] = -log_R_vee[..., 1]
            MatExp[..., 2, 1] = log_R_vee[..., 0]
            MatExp[..., :3, 3] = log_t_vee
            #MatExp[..., 3, 3] 'TODO???
            return torch.matrix_exp(MatExp)
        else: #elif True:
            input = input.permute(0, 2, 3, 4, 1)
            log_R_vee = input[..., :3]
            log_t_vee = input[..., 3:]
            A = 0 * self.A
            A[..., 0, 1] = -log_R_vee[..., 2]
            A[..., 0, 2] = log_R_vee[..., 1]
            A[..., 1, 2] = -log_R_vee[..., 0]
            A[..., 1, 0] = +log_R_vee[..., 2]
            A[..., 2, 0] = -log_R_vee[..., 1]
            A[..., 2, 1] = +log_R_vee[..., 0]
            rot_dim = dim * (dim - 1) // 2  # rotational degrees of freedom
            norm = torch.linalg.vector_norm(log_R_vee, dim=-1)  # + 1e-7
            MatExp = 0 * self.MatExp
            A_sq = torch.linalg.matrix_power(A, 2)
            norm_sq = norm * norm
            a = torch.where(norm < 1e-7, 1 - norm_sq / 6 * (1 - norm_sq / 20 * (1 - norm_sq / 42)),
                            torch.sin(norm) / norm)
            b = torch.where(norm < 1e-7, (1 - norm_sq / 12 * (1 - norm_sq / 30 * (1 - norm_sq / 56))) / 2,
                            (1 - torch.cos(norm)) / norm_sq)
            c = torch.where(norm < 1e-7, (1 - norm_sq / 20 * (1 - norm_sq / 42 * (1 - norm_sq / 72))) / 6,
                            (norm - torch.sin(norm)) / (norm_sq * norm))  # Taylorentwicklung'''
            a, b, c = a.unsqueeze(4).unsqueeze(5), b.unsqueeze(4).unsqueeze(5), c.unsqueeze(4).unsqueeze(5)
            rot = self.I + a * A + b * A_sq
            V = self.I + b * A + c * A_sq
            MatExp[..., :3, :3] = rot
            MatExp[..., :3, 3] = torch.einsum("imnojk,imnok->imnoj", V, log_t_vee)
            MatExp[..., 3, 3] = 1
            # mult = torch.exp(input[:, -1]).unsqueeze(1).permute(0, 2, 3, 4, 1).unsqueeze(5)
            return MatExp  # rodriguez formular

    def forward(self, src, flow, device, padding='zeros', mode='bilinear'):
        # new locations

        shape = self.grid.shape[2:]
        Matrix = self.Matexp(flow).reshape(1, shape[0], shape[1], shape[2], 4, 4)
        if len(shape) == 2:
            new_loc = torch.einsum("ilmjk,iklm->ijlm", Matrix,
                                   self.grid)  # batchwise matrix-vector product
            new_locs = torch.zeros_like(new_loc)
            for i in range(len(self.shape)):
                new_locs[:, i, ...] = new_loc[:, i, ...] / ((self.shape[i] - 1) / (max(self.shape) - 1))
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
            return nnf.grid_sample(src.reshape(1, -1, shape[0], shape[1]), new_locs, align_corners=True, mode=mode,
                                   padding_mode=padding), new_loc[:, :-1], self.grid[:, :-1]
        elif len(self.shape) == 3:
            new_loc = torch.einsum("ilmnjk,iklmn->ijlmn", Matrix[..., :-1, :], self.grid)
            new_locs = torch.zeros_like(new_loc)
            for i in range(len(self.shape)):
                new_locs[:, i, ...] = new_loc[:, i, ...] / (self.shape[i] - 1) * (max(shape) - 1)
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = torch.flip(new_locs, dims=(4,))
            return nnf.grid_sample(
                src.reshape(1, -1, self.shape[0], self.shape[1], self.shape[2]),
                new_locs, align_corners=True, mode=self.mode, padding_mode=padding), new_loc, self.grid[:, :-1]
        else:
            return None


class LietoVec(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.type(torch.FloatTensor)
        self.shape = grid.shape[1:]
        for i in range(len(self.shape)):
            grid[i, ...] = 2 * (grid[i, ...] / (self.shape[i] - 1) - 0.5) * (self.shape[i] - 1) / (max(self.shape) - 1)
        grid = torch.cat((grid, torch.ones_like(grids[0]).unsqueeze(0)))
        grid = torch.unsqueeze(grid, 0).to('cuda')
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, flow, device):
        # new locations
        if len(self.shape) == 3:
            new_loc = torch.einsum("ilmnjk,iklmn->ijlmn", flow, self.grid)
        else:
            new_loc = torch.einsum("ijklm,iklm->ijlm", flow[:, :, :], self.grid)
        return new_loc - self.grid[:, :-1, ...]


class LietoVecfield(nn.Module):
    def __init__(self, size, ndims=3):
        super().__init__()
        self.dim = ndims
        self.rot_dim = ndims * (ndims - 1) // 2
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        # need to normalize grid values to [-1, 1] for resampler
        shape = grid.shape[2:]
        for i in range(len(shape)):
            grid[:, i, ...] = 2 * (grid[:, i, ...] / (shape[i] - 1) - 0.5)
        self.register_buffer('grid', grid)

    def rotation(self, input, dim, device):
        size = input.size()
        if len(size) == 4:
            A = torch.zeros((size[0], dim, dim, size[2], size[3])).to(device)
            A[:, 0, 0, ...] = torch.cos(input[:, 0, ...])
            A[:, 0, 1, ...] = -torch.sin(input[:, 0, ...])
            A[:, 1, 0, ...] = torch.sin(input[:, 0, ...])
            A[:, 1, 1, ...] = torch.cos(input[:, 0, ...])
            return A
        else:
            rot_dim = dim * (dim - 1) // 2  # rotational degrees of freedom
            A = torch.zeros((size[0], dim, dim, size[2], size[3], size[4])).to(device)
            i, j = torch.triu_indices(dim, dim, offset=1)
            A[:, i, j, ...] = input[:, :rot_dim, ...]
            A = A - A.transpose(1, 2)
            return A

    def forward(self, flow, device):
        # new locations
        # need to normalize grid values to [-1, 1] for resampler
        rotMat = self.rotation(flow, self.dim, device)
        rot_dim = self.dim * (self.dim - 1) // 2
        if self.dim == 2:
            newflow = torch.einsum("ijklm,iklm->ijlm", rotMat, self.grid) + flow[:, rot_dim:-1, ...] + flow[:, -1,
                                                                                                       ...] * self.grid
            # flow[:, -1, ...] * self.grid
        if self.dim == 3:
            newflow = torch.einsum("ijklmn,iklmn->ijlmn", rotMat, self.grid) + flow[:, rot_dim:-1, ...] + flow[:, -1,
                                                                                                          ...] * self.grid
            # flow[:, -1, ...] * self.grid
        return newflow



class AffineReform(nn.Module):
    """
    Lie Group Transformer
    """

    def __init__(self, size, ndims=3, device='cuda'):
        super().__init__()
        self.mode = 'bilinear'

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.type(torch.FloatTensor).to(device)
        self.shape = grid.shape[1:]
        maxdir = max(self.shape)
        for i in range(len(self.shape)):
            grid[i, ...] = 2 * (grid[i, ...] / (self.shape[i] - 1) - 0.5) * (self.shape[i] - 1) / (max(self.shape) - 1)
        grid = torch.cat((grid, torch.ones_like(grid[0].unsqueeze(0))))
        grid = torch.unsqueeze(grid, 0)
        if ndims == 3:
            I = torch.zeros(1, size[0], size[1], size[2], ndims, ndims).to(device)
            I[0, ...] = torch.eye(3)
            self.I = I.permute(0, 4, 5, 1, 2, 3)
            A = torch.zeros(1, ndims, ndims, size[0], size[1], size[2])
            self.A = A.to(device)
            self.i, self.j = torch.triu_indices(ndims, ndims, offset=1)
        elif ndims == 2:
            V = torch.zeros(1, size[0], size[1], ndims, ndims).to(device)
            self.V = V
            A = torch.zeros(1, size[0], size[1], ndims, ndims)
            self.A = A.to(device)
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        # self.register_buffer('grid', self.grid, persistent=False)
        self.grid = grid.to(device)

    def Mat_init(self, input, dim, device):
        size = input.size()
        if dim == 2:
            rot_dim = dim * (dim - 1) // 2  # rotational degrees of freedom
            angle = input[:, 0, ...]
            A = 0 * self.A
            MatExp = torch.zeros(size[0], size[2], size[3], dim, dim + 1).to(device)
            A[:, :, :, 0, 0] = torch.cos(angle)
            A[:, :, :, 0, 1] = -torch.sin(angle)
            A[:, :, :, 1, 0] = torch.sin(angle)
            A[:, :, :, 1, 1] = torch.cos(angle)

            V = 0 * self.V
            V[:, :, :, 0, 0] = abs(torch.sin(angle)) / (abs(angle) + 1e-7)
            V[:, :, :, 1, 1] = abs(torch.sin(angle)) / (abs(angle) + 1e-7)
            V[:, :, :, 0, 1] = -(torch.sin(angle) ** 2) / ((angle * (1 - torch.cos(angle))) + 1e-7)
            V[:, :, :, 1, 0] = (torch.sin(angle) ** 2) / ((angle * (1 - torch.cos(angle))) + 1e-7)

            MatExp[..., :2, :2] = A
            MatExp[..., :2, 2] = torch.einsum("ijkmn,injk->ijkm", V, input[:, rot_dim:])
            return MatExp  # rodriguez formular
        elif True:
            A = 0 * self.A
            MatExp = torch.zeros(size[0], dim, dim + 1, size[2], size[3], size[4]).to(device)
            MatExp[:, :3, :3] = input[:, :-3, ...].reshape(1, 3, 3, size[2], size[3], size[4])
            MatExp[:, :3, 3] = input[:, -3:, ...]
            return MatExp
        else:
            return None

    def forward(self, flow, device):
        # new locations
        shape = self.grid.shape[2:]
        Matrix = self.Mat_init(flow, len(shape), device)
        return torch.einsum("ijklmn,iklmn->ijlmn", Matrix, self.grid)


class AffMatrixInt(nn.Module):
    """
    Integrates a manifold valued field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps, device='cuda'):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.inshape = inshape
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = MatrixTransformer(inshape)
        dims = len(inshape)
        if len(inshape) == 2:
            V = torch.zeros(1, inshape[0], inshape[1], dims, dims).to(device)
            self.V = V
            A = torch.zeros(1, inshape[0], inshape[1], dims, dims).to(device)
            self.A = A
        if dims == 3:
            I = torch.zeros(1, 1, 1, 1, dims, dims)
            I[0, ...] = torch.eye(3)
            self.I = I.to(device)
            A = torch.zeros(1, inshape[0], inshape[1], inshape[2], dims, dims)
            self.A = A.to(device)
            self.i, self.j = torch.triu_indices(len(inshape), len(inshape), offset=1)
            self.MatExp = torch.zeros(1, inshape[0], inshape[1], inshape[2], dims + 1, dims + 1).to(device)

    def Matexp(self, input):
        size = input.size()
        if True:
            log_R_vee = input[..., :3]
            log_t_vee = input[..., 3:6]
            log_d_vee = input[..., 6:9]
            log_a_vee = input[..., 9:]
            MatExp = 0 * self.MatExp
            MatExp[..., self.i, self.j] = log_R_vee
            MatExp = (MatExp - MatExp.transpose(4, 5))  # create antisymmetric matrix
            MatExp[..., self.i, self.j] += log_a_vee
            MatExp[..., self.j, self.i] += log_a_vee
            dir_indizes = [0, 1, 2]
            MatExp[..., dir_indizes, dir_indizes] = log_d_vee
            MatExp[..., :3, 3] = log_t_vee
            return torch.matrix_exp(MatExp)
        else:
            rot_dim = 3 * (3 - 1) // 2  # rotational degrees of freedom
            A = 0 * self.A
            norm = torch.linalg.vector_norm(input[:, :rot_dim], dim=1)  # + 1e-7
            A[..., self.i, self.j] = input[:, :rot_dim, ...].permute(0, 2, 3, 4, 1)
            A = (A - A.transpose(4, 5))  # create antisymmetric matrix
            MatExp = 0 * self.MatExp
            A_sq = torch.linalg.matrix_power(A, 2)
            norm_sq = norm * norm
            a = torch.where(norm < 0.001, 1 - norm_sq / 6 * (1 - norm_sq / 20 * (1 - norm_sq / 42)),
                            torch.sin(norm) / norm)
            b = torch.where(norm < 0.001, (1 - norm_sq / 12 * (1 - norm_sq / 30 * (1 - norm_sq / 56))) / 2,
                            (1 - torch.cos(norm)) / norm_sq)
            c = torch.where(norm < 0.001, (1 - norm_sq / 20 * (1 - norm_sq / 42 * (1 - norm_sq / 72))) / 6,
                            (norm - torch.sin(norm)) / (norm_sq * norm))  # Taylorentwicklung'''

            a, b, c = a.unsqueeze(4).unsqueeze(5), b.unsqueeze(4).unsqueeze(5), c.unsqueeze(4).unsqueeze(5)
            rot = self.I + a * A + b * A_sq
            V = self.I + b * A + c * A_sq
            MatExp[..., :3, :3] = rot
            MatExp[..., :3, 3] = torch.einsum("ijkmno,iojkm->ijkmn", V, input[:, rot_dim:-1])
            MatExp[..., 3, 3] = 1
            mult = torch.exp(input[:, -1]).unsqueeze(1).permute(0, 2, 3, 4, 1).unsqueeze(5)
            return mult * MatExp

    def forward(self, vec, device='cuda'):
        if len(self.inshape) == 3:
            vec = vec * self.scale
            mat_field = self.Matexp(vec)
            for _ in range(self.nsteps):
                interp = self.transformer(mat_field, mat_field).reshape(1, 4, 4, self.inshape[0], self.inshape[1],
                                                                        self.inshape[2])
                mat_field = torch.einsum("ijkmno,imnokl->imnojl", interp, mat_field)
        else:
            vec = vec * self.scale
            mat_field = self.Matexp(vec).permute(0, 3, 4, 1, 2)
            for _ in range(self.nsteps):
                interp = self.transformer(mat_field, mat_field)
                reshaped_interp = interp.reshape(1, 3, 3, self.inshape[0], self.inshape[1])
                mat_field = torch.einsum("ijkmn,iklmn->ijlmn", reshaped_interp, mat_field)
        return mat_field[...,:3,:].reshape(1, self.inshape[0],self.inshape[1],self.inshape[2],12)


class DriftTransformer2(nn.Module):
    """
    Lie Group Transformer
    """

    def __init__(self, size, ndims=3, device='cuda'):
        super().__init__()
        self.mode = 'bilinear'

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids).float()
        self.shape = grid.shape[1:]
        for i in range(len(self.shape)):
            grid[i, ...] = 2 * (grid[i, ...] / (self.shape[i] - 1) - 0.5) * (self.shape[i] - 1) / (max(self.shape) - 1)
        grid = torch.cat((grid, torch.ones_like(grids[0].unsqueeze(0))))
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).to(device)
        if ndims == 3:
            I = torch.zeros(1, 1, 1, 1, ndims, ndims)
            I[0, ...] = torch.eye(3)
            self.I = I.to(device)
            A = torch.zeros(1, size[0], size[1], size[2], ndims, ndims)
            self.A = A.to(device)
            self.i, self.j = torch.triu_indices(len(size), len(size), offset=1)
            self.MatExp = torch.zeros(1, size[0], size[1], size[2], ndims + 1, ndims + 1).to(device)
        elif ndims == 2:
            V = torch.zeros(1, size[0], size[1], ndims, ndims).to(device)
            self.V = V
            A = torch.zeros(1, size[0], size[1], ndims, ndims)
            self.A = A.to(device)
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        # self.register_buffer('grid', self.grid, persistent=False)
        self.grid = grid.to(device)

    def Matexp(self, input, dim, device):
        size = input.size()
        if False:
            input = input.permute(0, 2, 3, 4, 1)
            log_R_vee = input[..., :3]
            log_t_vee = input[..., 3:]
            MatExp = 0 * self.MatExp
            MatExp[..., 0, 1] = -log_R_vee[..., 2]
            MatExp[..., 0, 2] = log_R_vee[..., 1]
            MatExp[..., 1, 2] = -log_R_vee[..., 0]
            MatExp[..., 1, 0] = log_R_vee[..., 2]
            MatExp[..., 2, 0] = -log_R_vee[..., 1]
            MatExp[..., 2, 1] = log_R_vee[..., 0]
            MatExp[..., :3, 3] = log_t_vee
            return torch.matrix_exp(MatExp)
        else:
            rot_dim = dim * (dim - 1) // 2  # rotational degrees of freedom
            A = 0 * self.A
            norm = torch.linalg.vector_norm(input[:, :rot_dim], dim=1)  # + 1e-7
            A[..., 0, 1] = -input[:, 2, ...]
            A[..., 0, 2] = input[:, 1, ...]
            A[..., 1, 2] = -input[:, 0, ...]
            A[..., 1, 0] = input[:, 2, ...]
            A[..., 2, 0] = -input[:, 1, ...]
            A[..., 2, 1] = input[:, 0, ...]
            MatExp = 0 * self.MatExp
            A_sq = torch.linalg.matrix_power(A, 2)
            norm_sq = norm * norm
            a = torch.where(norm < 0.001, 1 - norm_sq / 6 * (1 - norm_sq / 20 * (1 - norm_sq / 42)),
                            torch.sin(norm) / norm)
            b = torch.where(norm < 0.001, (1 - norm_sq / 12 * (1 - norm_sq / 30 * (1 - norm_sq / 56))) / 2,
                            (1 - torch.cos(norm)) / norm_sq)
            c = torch.where(norm < 0.001, (1 - norm_sq / 20 * (1 - norm_sq / 42 * (1 - norm_sq / 72))) / 6,
                            (norm - torch.sin(norm)) / (norm_sq * norm))  # Taylorentwicklung'''
            a, b, c = a.unsqueeze(4).unsqueeze(5), b.unsqueeze(4).unsqueeze(5), c.unsqueeze(4).unsqueeze(5)
            rot = self.I + a * A + b * A_sq
            V = self.I + b * A + c * A_sq
            MatExp[..., :3, :3] = rot
            MatExp[..., :3, 3] = torch.einsum("ijkmno,iojkm->ijkmn", V, input[:, rot_dim:])
            MatExp[..., 3, 3] = 1
            # mult = torch.exp(input[:, -1]).unsqueeze(1).permute(0, 2, 3, 4, 1).unsqueeze(5)
            return MatExp  # rodriguez formular

    def forward(self, src, flow, device, padding='zeros', mode='bilinear'):
        # new locations

        shape = self.grid.shape[2:]
        Matrix = self.Matexp(flow, len(shape), device)
        if len(shape) == 2:
            new_loc = torch.einsum("ilmjk,iklm->ijlm", Matrix,
                                   self.grid)  # batchwise matrix-vector product
            new_locs = torch.zeros_like(new_loc)
            for i in range(len(self.shape)):
                new_locs[:, i, ...] = new_loc[:, i, ...] / ((self.shape[i] - 1) / (max(self.shape) - 1))
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
            return nnf.grid_sample(src.reshape(1, -1, shape[0], shape[1]), new_locs, align_corners=True, mode=mode,
                                   padding_mode=padding), new_loc[:, :-1], self.grid[:, :-1]
        elif len(shape) == 3:
            # new_loc = torch.einsum("ijklm,iklm->ijlm", rotMat, self.grid) + flow[:, 3:, ...]id.to(
            new_loc = torch.einsum("ijkmno,iojkm->ijkmn", Matrix,
                                   self.grid).permute(0, 4, 1, 2, 3)  # batchwise matrix-vector product
            new_locs = torch.zeros_like(new_loc)
            for i in range(len(self.shape)):
                new_locs[:, i, ...] = new_loc[:, i, ...] / ((self.shape[i] - 1) / (max(self.shape) - 1))
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
            return nnf.grid_sample(src.reshape(1, -1, shape[0], shape[1], shape[2]), new_locs, align_corners=True,
                                   mode=mode, padding_mode=padding), new_loc[:, :-1], self.grid[:, :-1]
        else:
            return None

class DriftInt2(nn.Module):
    """
    Integrates a manifold valued field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps, device='cuda', ndims=3):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.inshape = inshape
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = DriftTransformer2(inshape, ndims=ndims)
        dims = len(inshape)
        self.rot_dim = len(self.inshape) * (len(self.inshape) - 1) // 2  # rotational degrees of freedom
        if dims == 3:
            I = torch.zeros(1, 1, 1, 1, dims, dims)
            I[0, ...] = torch.eye(3)
            self.I = I.to(device)
            A = torch.zeros(1, inshape[0], inshape[1], inshape[2], dims, dims)
            self.A = A.to(device)
            self.MatExp = torch.zeros(1, inshape[0], inshape[1], inshape[2], ndims + 1, ndims + 1).to(device)
            self.SO3 = SO3.apply
        elif dims == 2:
            V = torch.zeros(1, inshape[0], inshape[1], dims, dims).to(device)
            self.V = V
            A = torch.zeros(1, inshape[0], inshape[1], dims, dims)
            self.A = A.to(device)

    def Matexp(self, input, dim, device):
        size = input.size()
        if False:
            input = input.permute(0, 2, 3, 4, 1)
            log_R_vee = input[..., :3]
            log_t_vee = input[..., 3:]
            MatExp = 0 * self.MatExp
            MatExp[..., 0, 1] = -log_R_vee[..., 2]
            MatExp[..., 0, 2] = log_R_vee[..., 1]
            MatExp[..., 1, 2] = -log_R_vee[..., 0]
            MatExp[..., 1, 0] = log_R_vee[..., 2]
            MatExp[..., 2, 0] = -log_R_vee[..., 1]
            MatExp[..., 2, 1] = log_R_vee[..., 0]
            MatExp[..., :3, 3] = log_t_vee
            return torch.matrix_exp(MatExp)
        elif True:
            rot_dim = dim * (dim - 1) // 2  # rotational degrees of freedom
            A = 0 * self.A
            norm = torch.linalg.vector_norm(input[:, :rot_dim], dim=1)  # + 1e-7
            A[..., 0, 1] = -input[:, 2, ...]
            A[..., 0, 2] = input[:, 1, ...]
            A[..., 1, 2] = -input[:, 0, ...]
            A[..., 1, 0] = input[:, 2, ...]
            A[..., 2, 0] = -input[:, 1, ...]
            A[..., 2, 1] = input[:, 0, ...]
            # A[..., self.i, self.j] = input[:, :rot_dim, ...].permute(0, 2, 3, 4, 1)
            # A = (A - A.transpose(4, 5))  # create antisymmetric matrix
            MatExp = 0 * self.MatExp
            A_sq = torch.linalg.matrix_power(A, 2)
            norm_sq = norm * norm
            a = torch.where(norm < 0.001, 1 - norm_sq / 6 * (1 - norm_sq / 20 * (1 - norm_sq / 42)),
                            torch.sin(norm) / norm)
            b = torch.where(norm < 0.001, (1 - norm_sq / 12 * (1 - norm_sq / 30 * (1 - norm_sq / 56))) / 2,
                            (1 - torch.cos(norm)) / norm_sq)
            c = torch.where(norm < 0.001, (1 - norm_sq / 20 * (1 - norm_sq / 42 * (1 - norm_sq / 72))) / 6,
                            (norm - torch.sin(norm)) / (norm_sq * norm))  # Taylorentwicklung'''
            a, b, c = a.unsqueeze(4).unsqueeze(5), b.unsqueeze(4).unsqueeze(5), c.unsqueeze(4).unsqueeze(5)
            rot = self.I + a * A + b * A_sq
            V = self.I + b * A + c * A_sq
            MatExp[..., :3, :3] = rot
            MatExp[..., :3, 3] = torch.einsum("ijkmno,iojkm->ijkmn", V, input[:, rot_dim:])
            MatExp[..., 3, 3] = 1
            # mult = torch.exp(input[:, -1]).unsqueeze(1).permute(0, 2, 3, 4, 1).unsqueeze(5)
            return MatExp  # rodriguez formular

    def Matlog(self, input, dim=3, device='cuda'):
        size = input.size()
        if dim == 2:
            input = input.permute(0, 3, 4, 1, 2)
            i, j = torch.triu_indices(dim, dim, offset=1)
            symdiff = (-input + input.transpose(1, 2))
            sin_wert = symdiff[:, i, j, :, :] / 2
            diag = input.diagonal(offset=0, dim1=1, dim2=2)
            cos_wert = diag[..., :-1].sum(-1) / 2
            angle = torch.atan2(sin_wert, cos_wert)
            Vinv = 0 * self.V
            Vinv[..., 0, 0] = abs(torch.sin(angle)) / (abs(angle) + 1e-7)
            Vinv[..., 1, 1] = abs(torch.sin(angle)) / (abs(angle) + 1e-7)
            Vinv[..., 0, 1] = (torch.sin(angle) ** 2) / ((angle * (1 - torch.cos(angle))) + 1e-7)
            Vinv[:, :, :, 1, 0] = -(torch.sin(angle) ** 2) / ((angle * (1 - torch.cos(angle))) + 1e-7)
            trans = torch.einsum("ijklm,imjk->iljk", Vinv, input[:, :2, 2, ...])
            trans = angle ** 2 / (2 * (1 - torch.cos(angle))) * trans
            vec = torch.concatenate((angle, trans), dim=1)
            return vec

        else:
            i, j = torch.triu_indices(dim, dim, offset=1)
            # symdiff = (input[:, :, :, :, :-1, :-1] - input[:, :, :, :, :-1, :-1].transpose(4, 5))
            omega = self.SO3(input[:, :, :, :, :-1, :-1].reshape(-1, 3, 3)).reshape(size[0], size[1], size[2], size[3],
                                                                                    3)
            omegax = 0 * self.A
            omegax[..., 0, 1] = -omega[..., 2]
            omegax[..., 0, 2] = +omega[..., 1]
            omegax[..., 1, 2] = -omega[..., 0]
            omegax[..., 1, 0] = +omega[..., 2]
            omegax[..., 2, 0] = -omega[..., 1]
            omegax[..., 2, 1] = +omega[..., 0]
            '''tester = torch.rand(100, 6)  # numeric gradient check
            R = se3_exp_map(tester)
            R.requires_grad = True
            test = gradcheck(self.SO3, R[..., :3, :3].double(), eps=5e-7, atol=1e-6)
            print(test)'''
            norm = torch.linalg.vector_norm(omega, dim=4)
            norm_sq = norm * norm
            omega_sq = torch.linalg.matrix_power(omegax, 2)  # rodriguez formular
            b = torch.where(norm < 0.001, (1 / 12 * (1 + norm_sq / 60 * (1 + norm_sq / 42 * (1 + norm_sq / 40)))),
                            1 / norm_sq * (1 - (norm * torch.sin(norm)) / (
                                    2 * (1 - torch.clamp(torch.cos(norm), max=1 - 1e-7))))).unsqueeze(4).unsqueeze(5)
            # b = (1 / 12 * (1 + norm_sq / 60 * (1 + norm_sq / 42 * (1 + norm_sq / 40)))).unsqueeze(4).unsqueeze(5)
            Vinv = self.I - omegax / 2 + b * omega_sq  # rodriguez formular
            trans = torch.einsum("ijkmno,ijkmo->ijkmn", Vinv, input[..., :3, 3])
            vec = torch.concatenate((omega, trans), dim=4)
            return vec.permute(0, 4, 1, 2, 3)

    def forward(self, vec, device, padding='zeros'):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            interp, _, _ = self.transformer(vec, vec, padding=padding, device=device)
            Mat2 = self.Matexp(interp, len(self.inshape), device)
            Mat = self.Matexp(vec, len(self.inshape), device)
            if len(self.inshape) == 2:
                mulMat = torch.einsum("ijkno,ijkop->ijknp", Mat2, Mat)
            else:
                mulMat = torch.einsum("ijkmno,ijkmop->ijkmnp", Mat2, Mat)
            # vec[:, :3] = vec[:, :3] + interp[:, :3]
            vec = self.Matlog(mulMat, len(self.inshape), device=device)
        return vec

