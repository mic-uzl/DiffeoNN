import torch
import torch.nn.functional as F
import numpy as np
import math

torch.set_default_dtype(torch.float64)


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, ndims=3):
        # compute filters
        # set window size
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        win = [9] * ndims if win is None else win
        self.win = win
        self.sum_filt = torch.ones([1, 1, *win]).to('cuda')

    def loss(self, y_true, y_pred, ndims=3):
        if len(y_true.size()) == 3:
            H, W, Z = y_true.size()
            Ii = y_true.reshape(1, 1, H, W, Z)
            Ji = y_pred.reshape(1, 1, H, W, Z)
        else:
            Ii = y_true
            Ji = y_pred
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        pad_no = math.floor(self.win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji
        I_sum = conv_fn(Ii, self.sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, self.sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, self.sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, self.sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, self.sum_filt, stride=stride, padding=padding)

        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1 - torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class ME:
    """
    Mean error (L1)loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean(abs(y_true - y_pred))


class Dice:
    """
    N-D dice for segmentation
    """

    def __init__(self, penalty='l1', loss_mult=1.):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return 1 - self.loss_mult * dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = (y[1:, ...] - y[:-1, ...]) * (max(y.size()) - 1) / 2

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, pos_flow):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(pos_flow)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(pos_flow)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) # / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class GradNorm:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = (y[1:, ...] - y[:-1, ...]) * (max(y.size()) - 1) / 2

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, pos_flow):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(pos_flow)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(pos_flow)]

        df = [torch.max(torch.sum(f, dim=1)) for f in dif]
        grad = torch.clip(max(df) - 10, min=0)  # / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class Lip:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = (y[1:, ...] - y[:-1, ...]) * (max(y.size()) - 1) / 2

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, pos_flow):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(pos_flow)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(pos_flow)]

        df = [torch.mean(torch.clip(torch.flatten(f) - 2, min=0)) for f in dif]
        grad = sum(df)  # / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class Laplace:
    """
    N-D Laplacian loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _2diffs(self, y):

        # df1 = (-y[1:2, ...] + y[2:3, ...]) * (max(y.size()) - 1) ** 2 / 4  # boundary condition
        dfi = (-4*y[...,1:-1,1:-1] + y[...,:-2,1:-1]+y[...,2:,1:-1]+y[...,1:-1,:-2]+y[...,1:-1,2:]) * (max(y.size()) - 1) ** 2 / 4  # central differences
        return dfi

    def loss(self, pos_flow):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._2diffs(pos_flow)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = self._2diffs(pos_flow)**2


        df = torch.mean(torch.flatten(dif, start_dim=2), dim=-1)
        grad = sum(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class detJac:
    """
    penalizes negative Determinants of the Jacobian.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = (y[1:, ...] - y[:-1, ...]) * (max(y.size()) - 1) / 2  # nodal differences

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def _crossdiffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)
        #print('ndims', ndims)
        if ndims == 2:
            #df = [None] * ndims
            df1 = (y[:, :, 1:, :-1] - y[:, :, 1:, 1:]) * (max(y.size()) - 1) / 2
            df2 = (y[:, :, :-1, 1:] - y[:, :, :-1, :-1]) * (max(y.size()) - 1) / 2
            return [df1, df2]
        elif ndims == 3:    
            df1 = (y[:, :, 1:, :-1, :-1] - y[:, :, :-1, :-1, 1:]) * (max(y.size()) - 1) / 2  # nodal differences
            df2 = (y[:, :, :-1, 1:, :-1] - y[:, :, :-1, :-1, 1:]) * (max(y.size()) - 1) / 2  # nodal differences
            df3 = (y[:, :, 1:, 1:, 1:] - y[:, :, :-1, :-1, 1:]) * (max(y.size()) - 1) / 2  # nodal differences
            return [df1, df2, df3]

    def loss(self, pos_flow):
        diffs = self._diffs(pos_flow)
        cross_diff = self._crossdiffs(pos_flow)
        dims = len(diffs)
        assert dims == 2 or dims == 3, 'only dimensions 2 and 3 are supported. Got %d' % dims
        if dims == 2:
            dx = diffs[0]
            dy = diffs[1]
            print('dx', dx.shape, 'dy', dy.shape)
            # det = dx[:, 0, :, :-2] * dy[:, 1, :-2, :] - dx[:, 1, :, :-2] * dy[:, 0, :-2, :]
            stacked = torch.concatenate((dx[..., :-2], dy[..., :-2, :]), dim=0)
            stacked = stacked.permute(2, 3, 1, 0)
            determ = torch.det(stacked)
        if dims == 3:
            dx = diffs[0]
            dy = diffs[1]
            dz = diffs[2]
            stacked = torch.concatenate((dx[..., :, :-1, :-1], dy[..., :-1, :, :-1], dz[..., :-1, :-1, :]),
                                        dim=0).permute(2, 3, 4, 1, 0)
            stacked2 = torch.concatenate((-dx[..., :, 1:, :-1], -dy[..., 1:, :, :-1], dz[..., 1:, 1:, :]),
                                         dim=0).permute(2, 3, 4, 1, 0)
            stacked3 = torch.concatenate((-dx[..., :, :-1, 1:], dy[..., 1:, :, 1:], -dz[..., 1:, :-1, :]),
                                         dim=0).permute(2, 3, 4, 1, 0)
            stacked4 = torch.concatenate((dx[..., :, 1:, 1:], -dy[..., :-1, :, 1:], -dz[..., :-1, 1:, :]),
                                         dim=0).permute(2, 3, 4, 1, 0)
            stacked5 = torch.concatenate(cross_diff, dim=0).permute(2, 3, 4, 1, 0)
            determ = torch.det(stacked) - 0.05
            determ2 = torch.det(stacked2) - 0.05
            determ3 = torch.det(stacked3) - 0.05
            determ4 = torch.det(stacked4) - 0.05
            determ5 = torch.det(stacked5) - 0.1
        return (torch.mean(abs(determ.clip(max=0))) + torch.mean(abs(determ2.clip(max=0))) + torch.mean(
            abs(determ3.clip(max=0)))
                + torch.mean(abs(determ4.clip(max=0))) + 2 * torch.mean(
                    abs(determ5.clip(max=0)))) / 6  # torch.mean(abs(torch.log(torch.clip(determ, min=1e-7))))
        # torch.mean(abs(1-determ)) torch.mean(abs(determ.clip(max=0)))
        # return (torch.mean(abs(determ-1)) + torch.mean(abs(determ2-1)) + torch.mean(
        #    abs(determ3-1))
        #        + torch.mean(abs(determ4-1)) + 2 * torch.mean(abs(determ5-1))) / 6


class GlobNCC:
    """
    Global normalized cross correlation loss.
    """

    def __init__(self, win=None, ndims=3):
        # compute filters
        # set window size
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        win = [9] * ndims if win is None else win
        self.win = win
        self.sum_filt = torch.ones([1, 1, *win]).to('cuda')

    def loss(self, y_true, y_pred, ndims=3):
        if len(y_true.size()) >= 3:
            H, W, Z = y_true.size()[-3:]
            Ii = y_true.reshape(1, -1, H, W, Z)
            Ji = y_pred.reshape(1, -1, H, W, Z)
        else:
            Ii = y_true
            Ji = y_pred

        # compute CC squares
        Ii, Ji = Ii - torch.mean(Ii, dim=(2, 3, 4), keepdim=True), Ji - torch.mean(Ji, dim=(2, 3, 4), keepdim=True)
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = torch.sum(Ii * Ji, dim=(2, 3, 4), keepdim=True)
        cc = IJ * IJ / (torch.sum(I2, dim=(2, 3, 4), keepdim=True) * torch.sum(J2, dim=(2, 3, 4), keepdim=True) + 1e-7)
        return 1 - torch.mean(cc)


def compute_jacobian_loss(input_coords, output, dims, batch_size=None):
    """Compute the jacobian regularization loss."""

    # Compute Jacobian matrices
    jac = compute_jacobian_matrix(input_coords, output, dims)

    # Compute determinants and calculate loss
    determ = torch.det(jac)
    loss = torch.mean(abs(determ.clip(max=0)))
    return loss


def compute_jacobian_matrix(output, dims, add_identity=False):
    """Compute the Jacobian matrix of the output wrt the input."""

    jacobian_matrix = np.zeros((*output.shape[:-1], dims, dims))
    for i in range(dims):
        jacobian_matrix[..., i, :] = np.array(np.gradient(output[..., i])).transpose(1,2,3,0)
        if add_identity:
            jacobian_matrix[..., i, i] += np.ones_like(jacobian_matrix[..., i, i])
    return np.linalg.det(jacobian_matrix)


def gradient(input_coords, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input_coords], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad


class GradAuto:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l2', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, jac):
        if self.penalty == 'l1':
            return torch.mean(abs(jac))
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            return torch.mean(jac ** 2)


class detJacAuto:
    """
    penalizes negative Determinants of the Jacobian.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, jacobian):
        assert jacobian.size()[1] == jacobian.size()[2], 'only square matrices are possible'
        dims = jacobian.size()[1]
        assert dims == 2 or dims == 3, 'only dimensions 2 and 3 are supported. Got %d' % dims
        # Compute determinants and calculate loss
        determ = torch.det(jacobian)
        loss = torch.mean(abs(determ - 1))
        return loss


class Metrik_detJac:
    """
    penalizes negative Determinants of the Jacobian.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = (y[1:, ...] - y[:-1, ...]) * (max(y.size()) - 1) / 2  # nodal differences

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def _crossdiffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)
        df = [None] * ndims
        df1 = (y[:, :, 1:, :-1, :-1] - y[:, :, :-1, :-1, 1:]) * (max(y.size()) - 1) / 2  # nodal differences
        df2 = (y[:, :, :-1, 1:, :-1] - y[:, :, :-1, :-1, 1:]) * (max(y.size()) - 1) / 2  # nodal differences
        df3 = (y[:, :, 1:, 1:, 1:] - y[:, :, :-1, :-1, 1:]) * (max(y.size()) - 1) / 2  # nodal differences
        return [df1, df2, df3]

    def loss(self, pos_flow):
        diffs = self._diffs(pos_flow)
        cross_diff = self._crossdiffs(pos_flow)
        dims = len(diffs)
        assert dims == 2 or dims == 3, 'only dimensions 2 and 3 are supported. Got %d' % dims
        if dims == 2:
            dx = diffs[0]
            dy = diffs[1]
            # det = dx[:, 0, :, :-2] * dy[:, 1, :-2, :] - dx[:, 1, :, :-2] * dy[:, 0, :-2, :]
            stacked = torch.concatenate((dx[..., :-2], dy[..., :-2, :]), dim=0)
            stacked = stacked.permute(2, 3, 1, 0)
            determ = torch.det(stacked)
        if dims == 3:
            dx = diffs[0]
            dy = diffs[1]
            dz = diffs[2]
            # det = dx[:, 0, :, :-2] * dy[:, 1, :-2, :] - dx[:, 1, :, :-2] * dy[:, 0, :-2, :]
            stacked = torch.concatenate((dx[..., :, :-1, :-1], dy[..., :-1, :, :-1], dz[..., :-1, :-1, :]),
                                        dim=0).permute(2, 3, 4, 1, 0)
            stacked2 = torch.concatenate((-dx[..., :, 1:, :-1], -dy[..., 1:, :, :-1], dz[..., 1:, 1:, :]),
                                         dim=0).permute(2, 3, 4, 1, 0)
            stacked3 = torch.concatenate((-dx[..., :, :-1, 1:], dy[..., 1:, :, 1:], -dz[..., 1:, :-1, :]),
                                         dim=0).permute(2, 3, 4, 1, 0)
            stacked4 = torch.concatenate((dx[..., :, 1:, 1:], -dy[..., :-1, :, 1:], -dz[..., :-1, 1:, :]),
                                         dim=0).permute(2, 3, 4, 1, 0)
            stacked5 = torch.concatenate(cross_diff, dim=0).permute(2, 3, 4, 1, 0)
            determ = torch.det(stacked)
            determ2 = torch.det(stacked2)
            determ3 = torch.det(stacked3)
            determ4 = torch.det(stacked4)
            determ5 = torch.det(stacked5)
        return (torch.mean(torch.where(determ < 0, 1.0, 0.0)) + torch.mean(torch.where(determ2 < 0, 1.0, 0.0)) +
                torch.mean(torch.where(determ3 < 0, 1.0, 0.0)) + torch.mean(
                    torch.where(determ4 < 0, 1.0, 0.0)) + 2 * torch.mean(torch.where(determ5 < 0, 1.0, 0.0))) / 6


class OlddetJac:
    """
    penalizes negative Determinants of the Jacobian.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = (y[2:, ...] - y[:-2, ...]) * (max(y.size()) - 1) / 4  # central differences

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, pos_flow):
        diffs = self._diffs(pos_flow)
        dims = len(diffs)
        assert dims == 2 or dims == 3, 'only dimensions 2 and 3 are supported. Got %d' % dims
        if dims == 2:
            dx = diffs[0]
            dy = diffs[1]
            # det = dx[:, 0, :, :-2] * dy[:, 1, :-2, :] - dx[:, 1, :, :-2] * dy[:, 0, :-2, :]
            stacked = torch.concatenate((dx[..., :-2], dy[..., :-2, :]), dim=0)
            stacked = stacked.permute(2, 3, 1, 0)
            determ = torch.det(stacked)
        if dims == 3:
            dx = diffs[0]
            dy = diffs[1]
            dz = diffs[2]
            # det = dx[:, 0, :, :-2] * dy[:, 1, :-2, :] - dx[:, 1, :, :-2] * dy[:, 0, :-2, :]
            stacked = torch.concatenate((dx[..., :, 1:-1, 1:-1], dy[..., 1:-1, :, 1:-1], dz[..., 1:-1, 1:-1, :]), dim=0)
            stacked = stacked.permute(2, 3, 4, 1, 0)
            determ = torch.det(stacked) - 0.05
        return torch.mean(abs(determ.clip(max=0)))  # torch.mean(abs(torch.log(torch.clip(determ, min=0.00001))))
        # torch.mean(abs(1-determ)) torch.mean(abs(determ.clip(max=0)))


class MetrikOlddetJac:
    """
    penalizes negative Determinants of the Jacobian.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = (y[2:, ...] - y[:-2, ...]) * (max(y.size()) - 1) / 4  # central differences

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, pos_flow, add_identity=False):
        diffs = self._diffs(pos_flow)
        dims = len(diffs)
        assert dims == 2 or dims == 3, 'only dimensions 2 and 3 are supported. Got %d' % dims
        if dims == 2:
            dx = diffs[0]
            dy = diffs[1]
            # det = dx[:, 0, :, :-2] * dy[:, 1, :-2, :] - dx[:, 1, :, :-2] * dy[:, 0, :-2, :]
            stacked = torch.concatenate((dx[..., :-2], dy[..., :-2, :]), dim=0)
            stacked = stacked.permute(2, 3, 1, 0)
            determ = torch.det(stacked)
        if dims == 3:
            dx = diffs[0]
            dy = diffs[1]
            dz = diffs[2]
            if add_identity == True:
                dx[:,0, ...] += 1
                dy[:,1, ...] += 1
                dz[:,2, ...] += 1
            # det = dx[:, 0, :, :-2] * dy[:, 1, :-2, :] - dx[:, 1, :, :-2] * dy[:, 0, :-2, :]
            stacked = torch.concatenate((dx[..., :, 1:-1, 1:-1], dy[..., 1:-1, :, 1:-1], dz[..., 1:-1, 1:-1, :]), dim=0)
            stacked = stacked.permute(2, 3, 4, 0, 1)
            determ = torch.det(stacked)
        return torch.mean(
            torch.where(determ < 0, 1.0, 0.0))  # torch.mean(abs(torch.log(torch.clip(determ, min=0.00001))))
        # torch.mean(abs(1-determ)) torch.mean(abs(determ.clip(max=0)))
