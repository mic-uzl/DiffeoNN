from typing import List
import torch
import torch.nn as nn
import math
import numpy as np
import torch
from torch import nn
from torch.nn.init import _calculate_correct_fan
from torch.nn.functional import normalize

torch.set_default_dtype(torch.float32)


def compute_weight(self, do_power_iteration: bool) -> torch.Tensor:
    # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
    #     updated in power iteration **in-place**. This is very important
    #     because in `DataParallel` forward, the vectors (being buffers) are
    #     broadcast from the parallelized module to each module replica,
    #     which is a new module object created on the fly. And each replica
    #     runs its own spectral norm power iteration. So simply assigning
    #     the updated vectors to the module this function runs on will cause
    #     the update to be lost forever. And the next time the parallelized
    #     module is replicated, the same randomly initialized vectors are
    #     broadcast and used!
    #
    #     Therefore, to make the change propagate back, we rely on two
    #     important behaviors (also enforced via tests):
    #       1. `DataParallel` doesn't clone storage if the broadcast tensor
    #          is already on correct device; and it makes sure that the
    #          parallelized module is already on `device[0]`.
    #       2. If the out tensor in `out=` kwarg has correct shape, it will
    #          just fill in the values.
    #     Therefore, since the same power iteration is performed on all
    #     devices, simply updating the tensors in-place will make sure that
    #     the module replica on `device[0]` will update the _u vector on the
    #     parallelized module (by shared storage).
    #
    #    However, after we update `u` and `v` in-place, we need to **clone**
    #    them before using them to normalize the weight. This is to support
    #    backproping through two forward passes, e.g., the common pattern in
    #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
    #    complain that variables needed to do backward for the first forward
    #    (i.e., the `u` and `v` vectors) are changed in the second forward.
    weight = getattr(module, self.name + '_orig')
    u = getattr(module, self.name + '_u')
    v = getattr(module, self.name + '_v')
    weight_mat = self.reshape_weight_to_matrix(weight)

    if do_power_iteration:
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
            if self.n_power_iterations > 0:
                # See above on why we need to clone
                u = u.clone(memory_format=torch.contiguous_format)
                v = v.clone(memory_format=torch.contiguous_format)

    sigma = torch.dot(u, torch.mv(weight_mat, v))
    weight = weight / sigma
    return weight


def siren_uniform_(tensor: torch.Tensor, mode: str = 'fan_in', c: float = 6):
    r"""Fills the input `Tensor` with values according to the method
    described in ` Implicit Neural Representations with Periodic Activation
    Functions.` - Sitzmann, Martel et al. (2020), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \sqrt{\frac{6}{\text{fan\_mode}}}
    Also known as Siren initialization.

    Examples:
        >>> w = torch.empty(3, 5)
        >>> siren.init.siren_uniform_(w, mode='fan_in', c=6)

    :param tensor: an n-dimensional `torch.Tensor`
    :type tensor: torch.Tensor
    :param mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing
        ``'fan_in'`` preserves the magnitude of the variance of the weights in
        the forward pass. Choosing ``'fan_out'`` preserves the magnitudes in
        the backwards pass.s
    :type mode: str, optional
    :param c: value used to compute the bound. defaults to 6
    :type c: float, optional
    """
    fan = _calculate_correct_fan(tensor, mode)
    std = 1 / math.sqrt(fan)
    bound = math.sqrt(c) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.

        Example:
            >>> w = torch.tensor([3.14, 1.57])
            >>> Sine(w0=1)(w)
            torch.Tensor([0, 1])

        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                'input to forward() must be torch.xTensor')


class scaleSine(nn.Module):
    def __init__(self, w0: float = 1.0, s0: float = 1.0):
        """Sine activation function with w0 scaling support.

        Example:
            >>> w = torch.tensor([3.14, 1.57])
            >>> Sine(w0=1)(w)
            torch.Tensor([0, 1])

        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super(scaleSine, self).__init__()
        self.w0 = w0
        self.s0 = s0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return self.s0 * torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                'input to forward() must be torch.xTensor')


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class LipSIREN(nn.Module):
    def __init__(self, layers: List[int], in_features: int,
                 out_features: int,
                 w0: float = 1.0,
                 w0_initial: float = 30.0,
                 bias: bool = True,
                 initializer: str = 'siren',
                 c: float = 6):
        """
        SIREN model from the paper [Implicit Neural Representations with
        Periodic Activation Functions](https://arxiv.org/abs/2006.09661).

        :param layers: list of number of neurons in each hidden layer
        :type layers: List[int]
        :param in_features: number of input features
        :type in_features: int
        :param out_features: number of final output features
        :type out_features: int
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        :param w0_initial: `w0` of first layer. defaults to 30 (as used in the
            paper)
        :type w0_initial: float, optional
        :param bias: whether to use bias or not. defaults to
            True
        :type bias: bool, optional
        :param initializer: specifies which initializer to use. defaults to
            'siren'
        :type initializer: str, optional
        :param c: value used to compute the bound in the siren intializer.
            defaults to 6
        :type c: float, optional

        # References:
            -   [Implicit Neural Representations with Periodic Activation
                 Functions](https://arxiv.org/abs/2006.09661)
        """
        super(LipSIREN, self).__init__()
        self._check_params(layers)
        self.layers = [torchlip.SpectralLinear(in_features, layers[0], bias=bias), Sine(
            w0=w0_initial)]

        for index in range(len(layers) - 1):
            if True:
                self.layers.extend([
                    torchlip.SpectralLinear(layers[index], layers[index + 1], bias=bias),
                    scaleSine(w0=w0, s0=2)
                ])
            else:
                self.layers.extend([ResNet(
                    nn.Sequential(torchlip.SpectralLinear(layers[index], layers[index + 1], bias=bias),
                                  scaleSine(w0=w0, s0=1.2)))
                ])

        self.layers.append(torchlip.SpectralLinear(layers[-1], out_features, bias=bias, k_coef_lip=1))
        self.network = nn.Sequential(*self.layers)

        if initializer is not None and initializer == 'siren':
            for m in self.network.modules():
                if isinstance(m, torchlip.SpectralLinear):
                    if m.out_features < 20:  # last layer
                        torch.nn.init.uniform_(m.weight, a=-0.0001, b=0.0001)
                    else:
                        siren_uniform_(m.weight, mode='fan_in', c=c)
                else:
                    for n in m.modules():
                        if isinstance(n, torchlip.SpectralLinear):
                            siren_uniform_(n.weight, mode='fan_in', c=c)

        #self.network = self.network.to(torch.float32)


    @staticmethod
    def _check_params(layers):
        assert isinstance(layers, list), 'layers should be a list of ints'
        assert len(layers) >= 1, 'layers should not be empty'

    def forward(self, X):
        return self.network(X)


class SIREN(nn.Module):
    def __init__(self, layers: List[int], in_features: int,
                 out_features: int,
                 w0: float = 1.0,
                 w0_initial: float = 30.0,
                 bias: bool = True,
                 initializer: str = 'siren',
                 c: float = 6):
        """
        SIREN model from the paper [Implicit Neural Representations with
        Periodic Activation Functions](https://arxiv.org/abs/2006.09661).

        :param layers: list of number of neurons in each hidden layer
        :type layers: List[int]
        :param in_features: number of input features
        :type in_features: int
        :param out_features: number of final output features
        :type out_features: int
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        :param w0_initial: `w0` of first layer. defaults to 30 (as used in the
            paper)
        :type w0_initial: float, optional
        :param bias: whether to use bias or not. defaults to
            True
        :type bias: bool, optional
        :param initializer: specifies which initializer to use. defaults to
            'siren'
        :type initializer: str, optional
        :param c: value used to compute the bound in the siren intializer.
            defaults to 6
        :type c: float, optional

        # References:
            -   [Implicit Neural Representations with Periodic Activation
                 Functions](https://arxiv.org/abs/2006.09661)
        """
        super(SIREN, self).__init__()
        self._check_params(layers)
        self.layers = [nn.Linear(in_features, layers[0], bias=bias), Sine(
            w0=w0_initial)]

        for index in range(len(layers) - 1):
            if False:
                self.layers.extend([nn.Linear(layers[index], layers[index + 1], bias=bias),
                                    Sine(w0=w0)
                                    ])
            else:
                self.layers.extend([ResNet(
                    nn.Sequential(nn.Linear(layers[index], layers[index + 1], bias=bias),
                                  Sine(w0=w0)))
                ])

        self.layers.append(nn.Linear(layers[-1], out_features, bias=bias))
        self.network = nn.Sequential(*self.layers)

        if initializer is not None and initializer == 'siren':
            for m in self.network.modules():
                if isinstance(m, nn.Linear):
                    siren_uniform_(m.weight, mode='fan_in', c=c)

        '''if initializer is not None and initializer == 'siren':
            for m in self.network.modules():
                if isinstance(m, nn.Linear):
                    if m.out_features < 20:  # last layer
                        torch.nn.init.uniform_(m.weight, a=-0.0001, b=0.0001)
                    else:
                        siren_uniform_(m.weight, mode='fan_in', c=c)
                else:
                    for n in m.modules():
                        if isinstance(n, nn.Linear):
                            siren_uniform_(n.weight, mode='fan_in', c=c)'''
        #self.network = self.network.to(torch.float32)


    @staticmethod
    def _check_params(layers):
        assert isinstance(layers, list), 'layers should be a list of ints'
        assert len(layers) >= 1, 'layers should not be empty'

    def forward(self, X):
        return self.network(X)


# !/usr/bin/env python

class RealGaborLayer(nn.Module):
    '''
        Implicit representation with Gabor nonlinearity

        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0

        return torch.cos(omega) * torch.exp(-(scale ** 2))


class ComplexGaborLayer3D(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity with 2D activation function

        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=1,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        if self.is_first:
            dtype = torch.double
        else:
            dtype = torch.cdouble

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)

        # First Gaussian window
        self.scale_first = nn.Linear(in_features,
                                     out_features,
                                     bias=bias,
                                     dtype=dtype)

        # Second Gaussian window
        self.scale_second = nn.Linear(in_features,
                                      out_features,
                                      bias=bias,
                                      dtype=dtype)

        # trird Gaussian window
        self.scale_third = nn.Linear(in_features,
                                     out_features,
                                     bias=bias,
                                     dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)

        scale_x = lin
        scale_y = self.scale_second(input)
        scale_z = self.scale_third(input)

        freq_term = torch.exp(1j * self.omega_0 * lin)

        arg = scale_x.abs().square() + scale_y.abs().square() + scale_z.abs().square()
        gauss_term = torch.exp(-self.scale_0 * arg)

        return freq_term * gauss_term


class INR(nn.Module):
    def __init__(self, in_features, hidden_features,
                 hidden_layers,
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=1., scale=2,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()

        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer3D

        # Since complex numbers are two real numbers, reduce the number of
        # hidden parameters by 4
        hidden_features = int(hidden_features / np.sqrt(2))
        dtype = torch.cdouble
        self.complex = True
        self.wavelet = 'gabor'

        # Legacy parameter
        self.pos_encode = False

        self.net = []
        self.net.append(self.nonlin(in_features,
                                    hidden_features,
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features,
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        final_linear = nn.Linear(hidden_features,
                                 out_features,
                                 dtype=dtype)
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

        # for m in self.net.modules():
        #    if isinstance(m, nn.Linear):
        #        siren_uniform_(m.weight, mode='fan_in', c=6)

    def forward(self, coords):
        output = self.net(coords)

        if self.wavelet == 'gabor':
            return output.real

        return output


class Linearity(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = [nn.Linear(in_features, out_features, bias=True)]
        self.net = nn.Sequential(*self.layers)

    def forward(self, coords):
        return self.net(coords)
