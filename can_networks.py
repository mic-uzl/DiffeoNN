
import numpy as np
import torch
import can_layers as layers
from can_modelio import LoadableModel, store_config_args
from can_sirenfork import SIREN

torch.set_default_dtype(torch.float64)

class NVelo(LoadableModel):
    """
    65 network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 sir_layers,
                 in_features,
                 out_features,
                 w0,
                 w0_initial,
                 initializer,
                 c,
                 int_steps=7,
                 int_downsize=1,
                 bidir=False,
                 use_probs=False,
                 scale_fac=0.005
                 ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            layers: layers of FCN list in form [width,width,...,width], or as a single integer.
                If None (default), the architecture is defined by the default config.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            siren_half_res: Skip the last upsampling. Requires that int_downsize=2.
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        self.inshape = inshape
        self.int_downsize = int_downsize

        # ensure correct dimensionality
        ndims = len(inshape)
        self.ndims = ndims
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        # configure core siren model
        self.siren_model = SIREN(sir_layers, in_features, out_features, w0, w0_initial,
                                 initializer=initializer, c=c)

        # self.siren_model = INR(in_features, hidden_features=512, hidden_layers=3, out_features=ndims)

        '''# init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))'''

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        self.resize = None
        self.scale_fac = scale_fac
        # resize to full res
        if int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims, size=inshape)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(np.ceil(dim / self.int_downsize)) for dim in inshape]
        self.integrate = layers.VecInt(inshape, int_steps) if int_steps >= 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, coords, source, mask, mask_b, target=False, registration=False, device='cuda', oldflow=None,
                mode='bilinear'):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        # torch.cat(source, dim=1)
        x = self.scale_fac * self.siren_model(coords)
        if len(self.inshape) == 3:
            H, W, Z = [int(np.ceil(dim / self.int_downsize)) for dim in self.inshape]
            # transform into flow field
            preint_flow = x.reshape((1, H, W, Z, 3)).permute(0, 4, 1, 2, 3)
        else:
            H, W = [int(np.ceil(dim / self.int_downsize)) for dim in self.inshape]
            # transform into flow field
            preint_flow = x.reshape((1, H, W, 2)).permute(0, 3, 1, 2)
            # scale flow for integration

        # resize to final resolution
        if self.fullsize:
            preint_flow = self.fullsize(preint_flow)

        # integrate to produce diffeomorphic warp
        if oldflow != None:
            if self.integrate:
                pos_flow = self.integrate(preint_flow + oldflow)
                neg_flow = self.integrate(
                    -preint_flow - oldflow) if self.bidir else None  # negate flow for bidirectional model

        else:
            if self.integrate:
                pos_flow = self.integrate(preint_flow)
                neg_flow = self.integrate(-preint_flow) if self.bidir else None  # negate flow for bidirectional model

        # warp image with flow field
        mask_warped, _ = self.transformer(mask, pos_flow, mode=mode) if mask != None else (None, None)
        y_warped, new_locs = self.transformer(source, pos_flow, mode=mode, padding='zeros' )

        y_backward, new_locs_back = self.transformer(target, neg_flow, mode=mode) if self.bidir else (None, None)
        mask_back, _ = self.transformer(mask_b, neg_flow, mode=mode, padding='zeros') if mask_b != None else (None, None)

        # return non-integrated flow field if training
        if self.bidir:
            return (
                y_warped, y_backward, new_locs, new_locs_back, pos_flow, neg_flow, mask_warped, mask_back, preint_flow)
        else:
            return (
                y_warped, new_locs, pos_flow, preint_flow, mask_warped)

