import os
import argparse
import time
import gc
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from vae_train_E import loss_function_l2, VAE_cnn
from adversarial_loss_train import convexnet

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import can_generators as generators
import can_losses as losses
import can_networks as networks 
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# vae model path
vae_model_path = "/results/model/vae_cnn_0_energy_model.pth"
# load VAE model
vae_cnn = VAE_cnn(image_channels=1, h_dim=1024, z_dim=128).to(device)

try:
    vae_cnn.load_state_dict(torch.load(vae_model_path, map_location=device))
except Exception as e:
    print("Failed to load model:", e)
# set to eval mode
vae_cnn.eval()

# load pretrained adversarial discriminator
discriminator = convexnet().to(device)
discriminator.load_state_dict(torch.load("/results/model/adversarial_discriminator_otf.pth", map_location=device))
discriminator.eval()

# define VAE loss function
def vae_loss(x):
    recon, mu, log_var = vae_cnn(x)
    energy = loss_function_l2(recon, x, mu, log_var)
    return energy

# define negative squared L2 norm loss
def negative_squared_l2_norm(img):
    return -0.000001 * torch.sum(img ** 2)

# define adversarial loss function
def adversarial_loss(y_warped):
    return discriminator(y_warped).mean()


# define border loss function
def get_edge_mask(shape, width=5, device=device):
    """
    Create an edge mask with 1s in a strip of size `width` around the border,
    and 0 elsewhere. Works for 2D or 3D.
    """
    if len(shape) == 2:
        H, W = shape
        mask = torch.zeros((H, W), device=device)
        mask[:width, :] = 1
        mask[-width:, :] = 1
        mask[:, :width] = 1
        mask[:, -width:] = 1
    elif len(shape) == 3:
        H, W, D = shape
        mask = torch.zeros((H, W, D), device=device)
        mask[:width, :, :] = 1
        mask[-width:, :, :] = 1
        mask[:, :width, :] = 1
        mask[:, -width:, :] = 1
        mask[:, :, :width] = 1
        mask[:, :, -width:] = 1
    else:
        raise ValueError("Shape must be 2D or 3D")
    return mask

def border_loss(flow, edge_mask):
    # reshape for broadcasting
    while edge_mask.ndim < flow.ndim:
        edge_mask = edge_mask.unsqueeze(0)
    return (flow * edge_mask).pow(2).mean()

# parse the commandline
parser = argparse.ArgumentParser()


# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of training epochs (default: 1)')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 10)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

args = parser.parse_args()

bidir = args.bidir
validate = []
np.random.seed(10)
torch.set_default_dtype(torch.float64)

def get_trans(fix_path=None, mov_path=None, adv_weight = 0.0005, vae_weight_var=1e-4):
    """
    Load fixed and moving images, resize them to 128x128, convert to grayscale,
    and normalize pixel values to [0, 1].
    """
    # Load and preprocess images
    t_fix_img = Image.open(fix_path).resize((128, 128))
    t_mov_img = Image.open(mov_path).resize((128, 128))
    t_fix_img = t_fix_img.convert("L")
    t_mov_img = t_mov_img.convert("L")
    fix_img = np.array(t_fix_img, dtype=np.float32)/255.0
    mov_img = np.array(t_mov_img, dtype=np.float32)/255.0
    
    # add batch and channel dimensions
    generator = generators.gencors(
        mov_img, fix_img, batch_size=args.batch_size, add_feat_axis=False,
        downsample_fac=args.downsize)

    # extract shape from sampled input
    inshape = fix_img.shape
    if len(inshape) > 3:
        inshape = inshape[2:]
    print('Inshape:', inshape)
    print('Device:', device)

    # prepare modelfolder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert np.mod(args.batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

    # enabling cudnn determinism appears to speed up training by a lot
    gc.collect()
    torch.backends.cudnn.deterministic = False  # not args.cudnn_nondet
    # set up model parameters
    siren_layers = [256, 256, 256]
    in_features = args.dimension
    out_features = args.dimension
    initializer = 'siren'
    w0 = 1
    c = 6

    image_loss_func = adversarial_loss

    # need two image loss functions if bidirectional
    if bidir:
        lloss = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        lloss = [image_loss_func]  
        weights = [1]
    llambda = 1

    # prepare deformation loss
    lloss += [losses.Grad('l2').loss]  
    weights += [llambda]

    # prepare orientation loss
    lloss += [losses.OlddetJac('l2').loss]
    weights += [args.orientation]

    # prepare VAE loss
    lloss += [vae_loss]
    weights += [vae_weight_var]

    # generate inputs (and true outputs) and convert them to tensors
    inputs = next(generator)
    inputs = inputs.double().to(device).requires_grad_()
    y_mov = torch.from_numpy(mov_img).double().to(device)
    y_fix = torch.from_numpy(fix_img).double().to(device)

    # initialize loss tracking variables
    first_loss = None
    final_loss = None

    old_deform, older_deform = None, None

    # training loops
    for epoch in range(args.initial_epoch, args.epochs):
        # initialize epoch tracking variables
        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []

        #define model
        w0_initial, lr, scale_rate = 1.596494373459478, 0.0038925738388887413, 0.0036162354478909003
        model = networks.NVelo(inshape, siren_layers, in_features, out_features, w0,
                               w0_initial, initializer, c, int_downsize=args.downsize,
                               int_steps=args.int_steps, bidir=True, scale_fac=scale_rate)
        print('Run model')

        if nb_gpus > 1:
            # use multiple GPUs via DataParallel
            model = torch.nn.DataParallel(model)
            model.save = model.module.save

        # prepare the model for training and send to device
        model.to(device)  
        model.train()
        # set optimizer
        loss_plot = []
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # sweep steps
        H, W = [dim for dim in inshape]
        for step in range(args.steps_per_epoch + 1):

            step_start_time = time.time()
            # run inputs through the model to produce a warped image and flow field
            y_warped, y_backward, new_locs, new_locs_backward, dvf, dvf_backward, mask_warped, mask_back, preint_flow \
                = model(inputs, source=y_mov, target=y_fix, mask=None, mask_b=None,
                       oldflow=old_deform, registration=True)
            
            # calculate total loss
            loss = 0
            loss_list = []

            if (y_warped.min().item()<0.0 and y_warped.max()>1.0):
                print("Warning: warped image min/max:",y_warped.min().item(),y_warped.max().item())
                print("Clamped warped image to be in [0.0, 1.0]...")
                y_warped = torch.clamp(y_warped, 0.0, 1.0)


            for n, loss_function in enumerate(lloss):
                if n == 0: # image loss index
                    curr_loss = loss_function(y_warped.reshape(1, 1, H, W)) * weights[n] * adv_weight
                elif n == 1: # bidirectional image loss index
                    curr_loss = loss_function(preint_flow) * weights[n]
                elif n == 2: # deformation loss index
                    curr_loss = loss_function(new_locs) * weights[n]
                elif n == 3:  # VAE loss index
                    curr_loss = loss_function(y_warped.reshape(1, 1, H, W)) * weights[n]
                elif n == 4:  # border loss index
                    edge_mask = get_edge_mask(model.inshape, width=5, device=preint_flow.device)
                    curr_loss = loss_function(preint_flow, edge_mask) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss


            # record first loss         
            if epoch == args.initial_epoch and step == 0:
                first_loss = loss.item()

            # record losses
            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())
            loss_plot.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)
            if (step % 500) == 0:
                # print epoch info
                epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
                step_info = 'step %d/%d' % (step, args.steps_per_epoch)
                # time and loss info
                time_info = '%.4f sec/step' % np.mean(epoch_step_time)
                losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
                loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
                print(' - '.join((epoch_info, step_info, time_info, loss_info)), flush=True)
                # calcuate SSIM and negative Jacobian metric
                H, W = [dim for dim in inshape]
                
        # detach tensor from graph
        old_deform = preint_flow.detach()
    # final loss:
    final_loss = loss.item()
    print(f"Loss before any epoch and step: {first_loss:.6f}")
    print(f"Final loss after last epoch and step: {final_loss:.6f}")
    
    # final model save
    plt.loglog(loss_plot)
    plt.show()

    # obtain numpy arrays for visualization
    warped_img = y_warped.detach().cpu().numpy()[0, 0, ...]
    fixed_img = y_fix.detach().cpu().numpy()
    moved_img = y_mov.detach().cpu().numpy()

    new_locs = new_locs.detach().cpu().numpy()
    new_locs_backward = new_locs_backward.detach().cpu().numpy()

    # convert to numpy if tensor
    if isinstance(new_locs, torch.Tensor):
        new_locs = new_locs.detach().cpu().numpy()

    # clean up
    del model
    gc.collect()
    return fixed_img, moved_img, warped_img, new_locs, new_locs_backward, dvf_backward, inshape, first_loss, final_loss




