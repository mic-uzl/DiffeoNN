import os
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from unet_dataset import UNetDataset
from torch.utils.data import DataLoader, TensorDataset
from random_transforms_rbf import rand_trans2d
import numpy as np
import matplotlib.pyplot as plt
import time


# training parameters
batch_size=16
epochs=100
lr=1e-4
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

data_folder = "/data"
#train data
train_image_dir = data_folder + "/empi_pairs/train/images"
train_trans_image_dir = data_folder + "/trans_data/train/images"
#validation data
val_image_dir = data_folder + "/empi_pairs/val/images"
val_trans_image_dir = data_folder + "/trans_data/val/images"



#loading the datasets
img_size = 128  # size of the images to be resized to

train_dataset = UNetDataset(
    image_dir=train_image_dir,
    transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor()])
    )
val_dataset = UNetDataset(
    image_dir=val_image_dir,
    transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor()])
    )

class convexnet(nn.Module):
    def __init__(self, n_channels=16, kernel_size=5, n_layers=5, convex=True, n_chan=1):
        super().__init__()
        self.convex = False
        self.n_layers = n_layers
        self.leaky_relu = nn.ReLU()

        # these layers can have arbitrary weights
        self.wxs = nn.ModuleList([nn.Conv2d(n_chan, n_channels, kernel_size=kernel_size, stride=1, padding=2, bias=True) for _ in range(self.n_layers+1)])

        # these layers should have non-negative weights
        self.wzs = nn.ModuleList([nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=2, bias=False) for _ in range(self.n_layers)])
        self.final_conv2d = nn.Conv2d(n_channels, 1, kernel_size=kernel_size, stride=1, padding=2, bias=False)

        self.initialize_weights()

    def initialize_weights(self, min_val=0, max_val=1e-3):
        for layer in range(self.n_layers):
            self.wzs[layer].weight.data = min_val + (max_val - min_val) * torch.rand_like(self.wzs[layer].weight.data)
        self.final_conv2d.weight.data = min_val + (max_val - min_val) * torch.rand_like(self.final_conv2d.weight.data)

    def clamp_weights(self):
        for i in range(self.n_layers):
            self.wzs[i].weight.data.clamp_(0)
        self.final_conv2d.weight.data.clamp_(0)

    def forward(self, x, grady=False):
        if self.convex:
            self.clamp_weights()

        z = self.leaky_relu(self.wxs[0](x))
        for layer_idx in range(self.n_layers):
            z = self.leaky_relu(self.wzs[layer_idx](z) + self.wxs[layer_idx+1](x))
        z = self.final_conv2d(z)
        net_output = z.view(z.shape[0], -1).mean(dim=1,keepdim=True)
        assert net_output.shape[0] == x.shape[0], f"{net_output.shape}, {x.shape[0]}"
        return net_output

# compute WGAN-GP loss
def loss_wgan(D, real, fake, mu = 10.0):
        """Calculates the gradient penalty loss for WGAN GP"""

        alpha = torch.Tensor(np.random.random((real.size(0), 1, 1, 1))).type_as(real)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        d_real = D(real)
        d_fake = D(fake)
        net_interpolates = D(interpolates)

        grad_outputs = (torch.Tensor(real.shape[0], 1).fill_(1.0).type_as(real)).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=net_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        loss_grad = (((gradients.norm(2, dim=1) - 1)) ** 2).mean()
        loss = d_real.mean() - d_fake.mean() + mu * loss_grad
        return loss

# compute WGAN loss without gradient penalty
def wgan_loss_only(D, real, fake):
    return D(real).mean() - D(fake).mean()


def train_discriminator_on_the_fly(fix_images, val_images, path_discriminator=None, results_dir=None, batch_size=batch_size, epochs=epochs, lr=lr, device=device):
    os.makedirs(results_dir, exist_ok=True)

    # create datasets
    fix_tensor = fix_images.detach().clone().float()

    train_dataset = TensorDataset(fix_tensor)

    val_tensor = val_images.detach().clone().float()
    val_dataset = TensorDataset(val_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # create discriminator model
    discriminator = convexnet().to(device).float()
    # optimise with Adam
    optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # create results storage
    results = {
        'epoch': [],
        'train_loss': [],
        'val_loss': []
    }

    # training loop
    for epoch in range(epochs):
        time1 = time.time()
        print(f"Epoch {epoch + 1}/{epochs}: starting time {time1}")
        discriminator.train()
        losses = []
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            y_fix_batch = batch[0].to(device)
            B = y_fix_batch.size(0)

            # Generate fake images via rand_trans2d
            y_mov_batch = []
            y_new_fix_batch = []
            for i in range(B):
                # create warped image
                img_np = y_fix_batch[i, 0].cpu().numpy()

                warped_img, _ = rand_trans2d(img_np, num_points=12, max_disp=0.12, rotate=True)

                # scale
                warped_img = warped_img / 255.0
                img_np_norm = img_np / 255.0

                # convert back to [0,255]
                warped_img = torch.tensor(warped_img * 255.0, dtype=torch.float32)
                img_np_tensor = torch.tensor(img_np_norm * 255.0, dtype=torch.float32)

                # add channel + batch dims
                y_mov_batch.append(warped_img.unsqueeze(0).unsqueeze(0))
                y_new_fix_batch.append(img_np_tensor.unsqueeze(0).unsqueeze(0))

            # concatenate batches
            y_mov_batch = torch.cat(y_mov_batch, dim=0).to(device)
            y_new_fix_batch = torch.cat(y_new_fix_batch, dim=0).to(device)

            # compute loss and backpropagate
            optimizer.zero_grad()
            loss = loss_wgan(discriminator, y_new_fix_batch, y_mov_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # validation loss
        discriminator.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                y_fix_batch = batch[0].to(device)
                B = y_fix_batch.size(0)

                # Generate fake images via rand_trans2d
                y_mov_batch = []
                y_new_fix_batch = []
                for i in range(B):
                    img_np = y_fix_batch[i, 0].cpu().numpy()

                    warped_img, _ = rand_trans2d(img_np, num_points=12, max_disp=0.12, rotate=True)

                    # scale
                    warped_img = warped_img / 255.0
                    img_np_norm = img_np / 255.0

                    # convert back to [0,255]
                    warped_img = torch.tensor(warped_img * 255.0, dtype=torch.float32)
                    img_np_tensor = torch.tensor(img_np_norm * 255.0, dtype=torch.float32)

                    # add channel + batch dims
                    y_mov_batch.append(warped_img.unsqueeze(0).unsqueeze(0))
                    y_new_fix_batch.append(img_np_tensor.unsqueeze(0).unsqueeze(0))

                # concatenate batches
                y_mov_batch = torch.cat(y_mov_batch, dim=0).to(device)
                y_new_fix_batch = torch.cat(y_new_fix_batch, dim=0).to(device)
                # compute loss
                loss = wgan_loss_only(discriminator, y_new_fix_batch, y_mov_batch)
                val_losses.append(loss.item())

        # compute average losses and print
        train_loss_avg = np.mean(losses)
        val_loss_avg = np.mean(val_losses)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss_avg:.4f}, Val Loss = {val_loss_avg:.4f}")

        # save losses for plotting
        results['epoch'].append(epoch + 1)
        results['train_loss'].append(train_loss_avg)
        results['val_loss'].append(val_loss_avg)

        time2 = time.time()
        print(f"Ending time {time2}")
        print(f"Time difference: {time2-time1}")

    # save model
    torch.save(discriminator.state_dict(), path_discriminator)
    print(f"Discriminator saved to {path_discriminator}")

    # store results to text file
    with open(results_dir + "/train_adversarial.txt", "w") as f:
        f.write("Epoch\tTrain Loss\tVal Loss\n")
        for epoch, train_loss, val_loss in zip(results['epoch'], results['train_loss'], results['val_loss']):
            f.write(f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\n")

    # Plotting the training and validation loss
    plt.figure(figsize=(20, 10))
    plt.plot(results['epoch'], results['train_loss'], label='Train Loss', color='blue')
    plt.plot(results['epoch'], results['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(results_dir + "/train_adversarial_loss.png")

# trains dicriminator for training images/transformed images by sampling the transformations (and then transforming the images) on the fly
def main_on_the_fly():
    # Create the datasets
    fix_images = torch.stack([train_dataset[i][0].unsqueeze(0) for i in range(len(train_dataset))])
    val_images = torch.stack([val_dataset[i][0].unsqueeze(0) for i in range(len(val_dataset))])

    # Define the save path for the discriminator model
    save_path = "/results/model/adversarial_discriminator_otf.pth"
    results_dir= "/results/adversarial_loss_otf"
    # Train the discriminator
    train_discriminator_on_the_fly(fix_images, val_images, path_discriminator=save_path, results_dir=results_dir, batch_size=batch_size, epochs=epochs, lr=lr, device=device)

if __name__ == '__main__':
    main_on_the_fly()