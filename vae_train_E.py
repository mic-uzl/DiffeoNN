import torch
import torch.optim as optim
from pathlib import Path
from torch import nn
from torchvision import transforms, utils, datasets
import os
from math import sqrt
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from unet_dataset import UNetDataset
from unet import UNet
#from superminvae_cnn import VAE, loss_function

model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/reg_energy.pt"
epoch_number = 50 
shuffle_data_loader = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
lr = 0.0001

data_folder = "/data"

#train data
train_image_dir = data_folder + "/empi_pairs/train/images"
train_mask_dir = data_folder + "/empi_pairs/train/masks"
train_pred_mask_dir = data_folder + "/empi_pairs/train/pred_masks"
train_trans_image_dir = data_folder + "/trans_data/train/images"
train_trans_mask_dir = data_folder + "/trans_data/train/masks"
train_trans_pred_mask_dir = data_folder + "/trans_data/train/pred_masks"
#validation data
val_image_dir = data_folder + "/empi_pairs/val/images"
val_mask_dir = data_folder + "/empi_pairs/val/masks"
val_pred_mask_dir = data_folder + "/empi_pairs/val/pred_masks"
val_trans_image_dir = data_folder + "/trans_data/val/images"
val_trans_mask_dir = data_folder + "/trans_data/val/masks"
val_trans_pred_mask_dir = data_folder + "/trans_data/val/pred_masks"

vae_val_data_dir = val_image_dir
vae_train_data_dir = train_image_dir

results_path="/results/vae_data"
os.makedirs(results_path, exist_ok=True)

vae_cnn_path = f"/results/model/vae_cnn_energy_model.pth"

# Image size -- adjust to your dataset images
img_size = 128  # Assuming images are 128x128
x_dim = img_size * img_size

#training data squares
train_dataset = UNetDataset(
    image_dir=vae_train_data_dir,
    transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor()])
    )

#validation data squares
val_dataset = UNetDataset(
    image_dir=vae_val_data_dir,
    transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor()])
    )


# VAE modells

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)  # mu
        self.fc32 = nn.Linear(h_dim2, z_dim)  # log_var
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

class VAE_cnn(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=128):
        super(VAE_cnn, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),  # 8 → 4
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),  # 4 → 2
            nn.ReLU(),
        )
        self.h_dim = h_dim
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),  # 2 -> 4
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),   # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),    # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),    # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1),     # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(8, image_channels, kernel_size=5, stride=2, padding=2, output_padding=1), # 64 -> 128
            nn.Sigmoid(),
        )

        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), 256, int(sqrt(self.h_dim/256)), int(sqrt(self.h_dim/256)))
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

def loss_function(recon_x, x, mu, log_var):
    eps=1e-8
    if torch.sum(recon_x)!=0 and torch.sum(x)!=0:
        recon_x = recon_x/torch.sum(recon_x)
        x = x/torch.sum(x)
    recon_x = torch.clamp(recon_x, eps, 1.0 - eps)
    x = torch.clamp(x, eps, 1.0 - eps)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def l2_loss(recon_x,x):
    L2 = torch.sum((x-recon_x)**2) # sum over pixels per image
    return L2 

def loss_function_l2(recon_x, x, mu, log_var):
    L2 = torch.sum((x-recon_x)**2)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # Total energy per image
    return L2 + 0.0001 * KLD#L2 + BCE+ KLD  # shape: [batch_size]

def mse_loss_function(recon_x, x, mu, log_var):
    # MSE loss summed over pixels and batch
    MSE = F.mse_loss(recon_x, x, reduction='none')
    MSE = MSE.view(MSE.size(0), -1).sum(dim=1)  # sum over pixels per image

    # KL divergence per image
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)  # sum over latent dim per image

    # Total energy per image
    return MSE + KLD  # shape: [batch_size]

def train_vae_cnn(train_image_dir=vae_train_data_dir, batch_size=batch_size, epochs=epoch_number, device=device, lr=lr, index=0):
    # Create datasets
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_data_loader, num_workers=0)

    # Model, optimizer
    vae = VAE_cnn(image_channels=1, h_dim = 1024, z_dim=128).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # Loss array epochs
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        vae.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # batch can be dict or tuple depending on your dataset
            x = batch["image"] if isinstance(batch, dict) else batch[0]
            x = x.view(-1, 1, img_size, img_size).to(device)

            optimizer.zero_grad()
            recon, mu, log_var = vae(x)
            #loss = loss_function(recon, x, mu, log_var)
            loss = loss_function_l2(recon, x, mu, log_var)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))
        # save training loss
        train_loss.append(total_loss / len(dataloader))

        print(f"Epoch {epoch+1} average loss: {total_loss / len(dataloader):.4f}")

        # Optional: save sample reconstructions
        vae.eval()
        #evaluate on validation set
        if val_dataset is not None:
            #vae.eval()
            val_loss_epoch = 0.0
            with torch.no_grad():
                for val_batch in tqdm(torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False), desc=f"Validation Epoch {epoch+1}"):
                    val_x = val_batch["image"] if isinstance(val_batch, dict) else val_batch[0]
                    val_x = val_x.to(device).view(-1, 1, img_size, img_size)
                    recon_val, mu_val, log_var_val = vae(val_x)
                    #val_loss_epoch += loss_function(recon_val, val_x, mu_val, log_var_val).item()
                    val_loss_epoch += loss_function_l2(recon_val, val_x, mu_val, log_var_val).item()
            avg_val_loss = val_loss_epoch / len(val_dataset)
            print(f"Validation loss: {avg_val_loss:.4f}")
            val_loss.append(avg_val_loss)
    # Save training and validation loss
    torch.save({
        'train_loss': train_loss,
        'val_loss': val_loss
    }, f"/results/vae_data/vae_cnn_{index}_energy_loss.pt")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    if val_loss:
        plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('VAE CNN Training and Validation Loss')
    plt.legend()
    plt.savefig(f"/results/vae_data/vae_cnn_{index}_energy_loss.png")
    plt.close()

    # Save the final model
    torch.save(vae.state_dict(), f"/results/model/vae_cnn_{index}_energy_model.pth")
    print(f"Training complete and model saved as vae_cnn_{index}_energy_model.pth")
    return train_loss, val_loss

if __name__ == "__main__":
    train_vae_cnn(lr=lr)