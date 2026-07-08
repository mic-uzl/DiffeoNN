import os
from PIL import Image
from torch.utils.data import Dataset

def _load_filenames(folder, exts=(".png", ".jpg", ".jpeg", ".tif", ".bmp")):
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])

class UNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = _load_filenames(image_dir)
        self.mask_filenames = _load_filenames(mask_dir) if mask_dir else None
        self.transform = transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
            mask = Image.open(mask_path).convert("L")
            if self.mask_transform:
                mask = self.mask_transform(mask)
                mask = (mask > 0.5).long().squeeze(0)
            return image, mask
        else:
            return image

