import os
import random
from pathlib import Path

import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "DIV2K"

class DIV2KDataset(Dataset):
    def __init__(
            self,
            hr_dir: str = None,
            lr_dir: str = None,
            scale: int = 4,
            lr_crop: int = 24, # SRGAN paper uses 96x96 HR sub images, 96/4 = 24
            training: bool = True,
            ext: str = ".png",
    ):

        if training:
            if hr_dir is None:
                hr_dir = DATA_ROOT / "DIV2K_train_HR"
            if lr_dir is None:
                lr_dir = DATA_ROOT / "DIV2K_train_LR_bicubic" / f"X{scale}"
        else:
            if hr_dir is None:
                hr_dir = DATA_ROOT / "DIV2K_val_HR"
            if lr_dir is None:
                lr_dir = DATA_ROOT / "DIV2K_val_LR_bicubic" / f"X{scale}"

        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.lr_crop = lr_crop
        self.training = training

        # Sort by ID
        hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(ext)])
        self.ids = [os.path.splitext(f)[0] for f in hr_files]

        # Sanity check for HR-LR image mapping
        self.lr_paths = {}
        for _id in self.ids:
            cand1 = os.path.join(lr_dir, f"{_id}x{scale}{ext}")
            cand2 = os.path.join(lr_dir, f"{_id}{ext}")
            if os.path.exists(cand1):
                self.lr_paths[_id] = cand1
            elif os.path.exists(cand2):
                self.lr_paths[_id] = cand2
            else:
                raise FileNotFoundError(f"No LR file for id={_id} in {lr_dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        hr_path = os.path.join(self.hr_dir, f"{_id}.png")
        lr_path = self.lr_paths[_id]

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        if self.training:
            lr_w, lr_h = lr.size

            if lr_w < self.lr_crop or lr_h < self.lr_crop:
                raise ValueError(f"LR too small for crop: {lr.size}, crop={self.lr_crop}")

            x_lr = random.randint(0, lr_w - self.lr_crop)
            y_lr = random.randint(0, lr_h - self.lr_crop)

            # LR crop
            lr = lr.crop((x_lr, y_lr, x_lr + self.lr_crop, y_lr + self.lr_crop))

            # HR crop aligned
            hr_crop = self.lr_crop * self.scale
            x_hr = x_lr * self.scale
            y_hr = y_lr * self.scale
            hr = hr.crop((x_hr, y_hr, x_hr + hr_crop, y_hr + hr_crop))

        lr = F.to_tensor(lr) # LR remains at [0, 1]
        hr = F.to_tensor(hr)
        hr = hr * 2.0 - 1.0 # Scale HR to [-1, 1]
        return lr, hr