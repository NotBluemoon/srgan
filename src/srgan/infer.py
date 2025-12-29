import torch
from pathlib import Path
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T
from src.srgan.models.generator import Generator

def infer (opt):
    device = opt.device
    print(f'Running inference on device: {device}')

    srgan_path = 'models/srgan.pth'
    generator = Generator(opt).to(device)
    generator.load_state_dict(torch.load(srgan_path, map_location=device))
    generator.eval()

    in_dir = Path(opt.in_dir)
    last_two = Path(*in_dir.parts[-2:])
    dataset = last_two.parts[0]
    dataset_scale = last_two.parts[1]
    out_dir = Path("results") / dataset / dataset_scale
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(in_dir.glob("*LR.png")) +
                    list(in_dir.glob("*LR.jpg")) +
                    list(in_dir.glob("*LR.jpeg")))

    for img_path in tqdm(images, desc=f"SR {in_dir.name}"):
        img = Image.open(img_path).convert("RGB")
        lr = F.to_tensor(img).unsqueeze(0).to(device) # [0, 1]

        with torch.no_grad():
            sr = generator(lr)

        tqdm.write(f"lr min/max: {lr.min().item():.4f} {lr.max().item():.4f}")
        tqdm.write(f"sr min/max: {sr.min().item():.4f} {sr.max().item():.4f}")

        # denormalize sr
        sr = sr.squeeze(0).detach().cpu()
        sr = sr * 0.5 + 0.5
        sr = sr.clamp(0,1)

        save_name = img_path.name.replace("LR.png", "SR.png")
        save_path = out_dir / save_name
        T.ToPILImage()(sr).save(save_path)


