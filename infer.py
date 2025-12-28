import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from project.models.generator import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = Generator().to(device)
generator.load_state_dict(torch.load("models/srgan_final.pth", map_location=device))
generator.eval()

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

def super_resolve(image_path, out_path):
    img = Image.open(image_path).convert("RGB")
    lr = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr = generator(lr)

    sr = (sr.squeeze(0).cpu() * 0.5 + 0.5).clamp(0,1)
    T.ToPILImage()(sr).save(out_path)

super_resolve("images/lr/cat.png", "images/sr/cat_sr.png")
