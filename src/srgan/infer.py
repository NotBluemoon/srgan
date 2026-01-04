"""
generator_pretrained.pth ：: "model" - generator.state_dict()

ptg_{step}.pth :    "generator_state_dict" - generator.state_dict(),
                    "optimizer" - optimizer.state_dict(),
                    "step" - current_step,

srgan.pth : no keys

srgan_step{step}.pth :  "generator_state_dict": generator.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "optimizer_G": optimizer_G.state_dict(),
                        "optimizer_D": optimizer_D.state_dict(),
                        "step": current_step,
"""


from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

from src.srgan.models.generator import Generator
from src.srgan.helpers import load_infer_model


def infer (opt):
    device = opt.device
    generator = Generator(opt).to(device)

    print(f'Running inference on device: {device}')

    # Load model based on input
    infer_model = opt.infer_model.lower()
    if infer_model == "srgan":
        if opt.infer_step == 'final':
            in_path_model = Path('models/srgan.pth')
            out_dir = Path('results/Set14/image_SRF_4_SRGAN')
            load_infer_model(generator, in_path_model, device=device, key=None)
        else:
            result_step = f"{int(opt.infer_step):07d}"
            in_path_model = Path(f'outputs/srgan/srgan_step{result_step}.pth')
            out_dir = Path(f'results/Set14/image_SRF_4_SRGAN_step{result_step}')
            load_infer_model(generator, in_path_model, device=device, key="generator_state_dict")

        if not in_path_model.exists():
            raise Exception(f"Model does not exist for step {opt.infer_step}")

    elif infer_model == "srresnet":
        if opt.infer_step == 'final':
            path = Path('outputs/generator_pretrained.pth')
            out_dir = Path('results/Set14/image_SRF_4_SRResNet')
            load_infer_model(generator, path, device=device, key="model")
        else:
            result_step = f"{int(opt.infer_step):07d}"
            path = Path(f'outputs/generator_pretrained/ptg_{result_step}.pth')
            out_dir = Path(f'results/Set14/image_SRF_4_SRResNet_step{result_step}')
            load_infer_model(generator, path, device=device, key="generator_state_dict")

    else:
        raise Exception("Invalid model, use either 'SRGAN' or 'SRResNet'")


    generator.eval()

    in_dir_img = Path('data/Set14/image_SRF_4')
    # last_two = Path(*in_dir_img.parts[-2:])
    # dataset = last_two.parts[0]
    # dataset_scale = last_two.parts[1]
    # out_dir = Path("results") / dataset / dataset_scale
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(in_dir_img.glob("*LR.png")) +
                    list(in_dir_img.glob("*LR.jpg")) +
                    list(in_dir_img.glob("*LR.jpeg")))

    for img_path in tqdm(images, desc=f"SR {in_dir_img.name}"):
        img = Image.open(img_path).convert("RGB")
        lr = F.to_tensor(img).unsqueeze(0).to(device) # image range [0, 1]

        with torch.no_grad():
            sr = generator(lr) # image range [-1, 1]

        # tqdm.write(f"lr min/max: {lr.min().item():.4f} {lr.max().item():.4f}")
        # tqdm.write(f"sr min/max: {sr.min().item():.4f} {sr.max().item():.4f}")

        # [-1, 1] -> [0, 1]
        sr = sr.squeeze(0).detach().cpu()
        sr = sr * 0.5 + 0.5
        sr = sr.clamp(0,1)

        save_name = img_path.name.replace("LR.png", "SR.png")
        save_path = out_dir / save_name
        T.ToPILImage('RGB')(sr).save(save_path)


