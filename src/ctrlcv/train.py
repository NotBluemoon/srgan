"""
SRGAN based on the paper: https://doi.org/10.48550/arXiv.1609.04802
DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.div2k import DIV2KDataset
from models.discriminator import Discriminator
from models.feature_extractor import FeatureExtractor
from models.generator import Generator
from helpers import *

project_root = Path(__file__).resolve().parents[3]
print(project_root)

def train_srgan (opt):
    project_root = Path(__file__).resolve().parents[3]
    pretrained_gen_path = project_root/'models'/'generator_pretrained.pth'

    hr_shape = (opt.hr_height, opt.hr_width)

    generator = Generator().to(opt.device)
    discriminator = Discriminator().to(opt.device)
    feature_extractor = FeatureExtractor().to(opt.device).eval()

    criterion_content = torch.nn.MSELoss()

    # Load dataset
    train_data = DIV2KDataset(training=True)

    train_loader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # Train SRGAN
    if os.path.exists(pretrained_gen_path):
        checkpoint = torch.load(pretrained_gen_path, map_location=opt.device)
        generator.load_state_dict(checkpoint['model'])

    target_step = 200_000
    switch_step = 100_000
    current_step = 0

    generator.train()
    discriminator.train()

    optimizer_D = torch.Optimizer.Adam(generator.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))
    optimizer_G = torch.Optimizer.Adam(generator.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))

    while current_step < target_step:
        for lr, hr in train_loader:
            lr = lr.to(opt.device, non_blocking=True)
            hr = hr.to(opt.device, non_blocking=True)

        if current_step == switch_step:
            for pg in optimizer_D.param_groups: pg["lr"] = 1e-5
            print('Learning rate switched from 10^-4 to 10^-5')

            # ------------------------------------
            # Train generator with perceptual loss
            #-------------------------------------
            optimizer_G.zero_grad(set_to_none=True)

            gen_hr = generator(lr)

            # Content loss
            feature_extractor.eval()
            gen_features = FeatureExtractor(gen_hr)
            real_features = FeatureExtractor(hr)
            content_loss = criterion_content(gen_features, real_features.detach())
            content_loss = content_loss * 0.006 # Rescaled with a factor of 0.006 with reference to the SRGAN paper

            # Adversarial loss
            adversarial_loss = F.binary_cross_entropy(discriminator(gen_hr), torch.ones_like(discriminator(gen_hr)), reduction='sum')

            # Perceptual loss
            loss_G = content_loss + 1e-3 * adversarial_loss
            loss_G.backward()

            optimizer_G.step()

            current_step += 1

            if current_step % 100 == 0:
                print(f"[GAN] step {current_step}/{target_step} | "
                      f"loss_D={loss_D.item():.4f} "
                      f"content_loss={content_loss.item():.4f} "
                      f"loss_adv={adversarial_loss.item():.4f} "
                      f"loss_G={loss_G.item():.4f}")

            # periodic checkpoint
            if current_step % 5000 == 0:
                os.makedirs("../../models", exist_ok=True)
                torch.save({
                    "G": generator.state_dict(),
                    "D": discriminator.state_dict(),
                    "opt_G": optimizer_G.state_dict(),
                    "opt_D": optimizer_D.state_dict(),
                    "step": current_step,
                }, "models/srgan_latest.pth")
                print("Saved checkpoint models/srgan_latest.pth")

            # ---------------------------------
            # Train discriminator with BCE Loss
            # ---------------------------------
            optimizer_D.zero_grad(set_to_none=True)

            with torch.no_grad():
                gen_hr = generator(lr)  # detach via no_grad

            pred_real = discriminator(hr)
            pred_fake = discriminator(gen_hr)

            real = torch.ones_like(pred_real)
            fake = torch.zeros_like(pred_fake)

            loss_D = (loss_D(pred_real, real) + loss_D(pred_fake, fake)) / 2
            loss_D.backward()
            optimizer_D.step()

            if current_step >= target_step:
                break







