"""
SRGAN based on the paper: https://doi.org/10.48550/arXiv.1609.04802
DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.srgan.helpers import load_srgan_checkpoint, init_log, delete_experiment_artifacts
from src.srgan.pretrain import pretrain_generator
from src.srgan.data.div2k import DIV2KDataset
from src.srgan.models.discriminator import Discriminator
from src.srgan.models.feature_extractor import FeatureExtractor
from src.srgan.models.generator import Generator


def train_srgan (opt):
    print(f'Training SRGAN with {opt.device}')

    if not opt.resume:
        delete_experiment_artifacts()

    project_root = Path(__file__).resolve().parents[2]
    pretrained_gen_path = project_root/'outputs'/'generator_pretrained.pth'

    wb, ws, log_path = init_log ()

    generator = Generator(opt).to(opt.device)
    discriminator = Discriminator().to(opt.device)
    feature_extractor = FeatureExtractor().to(opt.device).eval()

    criterion_content = torch.nn.MSELoss()
    criterion_discriminator = torch.nn.BCELoss()

    train_data = DIV2KDataset(training=True)

    train_loader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True
    )

    # Check if pretrained generator exists
    if os.path.exists(pretrained_gen_path):
        checkpoint = torch.load(pretrained_gen_path, map_location=opt.device)
        generator.load_state_dict(checkpoint['model'])
        print('Loading pretrained generator...')
    else:
        print('Pretraining generator...')
        pretrain_generator(opt, generator)

    target_step = opt.train_steps
    switch_step = target_step//2
    current_step = 0

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))

    # Check if final SRGAN exists, if not then check for checkpoints
    final_srgan_path = project_root / 'models' / 'srgan.pth'
    if os.path.exists(final_srgan_path):
        print("SRGAN already trained")
        return 0

    checkpoint_srgan_path = project_root / 'outputs' / 'srgan'
    os.makedirs(checkpoint_srgan_path, exist_ok=True)
    if any(checkpoint_srgan_path.iterdir()):
        print("Loading SRGAN from checkpoint...")
        current_step = load_srgan_checkpoint(generator, discriminator, optimizer_G, optimizer_D) # update current step
    else:
        print("Training SRGAN from scratch...")

    generator.train()
    discriminator.train()

    pbar = tqdm(total=opt.train_steps - current_step, initial=0, desc="Training SRGAN", unit="step")

    while current_step < target_step:
        for lr, hr in train_loader:
            lr = lr.to(opt.device, non_blocking=True)
            hr = hr.to(opt.device, non_blocking=True)

            if current_step == switch_step:
                for pg in optimizer_G.param_groups: pg["lr"] = 1e-5
                for pg in optimizer_D.param_groups: pg["lr"] = 1e-5
                tqdm.write('Learning rate switched from 10^-4 to 10^-5')

            # ------------------------------------
            # Train generator with perceptual loss
            #-------------------------------------
            optimizer_G.zero_grad(set_to_none=True)

            gen_hr = generator(lr)

            # Content loss
            feature_extractor.eval()
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(hr)
            content_loss = criterion_content(gen_features, real_features.detach())
            content_loss = content_loss * 0.006 # Scaled according to SRGAN paper

            # Adversarial loss
            pred_gen = discriminator(gen_hr)
            adversarial_loss = F.binary_cross_entropy(pred_gen, torch.ones_like(pred_gen), reduction='sum')

            # Perceptual loss
            loss_G = content_loss + 1e-3 * adversarial_loss
            loss_G.backward()

            optimizer_G.step()

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

            loss_D = (criterion_discriminator(pred_real, real) + criterion_discriminator(pred_fake, fake)) / 2
            loss_D.backward()
            optimizer_D.step()

            current_step += 1
            pbar.update(1)
            pbar.set_postfix(loss_G=float(loss_G.detach()), loss_D=float(loss_D.detach()))

            # -------------------
            # Periodic checkpoint
            # -------------------
            if current_step % opt.checkpoint_interval == 0:

                tqdm.write(f"[SRGAN] step {current_step}/{target_step} | "
                      f"loss_D={loss_D.item():.4f} "
                      f"content_loss={content_loss.item():.4f} "
                      f"adversarial_loss={adversarial_loss.item():.4f} "
                      f"loss_G={loss_G.item():.4f}")

                ws.append([
                    int(current_step),
                    float(loss_D.item()),
                    float(content_loss.item()),
                    float(adversarial_loss.item()),
                    float(loss_G.item()),
                ])

                wb.save(log_path)

                experiment_srgan_dir = project_root / 'outputs' / 'srgan'
                os.makedirs(experiment_srgan_dir, exist_ok=True)

                torch.save({
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D": optimizer_D.state_dict(),
                    "step": current_step,
                }, f"{experiment_srgan_dir}/srgan_step{current_step:07d}.pth")

                tqdm.write(f"Saved SRGAN checkpoint at step {current_step}")

            if current_step >= target_step:
                break

    pbar.close()

    os.makedirs(project_root / 'models', exist_ok=True)
    torch.save(generator.state_dict(), "models/srgan.pth")
    print('SRGAN training finished.')

    return None








