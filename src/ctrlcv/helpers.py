import os
import csv
import time
import torch
from torch.utils.tensorboard import SummaryWriter

def save_srgan_checkpoint (path, generator, discriminator, optimizer_G, optimizer_D, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    srgan = {
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),

        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),

        "step": step,
    }

    torch.save(srgan, path)

def load_srgan_checkpoint (path, generator, discriminator, optimizer_G=None, optimizer_D=None, device="cuda"):
    if not os.path.exists(path):
        return 0, False

    checkpoint = torch.load(path, map_location=device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    # Optimizers are optional (useful if you want to warm-start weights only)
    if optimizer_G is not None and "optimizer_G" in checkpoint:
        optimizer_G.load_state_dict(checkpoint["optimizer_G"])
    if optimizer_D is not None and "optimizer_D" in checkpoint:
        optimizer_D.load_state_dict(checkpoint["optimizer_D"])

    step = int(checkpoint.get("step", 0))

    return step, True