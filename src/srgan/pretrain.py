import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.srgan.helpers import load_ptg_checkpoint
from src.srgan.data.div2k import DIV2KDataset

from tqdm.auto import tqdm


def pretrain_generator (opt, generator):

    project_root = Path(__file__).resolve().parents[2]

    # Load dataset
    train_data = DIV2KDataset(training=True)

    train_loader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True
    )

    # Pretrain SRResNet (generator) with MSE
    generator.train()
    current_step = 0
    optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # check if there are any pretrained generator checkpoints
    checkpoint_ptg_dir = project_root/ 'outputs' / 'generator_pretrained'
    os.makedirs(checkpoint_ptg_dir, exist_ok=True)

    if any(checkpoint_ptg_dir.iterdir()):
        current_step = load_ptg_checkpoint(generator, optimizer, opt)
        print(f"Loading pretrained generator from checkpoint {current_step}")
    else:
        print("Pretraining generator from scratch...")

    # pretrain generator
    pbar = tqdm(total=opt.pretrain_steps - current_step, initial=0, desc="Pretraining generator", unit="step")

    # enable AMP
    use_amp = (opt.device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    while current_step < opt.pretrain_steps:
        for lr, hr in train_loader:
            lr = lr.to(opt.device, non_blocking=True)
            hr = hr.to(opt.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = generator(lr)
                loss = F.mse_loss(pred, hr) # MSE based according to paper

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=float(loss.detach()))

            if current_step % opt.checkpoint_interval == 0:
                torch.save({
                    "generator_state_dict": generator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": current_step,
                }, f"{checkpoint_ptg_dir}/ptg_{current_step:07d}.pth")

                tqdm.write(f"Saved pretrained generator checkpoint at step {current_step}")

            if current_step >= opt.pretrain_steps:
                break

    pbar.close()

    torch.save({
        "model": generator.state_dict(),
    }, f"{project_root}/outputs/generator_pretrained.pth")

    print("Saved final pretrained generator (SRResNet)",)