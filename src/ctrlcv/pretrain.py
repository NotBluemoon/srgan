import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.div2k import DIV2KDataset


def pretrain_generator (opt, generator, PTG_PATH):
    # Load dataset
    train_data = DIV2KDataset(training=True)

    train_loader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # Pretrain SRResNet (generator) with MSE
    generator.train()
    pretrain_step = 0
    optimizer = torch.Optimizer.Adam(generator.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))

    checkpoint_step_interval = 1000

    if os.path.exists(PTG_PATH):
        checkpoint = torch.load(PTG_PATH)
        generator.load_state_dict(checkpoint["model"])
        pretrain_step = checkpoint.get("step", 0)
        print(f"Resumed from step {pretrain_step}")
    else:
        print("Starting fresh pretraining")

    # pretrain generator
    while pretrain_step < 100_000:
        for lr, hr in train_loader:
            lr = lr.cuda(non_blocking=True)
            hr = hr.cuda(non_blocking=True)

            pred = generator(lr)
            loss = F.mse_loss(pred, hr) # MSE-based according to SRGAN paper

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pretrain_step += 1

            if pretrain_step % checkpoint_step_interval == 0:
                os.makedirs(os.path.dirname(PTG_PATH), exist_ok=True)
                torch.save({
                    "model": generator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": pretrain_step,
                }, PTG_PATH)

            if pretrain_step >= 100_000:
                break

    torch.save({
        "model": generator.state_dict(),
    }, "generator_pretrained.pth")

    print("Saved final pretrained generator (SRResNet) to", PTG_PATH)