import os
import time
import math
import csv
import torch
from torch.utils.tensorboard import SummaryWriter

# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def psnr_torch(pred, target, data_range=2.0, eps=1e-8):
    """
    pred/target: tensors in [-1,1] if data_range=2.0
    returns scalar PSNR (float)
    """
    mse = torch.mean((pred - target) ** 2).clamp_min(eps)
    return float(20.0 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10.0 * torch.log10(mse))

@torch.no_grad()
def validate_sr(generator, val_loader, device="cuda", max_batches=None):
    generator.eval()
    psnr_sum = 0.0
    n = 0

    for i, (lr, hr) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        fake_hr = generator(lr)
        psnr_sum += psnr_torch(fake_hr, hr, data_range=2.0)
        n += 1

    generator.train()
    return (psnr_sum / max(n, 1))

# -------------------------
# Checkpoint
# -------------------------
def save_checkpoint(path, generator, discriminator, optimizer_G, optimizer_D, step, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "global_step": step,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)

def load_checkpoint_if_exists(path, generator, discriminator, optimizer_G, optimizer_D, device="cuda"):
    if not os.path.exists(path):
        return 0, False
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    optimizer_G.load_state_dict(ckpt["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(ckpt["optimizer_D_state_dict"])
    return int(ckpt.get("global_step", 0)), True

def optimizer_to_device(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

# -------------------------
# Train with Validation
# -------------------------
device = "cuda"
generator.to(device)
discriminator.to(device)

vgg = VGG54().to(device).eval()

total_steps = 200_000
switch_step = 100_000

validate_every = 2000      # run val every N steps
save_every = 5000          # save latest every N steps
val_max_batches = None     # set e.g. 50 if val is huge and you want speed

ckpt_latest = "models/srgan_latest.pth"
ckpt_best = "models/srgan_best_psnr.pth"

log_dir = "runs/srgan"
csv_path = "logs/srgan_train.csv"
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# CSV logging (append)
csv_new = not os.path.exists(csv_path)
csv_f = open(csv_path, "a", newline="", encoding="utf-8")
csv_w = csv.writer(csv_f)
if csv_new:
    csv_w.writerow(["time", "step", "loss_D", "loss_content", "loss_adv", "loss_G", "val_psnr", "lrG", "lrD"])

# Resume
gan_step, resumed = load_checkpoint_if_exists(
    ckpt_latest, generator, discriminator, optimizer_G, optimizer_D, device=device
)
optimizer_to_device(optimizer_G, device)
optimizer_to_device(optimizer_D, device)
print(f"Resume: {resumed} | starting step={gan_step}")

# If resuming past LR switch, enforce it immediately
if gan_step >= switch_step:
    for pg in optimizer_G.param_groups: pg["lr"] = 1e-5
    for pg in optimizer_D.param_groups: pg["lr"] = 1e-5
    print(f"LR already switched (resumed at step {gan_step})")

generator.train()
discriminator.train()

best_val_psnr = -1e9
t0 = time.time()

while gan_step < total_steps:
    for lr, hr in train_loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        # LR switch
        if gan_step == switch_step:
            for pg in optimizer_G.param_groups: pg["lr"] = 1e-5
            for pg in optimizer_D.param_groups: pg["lr"] = 1e-5
            print("Switched LR to 1e-5 at step", gan_step)

        # -------------------------
        # Train Discriminator
        # -------------------------
        optimizer_D.zero_grad(set_to_none=True)

        with torch.no_grad():
            fake_hr = generator(lr)

        pred_real = discriminator(hr)
        pred_fake = discriminator(fake_hr)

        real = torch.ones_like(pred_real)
        fake = torch.zeros_like(pred_fake)

        loss_D = 0.5 * (bce(pred_real, real) + bce(pred_fake, fake))
        loss_D.backward()
        optimizer_D.step()

        # -------------------------
        # Train Generator
        # -------------------------
        optimizer_G.zero_grad(set_to_none=True)

        fake_hr = generator(lr)
        pred_fake_for_G = discriminator(fake_hr)

        loss_content = content_loss_vgg(fake_hr, hr, vgg)
        loss_adv = bce(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
        loss_G = loss_content + 1e-3 * loss_adv

        loss_G.backward()
        optimizer_G.step()

        gan_step += 1

        # -------------------------
        # Train logging
        # -------------------------
        if gan_step % 100 == 0:
            lrG = optimizer_G.param_groups[0]["lr"]
            lrD = optimizer_D.param_groups[0]["lr"]

            writer.add_scalar("loss/D", loss_D.item(), gan_step)
            writer.add_scalar("loss/content", loss_content.item(), gan_step)
            writer.add_scalar("loss/adv", loss_adv.item(), gan_step)
            writer.add_scalar("loss/G_total", loss_G.item(), gan_step)
            writer.add_scalar("lr/G", lrG, gan_step)
            writer.add_scalar("lr/D", lrD, gan_step)
            writer.add_scalar("time/elapsed_sec", time.time() - t0, gan_step)

            print(
                f"[GAN] step {gan_step}/{total_steps} | "
                f"loss_D={loss_D.item():.4f} "
                f"loss_content={loss_content.item():.4f} "
                f"loss_adv={loss_adv.item():.4f} "
                f"loss_G={loss_G.item():.4f}"
            )

        # -------------------------
        # Validation
        # -------------------------
        val_psnr = None
        if gan_step % validate_every == 0:
            val_psnr = validate_sr(generator, val_loader, device=device, max_batches=val_max_batches)
            writer.add_scalar("val/PSNR", val_psnr, gan_step)
            print(f"[VAL] step {gan_step} | PSNR={val_psnr:.3f} dB")

            # Save best by validation PSNR
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                save_checkpoint(
                    ckpt_best, generator, discriminator, optimizer_G, optimizer_D, gan_step,
                    extra={"best_val_psnr": best_val_psnr}
                )
                print(f"Saved BEST checkpoint: {ckpt_best} (PSNR={best_val_psnr:.3f})")

        # -------------------------
        # Periodic checkpoint
        # -------------------------
        if gan_step % save_every == 0:
            save_checkpoint(ckpt_latest, generator, discriminator, optimizer_G, optimizer_D, gan_step)
            print("Saved checkpoint", ckpt_latest)

        # CSV log (every 100 steps; include val_psnr if it happened)
        if gan_step % 100 == 0:
            lrG = optimizer_G.param_groups[0]["lr"]
            lrD = optimizer_D.param_groups[0]["lr"]
            csv_w.writerow([
                time.time(),
                gan_step,
                float(loss_D.item()),
                float(loss_content.item()),
                float(loss_adv.item()),
                float(loss_G.item()),
                "" if val_psnr is None else float(val_psnr),
                float(lrG),
                float(lrD),
            ])
            csv_f.flush()

        if gan_step >= total_steps:
            break

# Final save + cleanup
save_checkpoint(ckpt_latest, generator, discriminator, optimizer_G, optimizer_D, gan_step)
writer.close()
csv_f.close()
print("Done. Final checkpoint:", ckpt_latest)
