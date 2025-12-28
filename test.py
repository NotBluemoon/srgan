import os
import time
import math
import csv
import torch
from torch.utils.tensorboard import SummaryWriter
#
# # -------------------------
# # Metrics
# # -------------------------
# @torch.no_grad()
# def psnr_torch(pred, target, data_range=2.0, eps=1e-8):
#     """
#     pred/target: tensors in [-1,1] if data_range=2.0
#     returns scalar PSNR (float)
#     """
#     mse = torch.mean((pred - target) ** 2).clamp_min(eps)
#     return float(20.0 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10.0 * torch.log10(mse))
#
# @torch.no_grad()
# def validate_sr(generator, val_loader, device="cuda", max_batches=None):
#     generator.eval()
#     psnr_sum = 0.0
#     n = 0
#
#     for i, (lr, hr) in enumerate(val_loader):
#         if max_batches is not None and i >= max_batches:
#             break
#
#         lr = lr.to(device, non_blocking=True)
#         hr = hr.to(device, non_blocking=True)
#
#         fake_hr = generator(lr)
#         psnr_sum += psnr_torch(fake_hr, hr, data_range=2.0)
#         n += 1
#
#     generator.train()
#     return (psnr_sum / max(n, 1))

print(torch.cuda.is_available())
