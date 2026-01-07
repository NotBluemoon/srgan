"""
All test metrics are calculated on the y-channel of center-cropped, removal of 4-pixel wide strip of each border
"""


from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def compute_psnr (sr, hr, eps: float = 1e-10):
    return peak_signal_noise_ratio(sr, hr, data_range=1.0)


def compute_ssim (sr, hr):
    return structural_similarity(sr, hr, data_range=1.0)


# def _rgb_to_y (x):
#     """
#     Convert RGB tensor in [0,1] to luminance (Y) using BT.601
#     """
#     r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
#     r255, g255, b255 = r * 255.0, g * 255.0, b * 255.0
#     y = (16.0 + 65.481 * r255 + 128.553 * g255 + 24.966 * b255) / 255.0
#
#     return y


def shave_border(x, border=4):
    if border <= 0:
        return x

    return x[..., border:-border, border:-border]


def centre_crop (sr, hr):
    # sr, hr: (N,C,H,W)
    h_sr, w_sr = sr.shape[-2:]
    h_hr, w_hr = hr.shape[-2:]

    h = min(h_sr, h_hr)
    w = min(w_sr, w_hr)

    def crop(x):
        h_x, w_x = x.shape[-2:]
        top = (h_x - h) // 2
        left = (w_x - w) // 2
        return x[..., top:top+h, left:left+w]

    return crop(sr), crop(hr)


def test_srgan (opt):

    project_root = Path(__file__).resolve().parents[2]

    # TODO change this to be more flexible
    hr_dir = project_root / 'data' / 'Set14' / 'image_SRF_4'
    sr_dir = Path(opt.test_dir)

    psnr_array = []
    ssim_array = []

    log = ""

    for sr_img_path in sr_dir.iterdir():
        sr_img_name = sr_img_path.name
        hr_img_name = sr_img_name.replace("SR.png", "HR.png")
        hr_img_path = hr_dir / hr_img_name

        if hr_img_path.exists():
            sr = Image.open(sr_img_path).convert('RGB')
            sr = TF.to_tensor(sr).unsqueeze(0) # [0, 1]

            hr = Image.open(hr_img_path).convert('RGB')
            hr = TF.to_tensor(hr).unsqueeze(0) # [0, 1]

            sr, hr = centre_crop(sr, hr)

            sr = shave_border(sr)
            hr = shave_border(hr)

            sr = sr.squeeze(0)
            hr = hr.squeeze(0)

            sr_y = 16/255 + (65.481*sr[0] + 128.553*sr[1] + 24.966*sr[2]) / 255
            hr_y = 16/255 + (65.481*hr[0] + 128.553*hr[1] + 24.966*hr[2]) / 255

            sr_y = sr_y.cpu().numpy()
            hr_y = hr_y.cpu().numpy()

            psnr = compute_psnr(sr_y, hr_y)
            ssim = compute_ssim(sr_y, hr_y)

            psnr_array.append(psnr)
            ssim_array.append(ssim)

            log += f"{sr_img_name} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}\n"

    # print(log)

    if psnr_array:
        avg_psnr = sum(psnr_array) / len(psnr_array)
        avg_ssim = sum(ssim_array) / len(ssim_array)
        print(f"[RESULT] Avg PSNR: {avg_psnr:.2f} dB")
        print(f"[RESULT] Avg SSIM: {avg_ssim:.4f}\n")

    print('SRGAN test finished\n')








