import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr_ssim(img_out, img_clean, data_range):
    if isinstance(img_out, torch.Tensor):
        img_out = img_out.squeeze().cpu().detach().numpy().astype(np.float32)
        img_clean = img_clean.squeeze().cpu().detach().numpy().astype(np.float32)
    psnr = peak_signal_noise_ratio(img_clean, img_out, data_range=data_range)
    ssim = structural_similarity(img_clean, img_out, data_range=data_range)
    return psnr, ssim






