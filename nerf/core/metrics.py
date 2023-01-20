from torch import Tensor
from torch.nn import Module, MSELoss, Parameter

import numpy as np
import torch
import torch.nn.functional as F


def gaussian(window_size: int, sigma: float) -> Tensor:
    gauss = Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


def _ssim(img1: Tensor, img2: Tensor, window: Tensor, window_size: int, channel: int) -> Tensor:
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2  = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


class MSE(Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = MSELoss()
        
    @torch.inference_mode()
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.mse(a, b)


class PSNR(Module):
    def __init__(self) -> None:
        super().__init__()
        
    @torch.inference_mode()
    def forward(self, mse: Tensor) -> Tensor:
        return -10 * torch.log10(mse)


class SSIM(Module):
    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = window_size
        self.window = Parameter(create_window(window_size, 3))

    @torch.inference_mode()
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return _ssim(a, b, self.window, self.window_size, 3)