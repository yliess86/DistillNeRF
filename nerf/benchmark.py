from multiprocessing import BoundedSemaphore
from nerf.core.metrics import MSE, PSNR, SSIM
from nerf.core.nn import NeRFModel, PositionalEncoder
from nerf.core.render import Samples, render_volume
from nerf.data.blender import Blender
from nerf.data.utils import BoundingBox, Split
from time import time
from tqdm import tqdm

import numpy as np
import torch


class BenchmarkRecord:
    def __init__(
        self,
        fps: float,
        mse: float,
        psnr: float,
        ssim: float,
        n: int = 1,
        size: float = 0.,
    ) -> None:
        self.fps = fps
        self.mse = mse
        self.psnr = psnr
        self.ssim = ssim
        self.n = n
        self.size = size

    def __add__(self, other: "BenchmarkRecord") -> "BenchmarkRecord":
        self.fps += other.fps / self.n
        self.mse += other.mse / self.n
        self.psnr += other.psnr / self.n
        self.ssim += other.ssim / self.n
        return self

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.fps=:.2f}, {self.mse=:.2e}, {self.psnr=:.2f}, {self.ssim=:.2f}, {self.size=:.2f})"


class NeRFBenchmarker:
    def __init__(
        self,
        nerf: NeRFModel,
        phi_p: PositionalEncoder,
        phi_d: PositionalEncoder,
        samples: Samples,
    ) -> None:
        self.nerf = nerf
        self.phi_p = phi_p
        self.phi_d = phi_d

        self.mse = MSE()
        self.psnr = PSNR()
        self.ssim = SSIM()

        self.samples = samples
        self.device = torch.device("cpu")

    def to(self, device: torch.device) -> "NeRFBenchmarker":
        self.device = device
        
        self.nerf = self.nerf.to(self.device)
        self.phi_p = self.phi_p.to(self.device)
        self.phi_d = self.phi_d.to(self.device)

        self.mse = self.mse.to(self.device)
        self.psnr = self.psnr.to(self.device)
        self.ssim = self.ssim.to(self.device)
        
        return self

    def set_data(self, blender: str, scene: str, step: int, scale: float, batch_size: int) -> None:
        self.blender = blender
        self.scene = scene
        self.batch_size = batch_size
        
        self.test_set = Blender(self.blender, self.scene, Split.TEST, step, scale)
        self.planes = self.test_set.planes

    @torch.inference_mode()
    def _step_image(self, idx: int, perturb: bool = True, bbox: BoundingBox = None) -> BenchmarkRecord:
        modules = self.nerf, self.phi_p, self.phi_d

        target = torch.zeros((1, self.test_set.PIXELS, 3), device=self.device).float()
        predic = torch.zeros((1, self.test_set.PIXELS, 3), device=self.device).float()
        
        t0 = time()
        for start in range(0, self.test_set.PIXELS, self.batch_size):
            end     = min(start + self.batch_size, self.test_set.PIXELS)
            d_start = idx * self.test_set.PIXELS + start
            d_end   = idx * self.test_set.PIXELS + end

            batch = self.test_set[d_start:d_end].to(self.device)
            out = render_volume(*modules, batch.ray, self.planes, self.samples, perturb, bbox=bbox)
        
            target[:, start:end] = batch.pixel_color
            predic[:, start:end] = out.pixel_color
        t1 = time()

        target = target.view((1, self.test_set.H, self.test_set.W, 3)).permute((0, 3, 1, 2))
        predic = predic.view((1, self.test_set.H, self.test_set.W, 3)).permute((0, 3, 1, 2))

        fps   = 1. / (t1 - t0)
        mse   = self.mse(predic, target)
        psnr  = self.psnr(mse)
        ssim  = self.ssim(predic, target)

        return BenchmarkRecord(
            fps=fps,
            mse=mse.cpu().item(),
            psnr=psnr.cpu().item(),
            ssim=ssim.cpu().item(),
        )

    def benchmark(self, bbox: BoundedSemaphore = None) -> BenchmarkRecord:
        n_image = len(self.test_set) // self.test_set.PIXELS
        size = np.sum([np.product(p.size()) for p in self.nerf.parameters()]) * 32 * 1.25e-7
        bench = BenchmarkRecord(fps=0., mse=0., psnr=0., ssim=0., n=n_image, size=size)
        for idx in tqdm(range(n_image), desc="[NeRF] Benchmarking"):
            bench = bench + self._step_image(idx, bbox=bbox)
        print("[NeRF]", bench)
        return bench