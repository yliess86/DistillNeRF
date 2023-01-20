import numpy as np
import torch

from moviepy.editor import ImageSequenceClip
from multiprocessing import cpu_count

from torch._C import device
from nerf.core.nn import NeRFModel, PositionalEncoder
from nerf.core.render import render_volume, Samples
from nerf.data.turntable import TurnTable
from nerf.data.utils import BoundingBox, CameraPlanes
from torch.cuda.amp import autocast
from tqdm import tqdm


class NeRFInferer:
    def __init__(self, nerf: NeRFModel, phi_p: PositionalEncoder, phi_d: PositionalEncoder) -> None:
        self.nerf = nerf
        self.phi_p = phi_p
        self.phi_d = phi_d

        self.device = torch.device("cpu")

    def to(self, device: torch.device) -> "NeRFInferer":
        self.device = device
        self.nerf = self.nerf.to(device)
        self.phi_p = self.phi_p.to(device)
        self.phi_d = self.phi_d.to(device)
        return self

    def set_data(self, dataset: TurnTable, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.loader = dataset.loader(self.batch_size, cpu_count())

    @torch.inference_mode()
    def render(
        self,
        fps: int,
        planes: CameraPlanes,
        samples: Samples,
        path: str,
        bbox: BoundingBox = None,
    ) -> None:
        FRAMES = self.dataset.frames
        PIXELS = self.dataset.PIXELS
        SIZE = self.dataset.SIZE

        modules = self.nerf, self.phi_p, self.phi_d
        gif = torch.zeros((FRAMES * PIXELS, 3), dtype=torch.float32)
        for r, rays in enumerate(tqdm(self.loader, desc="[NeRF] Rendering")):
            s = r * self.batch_size
            e = min(s + self.batch_size, gif.size(0))
            
            rays = rays.to(self.device)
            with autocast():
                out = render_volume(*modules, rays, planes, samples, False, bbox=bbox)
            gif[s:e] = out.pixel_color.cpu()

        gif = gif.view(FRAMES, *SIZE, 3)
        gif = gif.clip(0., 1.) * 255.
        gif = gif.numpy().astype(np.uint8)
        gif = ImageSequenceClip(list(gif), durations=[1. / fps] * FRAMES)
        gif.write_gif(path, fps=fps)