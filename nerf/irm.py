import numpy as np
import torch

from nerf.core.nn import NeRFModel, PositionalEncoder
from nerf.data.irm import IRM
from nerf.data.utils import BoundingBox
from PIL import Image


class NeRFIRM:
    def __init__(self, nerf: NeRFModel, phi_p: PositionalEncoder, phi_d: PositionalEncoder) -> None:
        self.nerf = nerf
        self.phi_p = phi_p
        self.phi_d = phi_d
        self.device = torch.device("cpu")

    def to(self, device: torch.device) -> "NeRFIRM":
        self.device = device
        self.nerf = self.nerf.to(device)
        self.phi_p = self.phi_p.to(device)
        self.phi_d = self.phi_d.to(device)
        return self

    def set_data(self, bbox: BoundingBox, divisions: int) -> None:
        self.bbox = bbox
        self.divisions = divisions
        self.irm = IRM(self.bbox, self.divisions, samples=1)

    @torch.inference_mode()
    def scan(self, size: int, samples: int, batch_size: int, path: str) -> None:
        self.irm.scan(self.nerf, self.phi_p, self.phi_d, samples, batch_size)

        voxels = self.irm.voxels.view(self.divisions, self.divisions, self.divisions)
        slices = torch.vstack((
            torch.hstack(tuple(voxels[i, :, :] for i in range(self.divisions))),
            torch.hstack(tuple(voxels[:, i, :] for i in range(self.divisions))),
            torch.hstack(tuple(voxels[:, :, i] for i in range(self.divisions))),
        ))

        img = Image.fromarray((slices.numpy() * 255.).astype(np.uint8))
        img = img.resize((img.width * size, img.height * size), Image.NEAREST)
        img.save(path)