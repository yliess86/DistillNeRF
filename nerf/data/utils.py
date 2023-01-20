import torch

from dataclasses import dataclass
from enum import auto, Enum
from torch import Tensor
from typing import Tuple, Union


@dataclass
class CameraPlanes:
    near: float
    far: float


@dataclass
class BoundingBox:
    bottom: float
    top: float
    left: float
    right: float
    back: float
    front: float


class Split(Enum):
    TRAIN: int = auto()
    VAL: int = auto()
    TEST: int = auto()


def meshgrid(H: int, W: int) -> Tensor:
    xs = torch.linspace(0, W - 1, W, dtype=torch.float32)
    ys = torch.linspace(0, H - 1, H, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid((xs, ys), indexing="xy"))
    return grid.permute(1, 2, 0)


def voxelgrid(H: int, W: int, D: int) -> Tensor:
    xs = torch.linspace(0, W - 1, W, dtype=torch.float32)
    ys = torch.linspace(0, H - 1, H, dtype=torch.float32)
    zs = torch.linspace(0, D - 1, D, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid((xs, ys, zs), indexing="xy"))
    return grid.permute(1, 2, 3, 0)


def generate_directions(H: int, W: int, focal: int) -> Tensor:
    i, j = meshgrid(H, W).unbind(-1)
    x = (i - .5 * W) / focal
    y = (j - .5 * H) / focal
    z = torch.ones_like(i)
    return torch.stack((x, -y, -z), dim=-1)


def projection(directions: Tensor, pose: Tensor) -> Tuple[Tensor, Tensor]:
    rays_direction = directions @ pose[:3, :3].T
    rays_direction /= torch.norm(rays_direction, dim=-1, keepdim=True)
    rays_origin = pose[:3, -1].expand_as(rays_direction)
    return rays_origin, rays_direction


@dataclass
class Ray:
    origin: Tensor
    direction: Tensor

    def __len__(self) -> int:
        return self.origin.size(0)

    def __getitem__(self, idx: Union[int, slice]) -> "Ray":
        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            return self.__class__(self.origin[start:stop:step], self.direction[start:stop:step])
        return self.__class__(self.origin[idx], self.direction[idx])
    
    def to(self, device: torch.device) -> "Ray":
        self.origin = self.origin.to(device)
        self.direction = self.direction.to(device)
        return self