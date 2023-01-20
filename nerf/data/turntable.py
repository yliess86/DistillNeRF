import matplotlib.pyplot as plt
import numpy as np
import torch

from nerf.data.utils import generate_directions, projection, Ray
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union
from tqdm import tqdm


class TurnTable(Dataset):
    def __init__(self, H: int, W: int, focal: float, frames: int, radius: float) -> None:
        super().__init__()
        self.SIZE = self.H, self.W = H, W
        self.PIXELS = self.H * self.W

        self.focal = focal
        self.frames = frames
        self.radius = radius
        
        def turn(theta: float, phi: float, radius: float) -> Tensor:
            theta = torch.tensor([
                [np.cos(theta), 0., -np.sin(theta), 0.],
                [           0., 1.,             0., 0.],
                [np.sin(theta), 0.,  np.cos(theta), 0.],
                [           0., 0.,             0., 1.],
            ], dtype=torch.float32)
            
            phi = torch.tensor([
                [1.,          0.,           0., 0.],
                [0., np.cos(phi), -np.sin(phi), 0.],
                [0., np.sin(phi),  np.cos(phi), 0.],
                [0.,          0.,           0., 1.],
            ], dtype=torch.float32)
            
            radius = torch.tensor([
                [1., 0., 0.,            0.],
                [0., 1., 0., 0.02 * radius],
                [0., 0., 1.,        radius],
                [0., 0., 0.,            1.],
            ], dtype=torch.float32)
            
            return torch.tensor([
                [-1., 0., 0., 0.],
                [ 0., 0., 1., 0.],
                [ 0., 1., 0., 0.],
                [ 0., 0., 0., 1.],
            ], dtype=torch.float32) @ theta @ phi @ radius

        thetas = np.linspace(0., 2. * np.pi, self.frames + 1)[:-1]
        phi = -np.pi / 6.

        directions = generate_directions(*self.SIZE, self.focal)
        poses = torch.stack([turn(theta, phi, self.radius) for theta in thetas], dim=0)

        rays_origin = torch.zeros((poses.size(0), *self.SIZE, 3), dtype=torch.float32)
        rays_direction = torch.zeros((poses.size(0), *self.SIZE, 3), dtype=torch.float32)
        for p, pose in enumerate(tqdm(poses, desc=f"[TurnTable] Projecting")):
            rays_origin[p], rays_direction[p] = projection(directions, pose)

        rays_origin = rays_origin.view(-1, 3)
        rays_direction = rays_direction.view(-1, 3)
        self.rays = Ray(rays_origin, rays_direction)

    def __len__(self) -> int:
        return len(self.rays)

    def __getitem__(self, idx: Union[int, slice]) -> Ray:
        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            return self.rays[start:stop:step]
        return self.rays[idx]

    def collate_fn(self, batch: List[Ray]) -> Ray:
        origin = torch.stack([ray.origin for ray in batch], dim=0)
        direction = torch.stack([ray.direction for ray in batch], dim=0)
        return Ray(origin, direction)

    def loader(self, batch_size: int, jobs: int) -> DataLoader:
        return DataLoader(
            self,
            batch_size,
            shuffle=False,
            num_workers=jobs,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def plot(self, figsize: Tuple[int, int] = (8, 8)) -> None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        
        ax.set_title("Cameras")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        CORNERS = TL, TR, BL, BR = 0, self.W - 1, -self.W, -1
        FRAME = TL, TR, BR, BL, TL

        for s in range(0, len(self), self.PIXELS):
            e = s + self.PIXELS
            
            rays = self[s:e]
            ro, rd = rays.origin, rays.direction

            ax.scatter(ro[0, 0], ro[0, 1], ro[0, 2], c="blue")
            
            ax.plot(
                [ro[0, 0] + rd[p, 0] for p in FRAME],
                [ro[0, 1] + rd[p, 1] for p in FRAME],
                [ro[0, 2] + rd[p, 2] for p in FRAME],
                c="blue",
            )

            for p in CORNERS:
                ax.plot(
                    [ro[0, 0], ro[0, 0] + rd[p, 0]],
                    [ro[0, 1], ro[0, 1] + rd[p, 1]],
                    [ro[0, 2], ro[0, 2] + rd[p, 2]],
                    c="blue",
                )

        limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
        ax.set_box_aspect(np.ptp(limits, axis=1))
        
        for axis in "xyz":
            a, b = map(int, getattr(ax, f"get_{axis}lim")())
            getattr(ax, f"set_{axis}ticks")([a, (a + b) // 2, b])
        
        fig.canvas.draw()
        plt.show()