import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataclasses import dataclass
from nerf.data.utils import CameraPlanes, generate_directions, projection, Ray, Split
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union
from tqdm import tqdm


@dataclass
class BlenderData:
    pixel_color: Tensor
    ray: Ray

    def __len__(self) -> int:
        return self.pixel_color.size(0)
    
    def __getitem__(self, idx: Union[int, slice]) -> "BlenderData":
        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            return self.__class__(self.pixel_color[start:stop:step], self.ray[start:stop:step])
        return self.__class__(self.pixel_color[idx], self.ray[idx])

    def to(self, device: torch.device) -> "BlenderData":
        self.pixel_color = self.pixel_color.to(device)
        self.ray = self.ray.to(device)
        return self


class Blender(Dataset):
    def __init__(self, root: str, scene: str, split: Split, step: int, scale: int) -> None:
        super().__init__()
        self.root = root
        self.scene = scene
        self.split = split
        self.step = step
        self.scale = scale

        path = os.path.join(self.root, self.scene)
        
        meta_path = os.path.join(path, f"transforms_{split.name.lower()}.json")
        with open(meta_path, "r") as meta_fp: meta = json.load(meta_fp)

        self.H = self.W = int(800 * self.scale)
        self.PIXELS = self.H * self.W
        self.SIZE = self.H, self.W

        self.planes = CameraPlanes(2., 6.)
        self.focal = .5 * self.W / np.tan(.5 * float(meta["camera_angle_x"]))

        frames = meta["frames"][::step]
        renders = torch.zeros((len(frames), *self.SIZE, 3), dtype=torch.float32)
        poses = torch.zeros((len(frames), 4, 4), dtype=torch.float32)
        
        for f, frame in enumerate(tqdm(frames, desc=f"[Blender|{self.split.name.capitalize()}] Loading")):
            render = Image.open(os.path.join(path, f"{frame['file_path']}.png"))
            render = render.resize(self.SIZE, Image.LANCZOS) if self.scale < 1. else render
            render = np.array(render, dtype=np.float32) / 255.
            render = render[:, :, :3] * render[:, :, -1:]
            pose = np.array(frame["transform_matrix"], dtype=np.float32)
            renders[f], poses[f] = torch.from_numpy(render), torch.from_numpy(pose)

        directions = generate_directions(*self.SIZE, self.focal)
        rays_origin = torch.zeros((poses.size(0), *self.SIZE, 3), dtype=torch.float32)
        rays_direction = torch.zeros((poses.size(0), *self.SIZE, 3), dtype=torch.float32)
        for p, pose in enumerate(tqdm(poses, desc=f"[Blender|{self.split.name.capitalize()}] Projecting")):
            rays_origin[p], rays_direction[p] = projection(directions, pose)

        pixels_color = renders.view(-1, 3)
        rays_origin = rays_origin.view(-1, 3)
        rays_direction = rays_direction.view(-1, 3)
        self.data = BlenderData(pixels_color, Ray(rays_origin, rays_direction))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Union[int, slice]) -> BlenderData:
        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            return self.data[start:stop:step]
        return self.data[idx]

    def collate_fn(self, batch: List[BlenderData]) -> BlenderData:
        pixels_color = torch.stack([data.pixel_color for data in batch], dim=0)
        rays_origin = torch.stack([data.ray.origin for data in batch], dim=0)
        rays_direction = torch.stack([data.ray.direction for data in batch], dim=0)
        return BlenderData(pixels_color, Ray(rays_origin, rays_direction))

    def loader(self, batch_size: int, jobs: int) -> DataLoader:
        return DataLoader(
            self,
            batch_size,
            shuffle=self.split==Split.TRAIN,
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
            
            rays = self[s:e].ray
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