import torch
import numpy as np

from nerf.core.nn import NeRFModel, PositionalEncoder
from nerf.core.render import SampleData
from nerf.data.utils import BoundingBox, voxelgrid
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Tuple


def boundaries(voxels: np.ndarray, axis: int) -> Tuple[int, int]:
    nonzero = np.nonzero(voxels)[axis]
    return nonzero.min(), nonzero.max() + 1


class IRM(Dataset):
    @classmethod
    @torch.inference_mode()
    def refine(
        cls,
        bbox: BoundingBox,
        divisions: int,
        nerf: NeRFModel,
        phi_p: PositionalEncoder,
        phi_d: PositionalEncoder,
        samples: int,
        batch_size: int,
        threshold: float,
    ) -> BoundingBox:
        irm = cls(bbox, divisions, samples=1)
        irm.scan(nerf, phi_p, phi_d, samples, batch_size)
        print(f"[IRM] Raw {bbox}")
        
        irm.voxels[irm.voxels <  threshold] = 0.
        irm.voxels[irm.voxels >= threshold] = 1.

        voxels = irm.voxels.view(divisions, divisions, divisions)
        voxels = voxels.numpy()

        left, right = boundaries(voxels, axis=0)
        left  = bbox.left + left  * irm.voxel_size[0]
        right = bbox.left + right * irm.voxel_size[0]

        bottom, top = boundaries(voxels, axis=1)
        bottom = bbox.bottom + bottom * irm.voxel_size[1]
        top    = bbox.bottom + top    * irm.voxel_size[1]

        back, front = boundaries(voxels, axis=2)
        back  = bbox.back + back  * irm.voxel_size[2]
        front = bbox.back + front * irm.voxel_size[2]

        bbox = BoundingBox(left=bottom, right=top, bottom=left, top=right, back=back, front=front)
        print(f"[IRM] Refined {bbox}")

        return bbox

    def __init__(self, bbox: BoundingBox, divisions: int, samples: int) -> None:
        self.bbox = bbox
        self.divisions = divisions
        self.samples = samples

        self.grid = voxelgrid(self.divisions, self.divisions, self.divisions).view(-1, 3)
        self.voxels = torch.zeros((self.divisions ** 3), dtype=torch.float32)
        self.voxel_size = torch.tensor([
            (bbox.right - bbox.left  ) / self.divisions,
            (bbox.top   - bbox.bottom) / self.divisions,
            (bbox.front - bbox.back  ) / self.divisions,
        ], dtype=torch.float32)

    def __len__(self) -> int:
        return self.samples

    @torch.inference_mode()
    def scan(
        self,
        nerf: NeRFModel,
        phi_p: PositionalEncoder,
        phi_d: PositionalEncoder,
        samples: int,
        batch_size: int,
    ) -> None:
        device = next(nerf.parameters()).device
        
        ray_positions = self.grid.clone() / self.divisions
        ray_positions[:, 0] = self.bbox.left   + (self.bbox.right - self.bbox.left  ) * ray_positions[:, 0]
        ray_positions[:, 1] = self.bbox.bottom + (self.bbox.top   - self.bbox.bottom) * ray_positions[:, 1]
        ray_positions[:, 2] = self.bbox.back   + (self.bbox.front - self.bbox.back  ) * ray_positions[:, 2]

        for s in tqdm(range(0, self.voxels.size(0), batch_size), desc="[IRM] Scanning"):
            e = min(s + batch_size, self.voxels.size(0))

            for _ in range(samples):
                ray_offset = torch.rand((e - s, 3), dtype=torch.float32)
                ray_position = ray_positions[s:e] + ray_offset * self.voxel_size[None, :]
                ray_position = ray_position.to(device)
                
                ray_direction = torch.rand_like(ray_position)
                ray_direction = ray_direction / (torch.norm(ray_direction, dim=-1, keepdim=True) + 1e-5)

                out = nerf(phi_p(ray_position), phi_d(ray_direction))
                self.voxels[s:e] += out.sigma.cpu() / samples

        self.voxels = self.voxels / (torch.max(self.voxels) + 1e-10)

        p = self.voxels.view(-1) / self.voxels.sum()
        idxs = p.multinomial(num_samples=self.samples, replacement=True) if p.size(0) > 1 else [0] * self.samples
        self.ray_positions = self.grid[idxs] / self.divisions
        self.ray_positions[:, 0] = self.bbox.left   + (self.bbox.right - self.bbox.left  ) * self.ray_positions[:, 0]
        self.ray_positions[:, 1] = self.bbox.bottom + (self.bbox.top   - self.bbox.bottom) * self.ray_positions[:, 1]
        self.ray_positions[:, 2] = self.bbox.back   + (self.bbox.front - self.bbox.back  ) * self.ray_positions[:, 2]

    def __getitem__(self, idx: int) -> SampleData:
        ray_position = self.ray_positions[idx]

        ray_offset = torch.rand((3, ), dtype=torch.float32)
        ray_position = ray_position + ray_offset * self.voxel_size

        ray_direction = torch.rand_like(ray_position)
        ray_direction = ray_direction / (torch.norm(ray_direction, dim=-1, keepdim=True) + 1e-5)

        return SampleData(ray_position, ray_direction)

    def collate_fn(self, batch: List[SampleData]) -> SampleData:
        positions = torch.stack([sample.position for sample in batch], dim=0)
        directions = torch.stack([sample.direction for sample in batch], dim=0)
        return SampleData(positions, directions)

    def loader(self, batch_size: int, jobs: int) -> DataLoader:
        return DataLoader(
            self,
            batch_size,
            num_workers=jobs,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )