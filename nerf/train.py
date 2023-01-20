import numpy as np
import torch

from copy import deepcopy
from nerf.core.nn import NeRFModel, PositionalEncoder
from nerf.core.render import render_depth, render_volume, Samples
from nerf.core.scheduler import LogDecayScheduler
from nerf.data.blender import Blender
from nerf.data.utils import BoundingBox, Split
from os import cpu_count
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torch.nn.functional import mse_loss
from torch.optim import Adam, SGD
from tqdm import tqdm
from typing import NamedTuple


class TrainOutcome(NamedTuple):
    mse: float
    psnr: float


class NeRFTrainer:
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

        self.samples = samples
        self.device = torch.device("cpu")

    def to(self, device: torch.device) -> "NeRFTrainer":
        self.device = device
        self.nerf = self.nerf.to(self.device)
        self.phi_p = self.phi_p.to(self.device)
        self.phi_d = self.phi_d.to(self.device)
        return self

    def set_data(self, blender: str, scene: str, step: int, scale: float, batch_size: int) -> None:
        self.blender = blender
        self.scene = scene
        self.batch_size = batch_size
        
        self.train_set = Blender(self.blender, self.scene, Split.TRAIN, step, scale)
        self.train_loader = self.train_set.loader(self.batch_size, cpu_count())
        self.planes = self.train_set.planes
        
        self.val_set = Blender(self.blender, self.scene, Split.VAL, step, scale)
        self.val_loader = self.val_set.loader(self.batch_size, cpu_count())
        self.val_data = self.val_set[:self.val_set.PIXELS]
        
        self.test_set = Blender(self.blender, self.scene, Split.TEST, step, scale)
        self.test_loader  = self.test_set.loader(self.batch_size, cpu_count())

        self.best_psnr = 0
        self.last_best = 0

    def reptile(
        self,
        steps: int,
        lr: float,
        perturb: bool,
        model_path: str,
        render_path: str,
    ) -> TrainOutcome:
        self.optim = Adam(self.nerf.parameters(), lr=lr)
        self.scaler = GradScaler()

        meta_nerf = deepcopy(self.nerf)
        meta_optim = SGD(meta_nerf.parameters(), lr=lr * 1e3, momentum=.9)
        meta_scaler = GradScaler()

        total_mse, total_psnr = 0., 0.
        modules = self.nerf, self.phi_p, self.phi_d
        meta_modules = meta_nerf, self.phi_p, self.phi_d
        
        self.loader = self.train_loader
        batches = tqdm(self.loader, desc=f"[NeRF] Reptile")
        for batch in batches:
            batch = batch.to(self.device)

            for _ in range(steps):
                with autocast():
                    out = render_volume(*meta_modules, batch.ray, self.planes, self.samples, perturb)
                    meta_mse = mse_loss(out.pixel_color, batch.pixel_color)
                    
                meta_scaler.scale(meta_mse).backward()
                meta_scaler.step(meta_optim)
                meta_scaler.update()
                meta_optim.zero_grad(set_to_none=True)

            with torch.no_grad():
                for p, meta_p in zip(self.nerf.parameters(), meta_nerf.parameters()):
                    p.grad = self.scaler.scale(p - meta_p)
        
                with autocast():
                    out = render_volume(*modules, batch.ray, self.planes, self.samples, perturb)
                    mse = mse_loss(out.pixel_color, batch.pixel_color)

            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad(set_to_none=True)
            meta_nerf.load_state_dict(self.nerf.state_dict())

            total_mse += mse.item() / len(self.loader)
            total_psnr = -10 * np.log10(total_mse)
            
            batches.set_postfix(mse=f"{total_mse:.2e}", psnr=f"{total_psnr:.2e}")

        self.best_psnr = total_psnr
        with torch.inference_mode():
            self._render(render_path)
            self._save(model_path, 0)

        return TrainOutcome(total_mse, total_psnr)

    def fit(
        self,
        epochs: int,
        lr: float,
        perturb: bool,
        log: int,
        model_path: str,
        render_path: str,
    ) -> TrainOutcome:
        self.optim = Adam(self.nerf.parameters(), lr=lr)
        self.scaler = GradScaler()
        self.scheduler = LogDecayScheduler(self.optim, lr * 1e-2, lr, epochs * len(self.train_loader), 6)

        history = []

        for epoch in tqdm(range(1, epochs + 1), desc="[NeRF] Epoch"):
            self._step(perturb, Split.TRAIN)

            if epoch % log == 0 or epoch == epochs:
                with torch.inference_mode():
                    mse, psnr = self._step(False, Split.VAL)
                    history.append(mse)

                    if psnr > self.best_psnr:
                        self.best_psnr = psnr
                        self.last_best = 0
                        self._render(render_path)
                        self._save(model_path, epoch)

                    else:
                        self.last_best += 1
                        if self.last_best > 2: break
                        print(f"[NeRF] Training Stopped at epoch: {epoch}")

        with torch.inference_mode():
            outcome = self._step(False, Split.TEST)

        return outcome, history

    def _step(self, perturb: bool, split: Split, bbox: BoundingBox = None) -> TrainOutcome:
        train = split == Split.TRAIN
        self.nerf = self.nerf.train(mode=train)
        self.loader = (
            self.train_loader if split == Split.TRAIN else
            self.val_loader if split == Split.VAL else
            self.test_loader
        )
            
        total_mse, total_psnr = 0., 0.
        modules = self.nerf, self.phi_p, self.phi_d
        
        batches = tqdm(self.loader, desc=f"[NeRF] {split.name.capitalize()}")
        for batch in batches:
            batch = batch.to(self.device)

            with autocast():
                out = render_volume(*modules, batch.ray, self.planes, self.samples, perturb, bbox=bbox)
                mse = mse_loss(out.pixel_color, batch.pixel_color)

            if train:
                self.scaler.scale(mse).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.scheduler.step()
                self.optim.zero_grad(set_to_none=True)

            total_mse += mse.item() / len(self.loader)
            total_psnr = -10 * np.log10(total_mse)
            
            batches.set_postfix(
                mse=f"{total_mse:.2e}",
                psnr=f"{total_psnr:.2e}",
                lr=f"{self.scheduler.lr if train else 0.:.2e}",
            )

        return TrainOutcome(total_mse, total_psnr)

    @torch.inference_mode()
    def _render(self, path :str, bbox: BoundingBox = None) -> None:
        PIXELS, SIZE = self.val_set.PIXELS, self.val_set.SIZE
        data = self.val_set[:PIXELS]

        modules = self.nerf, self.phi_p, self.phi_d

        rgb_map = torch.zeros((PIXELS, 3))
        dep_map = torch.zeros((PIXELS, 3))
        for s in tqdm(range(0, PIXELS, self.batch_size), desc="[NeRF] Rendering"):
            e = min(s + self.batch_size, PIXELS)
            
            rays = data.ray[s:e].to(self.device)
            with autocast():
                out = render_volume(*modules, rays, self.planes, self.samples, False, bbox=bbox)
            rgb_map[s:e] = out.pixel_color.cpu()
            dep_map[s:e] = render_depth(out.pixel_weight, out.t)[:, None].repeat(1, 3).cpu()
        
        gtr_map = data.pixel_color.view(*SIZE, 3)
        rgb_map = rgb_map.view(*SIZE, 3).clip(0., 1.)
        dep_map = dep_map.view(*SIZE, 3).clip(0., 1.)

        img = torch.vstack((gtr_map, rgb_map, dep_map))
        img = Image.fromarray((img.numpy() * 255.).astype(np.uint8))
        img.save(path)

    def _save(self, path: str, epoch: int) -> None:
        torch.save({
            "epoch": epoch,
            "nerf": self.nerf.state_dict(),
            "optim": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
        }, path)