import numpy as np
import torch

from multiprocessing import cpu_count
from nerf.core.nn import NeRFModel, PositionalEncoder
from nerf.data.irm import IRM
from torch.cuda.amp import autocast, GradScaler
from torch.nn.functional import mse_loss
from torch.optim import Adam
from tqdm import tqdm
from typing import NamedTuple


class DistillOutcome(NamedTuple):
    mse: float
    psnr: float


class NeRFDistiller:
    def __init__(
        self,
        teacher: NeRFModel,
        student: NeRFModel,
        phi_p: PositionalEncoder,
        phi_d: PositionalEncoder,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.phi_p = phi_p
        self.phi_d = phi_d

        self.device = torch.device("cpu")

    def to(self, device: torch.device) -> "NeRFDistiller":
        self.device = device
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        self.phi_p = self.phi_p.to(self.device)
        self.phi_d = self.phi_d.to(self.device)
        return self

    def set_data(self, irm: IRM, batch_size: int) -> None:
        self.irm = irm
        self.batch_size = batch_size

        self.irm_loader = self.irm.loader(self.batch_size, cpu_count())

    def distill(
        self,
        epochs: int,
        lr: float,
        delta: float,
        model_path: str,
    ) -> DistillOutcome:
        self.optim = Adam(self.student.parameters(), lr=lr)
        self.scaler = GradScaler()

        history = []

        for epoch in tqdm(range(epochs), desc="[NeRF] Distill"):
            total_mse, total_psnr = 0., 0.
            
            batches = tqdm(self.irm_loader, desc=f"[NeRF] Distill")
            for batch in batches:
                batch = batch.to(self.device)

                with autocast():
                    phi_p = self.phi_p(batch.position)
                    phi_d = self.phi_d(batch.direction)
                    
                    s_sigma, s_rgb = self.student(phi_p, phi_d)
                    s_alpha = 1. - torch.exp(-s_sigma * delta)

                    with torch.no_grad():
                        t_sigma, t_rgb = self.teacher(phi_p, phi_d)
                        t_alpha = 1. - torch.exp(-t_sigma * delta)
                        t_sigma = t_sigma
                        t_alpha = t_alpha

                    l2 = sum(torch.norm(p) for p in self.student.rgb.parameters())
                    mse = .5 * (mse_loss(s_rgb, t_rgb) + mse_loss(s_alpha, t_alpha))
                    loss = mse + 1e-6 * l2

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)

                total_mse += mse.item() / len(self.irm_loader)
                total_psnr = -10 * np.log10(total_mse)
                
                batches.set_postfix(mse=f"{total_mse:.2e}", psnr=f"{total_psnr:.2e}")

            history.append(total_mse)

            with torch.inference_mode():
                self._save(model_path, epoch)

        return DistillOutcome(total_mse, total_psnr), history

    def _save(self, path: str, epoch: int) -> None:
        torch.save({
            "epoch": epoch,
            "nerf": self.student.state_dict(),
            "optim": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
        }, path)