import torch


from torch import Tensor
from torch.optim import Optimizer


class Scheduler:
    def __init__(self, optim: Optimizer, lr: float) -> None:
        self.optim = optim
        self.lr = lr

        self._update()

    def _update(self):
        for group in self.optim.param_groups:
            group["lr"] = self.lr

    def step(self) -> None:
        raise NotImplementedError("Step not Implemented Yet!")


class IdentiyScheduler(Scheduler):
    def __init__(self, optim: Optimizer, lr: float) -> None:
        super().__init__(optim, lr)

    def step(self) -> None:
        pass


class LogDecayScheduler(Scheduler):
    def __init__(
        self,
        optim: Optimizer,
        lr_min: float,
        lr_max: float,
        iterations: int,
        power: int,
    ) -> None:
        super().__init__(optim, lr_max)
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.power = power
        
        self.iterations = iterations
        self.iteration = 0.

    def step(self) -> None:
        self.iteration += 1

        t = self.iteration / self.iterations
        scale = 1 - 10 ** (-self.power * (1 - t))

        self.lr = self.lr_min + (self.lr_max - self.lr_min) * scale
        self._update()