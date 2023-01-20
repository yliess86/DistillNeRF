import numpy as np
import torch
import torch.jit as jit

from torch import Tensor
from torch.nn import Linear, Module, ModuleList, Sequential, SiLU
from typing import Callable, Iterable, List, NamedTuple


@jit.script
def positional_encoding(v: Tensor, freq_bands: Tensor) -> Tensor:
    pe = [v]
    for freq_band in freq_bands:
        fv = freq_band * v
        pe += [torch.sin(fv), torch.cos(fv)]
    return torch.cat(pe, dim=-1)


class PositionalEncoder(Module):
    def __init__(self, i_dim: int, n_freqs: int) -> None:
        super().__init__()
        self.i_dim = i_dim
        self.n_freqs = n_freqs
        self.o_dim = self.i_dim + (2 * self.n_freqs) * self.i_dim

        freq_bands = 2 ** torch.linspace(1, self.n_freqs - 1, self.n_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, v: Tensor) -> Tensor:
        return positional_encoding(v, self.freq_bands)


@jit.script
def shifted_softplus(x: Tensor) -> Tensor:
    shifted_x = x - 1.
    linear_x = shifted_x * (shifted_x >= 0).float()
    return torch.log1p(torch.exp(-torch.abs(shifted_x))) + linear_x


@jit.script
def widened_sigmoid(x: Tensor) -> Tensor:
    return .5 * (1. + (1. + 2. * 1e-3) * torch.tanh(.5 * x))


class ShiftedSoftplus(Module):
    def __init__(self) -> None: super().__init__()
    def forward(self, x: Tensor) -> Tensor: return shifted_softplus(x)


class WidenedSigmoid(Module):
    def __init__(self) -> None: super().__init__()
    def forward(self, x: Tensor) -> Tensor: return widened_sigmoid(x)


class NeRFOutput(NamedTuple):
    sigma: Tensor
    rgb: Tensor


class NeRFConfig(NamedTuple):
    name: str
    width: str
    depth: str
    residual: bool


class NeRFFactory:
    models: List[Callable] = []

    @staticmethod
    def register(*configs: Iterable[NeRFConfig]) -> Callable:
        def _register(cls: "NeRFModel") -> "NeRFModel":
            for config in configs:
                def __init__(self, phi_p_dim: int, phi_d_dim: int) -> None:
                    super(self.__class__, self).__init__(phi_p_dim, phi_d_dim, self.width, self.depth, self.residual)

                new_cls = type(config.name, (cls, ), {
                    "width": config.width,
                    "depth": config.depth,
                    "residual": config.residual,
                    "__init__": __init__,
                })
                
                setattr(NeRFFactory, config.name, new_cls)
                getattr(NeRFFactory, "models").append(new_cls)
            return cls
        return _register


@NeRFFactory.register(
    NeRFConfig( "NanoNeRF",  32, 4, False),
    NeRFConfig("MicroNeRF",  64, 4, False),
    NeRFConfig( "TinyNeRF", 128, 4,  True),
    NeRFConfig(     "NeRF", 256, 8,  True),
)
class NeRFModel(Module):
    def __init__(self, phi_p_dim: int, phi_d_dim: int, width: int, depth: int, residual: bool) -> None:
        super().__init__()
        self.phi_p_dim = phi_p_dim
        self.phi_d_dim = phi_d_dim
        self.width = width
        self.depth = depth
        self.residual = residual

        self.encoder = ModuleList()
        for d in range(self.depth):
            if d == 0:                                         i_dim = self.phi_p_dim
            elif self.residual and d == (self.depth // 2 - 1): i_dim = self.phi_p_dim + self.width
            else:                                              i_dim = self.width
            self.encoder.append(Sequential(Linear(i_dim, self.width), SiLU()))
        
        self.sigma = Sequential(Linear(self.width, 1), ShiftedSoftplus())
        self.features = Sequential(Linear(self.width, self.width))
        self.rgb = Sequential(
            Linear(self.phi_d_dim + self.width, self.width // 2), SiLU(),
            Linear(self.width // 2, 3), WidenedSigmoid(),
        )

    def forward(self, phi_p: Tensor, phi_d: Tensor) -> NeRFOutput:
        x = phi_p
        for d, layer in enumerate(self.encoder):
            if self.residual and d == (self.depth // 2 - 1):
                x = torch.cat((phi_p, x), dim=-1)
            x = layer(x)

        return NeRFOutput(
            self.sigma(x).view(phi_p.size()[:-1]),
            self.rgb(torch.cat((phi_d, self.features(x)), dim=-1)),
        )

    def bytes(self) -> int:
        return np.sum([np.product(p.size()) * 4 for p in self.parameters()])