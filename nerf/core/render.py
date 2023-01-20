import torch

from dataclasses import dataclass
from nerf.core.nn import NeRFModel, NeRFOutput, PositionalEncoder
from nerf.data.utils import BoundingBox, CameraPlanes, Ray
from torch import Tensor


def sample_pdf(bins: Tensor, weights: Tensor, samples: int, perturb: bool) -> Tensor:
    pdf = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat((torch.zeros_like(cdf[:, :1]), cdf), dim=-1)

    if perturb:
        u = torch.rand((weights.size(0), samples), device=weights.device)
        u = u.contiguous()
    else:
        u = torch.linspace(0., 1., samples, device=bins.device)
        u = u.expand(weights.size(0), samples).contiguous()

    idxs = torch.searchsorted(cdf, u, right=True)
    idxs_below = torch.clamp_min(idxs - 1, 0)
    idxs_above = torch.clamp_max(idxs, weights.size(1))
    idxs = torch.stack((idxs_below, idxs_above), dim=-1)
    idxs = idxs.view(weights.size(0), 2 * samples)

    cdf = torch.gather(cdf, dim=1, index=idxs).view(weights.size(0), samples, 2)
    bins = torch.gather(bins, dim=1, index=idxs).view(weights.size(0), samples, 2)

    den = cdf[:, :, 1] - cdf[:, :, 0]
    den[den < 1e-5] = 1.

    return bins[:, :, 0] + (u - cdf[:, :, 0]) / den * (bins[:, :, 1] - bins[:, :, 0])


def exclusive_cumprod(x: Tensor) -> Tensor:
    ecp = torch.roll(torch.cumprod(x, dim=-1), 1, dims=-1)
    ecp[:, 0] = 1.
    return ecp


def accumulate_weights(sigma: Tensor, delta: Tensor) -> Tensor:
    alpha = 1. - torch.exp(-sigma * delta)
    trans = exclusive_cumprod(1. - alpha + 1e-10)
    trans[trans <= 1e-4] = 0.
    return alpha * trans


def segment_lengths(t: Tensor) -> Tensor:
    delta = t[:, 1:] - t[:, :-1]
    delti = 1e10 * torch.ones((t.size(0), 1), device=t.device)
    return torch.cat((delta, delti), dim=-1)
    

@dataclass
class SampleData:
    position: Tensor
    direction: Tensor

    def to(self, device: torch.device) -> "SampleData":
        self.position = self.position.to(device)
        self.direction = self.direction.to(device)
        return self


def sample_rays(rays: Ray, t: Tensor, samples: int) -> SampleData:
    rx = rays.origin[:, None, :] + rays.direction[:, None, :] * t[:, :, None]
    rd = torch.repeat_interleave(rays.direction, repeats=samples, dim=0)
    return SampleData(rx, rd.view(t.size(0), samples, 3))


def intersect_rays(data: SampleData, bbox: BoundingBox) -> Tensor:
    x = (data.position[:, :, 0] >= bbox.left  ) & (data.position[:, :, 0] <= bbox.right)
    y = (data.position[:, :, 1] >= bbox.bottom) & (data.position[:, :, 1] <= bbox.top  )
    z = (data.position[:, :, 2] >= bbox.back  ) & (data.position[:, :, 2] <= bbox.front)
    return (x & y & z).float()


@dataclass
class Samples:
    coarse: int
    fine: Tensor

    @property
    def total(self) -> int:
        return self.coarse + self.fine


@dataclass
class RenderData:
    t: Tensor
    pixel_weight: Tensor
    pixel_color: Tensor


def sort_out(coarse: NeRFOutput, fine: NeRFOutput, idxs: Tensor) -> NeRFOutput:
    sigma = torch.cat((coarse.sigma, fine.sigma), dim=1)
    rgb   = torch.cat((coarse.rgb,   fine.rgb  ), dim=1)
    return NeRFOutput(
        torch.gather(sigma, dim=1, index=idxs),
        torch.gather(rgb,   dim=1, index=idxs[:, :, None].repeat(1, 1, 3)),
    )


def render_volume(
    nerf: NeRFModel,
    phi_p: PositionalEncoder,
    phi_d: PositionalEncoder,
    rays: Ray,
    planes: CameraPlanes,
    samples: Samples,
    perturb: bool,
    bbox: BoundingBox = None,
) -> RenderData:
    device = next(nerf.parameters()).device
    
    t = torch.linspace(planes.near, planes.far, samples.coarse, device=device)
    t = t.expand(rays.origin.size(0), samples.coarse).contiguous()
    
    bins = .5 * (t[:, :-1] + t[:, 1:])
    if perturb:
        tu = torch.cat((bins, t[:, -1:]), dim=-1)
        tl = torch.cat((t[:, :1], bins), dim=-1)
        t = tl + (tu - tl) * torch.rand_like(t)

    query = sample_rays(rays, t, samples.coarse)
    out: NeRFOutput = nerf(phi_p(query.position), phi_d(query.direction))

    if bbox is not None:
        mask = intersect_rays(query, bbox)
        out = NeRFOutput(out.sigma * mask, out.rgb * mask[:, :, None])
    
    weights = accumulate_weights(out.sigma, segment_lengths(t))
    weights = torch.nan_to_num(weights)

    if samples.fine > 0:
        weights = weights.detach()

        t_pdf = sample_pdf(bins, weights[:, 1:-1], samples.fine, perturb)
        t, idxs = torch.sort(torch.cat((t, t_pdf), dim=-1), dim=-1)
        
        query = sample_rays(rays, t_pdf, samples.fine)
        out_fine: NeRFOutput = nerf(phi_p(query.position), phi_d(query.direction))

        if bbox is not None:
            mask = intersect_rays(query, bbox)
            out_fine = NeRFOutput(out_fine.sigma * mask, out_fine.rgb * mask[:, :, None])
        
        out = sort_out(out, out_fine, idxs)

        weights = accumulate_weights(out.sigma, segment_lengths(t))
        weights = torch.nan_to_num(weights)

    colors = torch.sum(weights[:, :, None] * out.rgb, dim=-2)

    return RenderData(t, weights, colors)


def render_depth(weights: Tensor, t: Tensor) -> Tensor:
    depth = torch.sum(weights * t, dim=-1)
    epsil = 1e-10 * torch.ones_like(depth)
    return 1.0 / torch.max(epsil, depth / weights.sum(dim=-1))