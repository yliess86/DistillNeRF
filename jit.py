import os
import torch
import torch.jit as jit

from nerf.core.nn import NeRFFactory, PositionalEncoder
from tqdm import tqdm


phi_p = PositionalEncoder(3, 10)
phi_d = PositionalEncoder(3,  4)

root = "res"
for m in tqdm([m for m in os.listdir(root) if os.path.isdir(os.path.join(root, m))], desc="Model"):
    m_root = os.path.join(root, m)

    for s in tqdm([s for s in os.listdir(m_root) if os.path.isdir(os.path.join(m_root, s))], desc="Scene"):
        s_root = os.path.join(m_root, s)

        ckpt = os.path.join(s_root, f"{m}.pt")
        ckpt = torch.load(ckpt, map_location="cpu")

        nerf = getattr(NeRFFactory, m.replace("Distill", ""))
        nerf = nerf(phi_p.o_dim, phi_d.o_dim)
        nerf.load_state_dict(ckpt["nerf"])

        j_root = os.path.join(s_root, "jit")
        os.makedirs(j_root, exist_ok=True)

        jit.script(phi_p).save(os.path.join(j_root, "phi_p.pt"))
        jit.script(phi_d).save(os.path.join(j_root, "phi_d.pt"))
        jit.script(nerf ).save(os.path.join(j_root, "nerf.pt" ))