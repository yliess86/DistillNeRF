from nerf.core.nn import NeRFFactory, PositionalEncoder
from nerf.core.render import Samples
from nerf.data.irm import IRM
from nerf.data.turntable import TurnTable
from nerf.data.utils import CameraPlanes, BoundingBox
from nerf.benchmark import NeRFBenchmarker
from nerf.distill import NeRFDistiller
from nerf.infer import NeRFInferer
from nerf.irm import NeRFIRM
from nerf.train import NeRFTrainer

import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch


# ==== HP
blender = "dataset/blender"
scene = "chair"
threshold = 1e-2

bbox_size = -20., 20.
bbox = BoundingBox(*bbox_size, *bbox_size, *bbox_size)
delta = (6. - 2.) / 128

batch_size = 4_096
big_batch_size = batch_size * 4

device = "cuda"

reptile_steps  = 32
distill_epochs = 200
train_epochs   = 150

assert len(sys.argv) <= 2
method = str(sys.argv[1]) if len(sys.argv) == 2 else ""
method = method.capitalize()

# ==== PATH
os.makedirs(f"res/Best{method}NanoNeRF/{scene}", exist_ok=True)
m_path = f"res/Best{method}NanoNeRF/{scene}/Best{method}NanoNeRF.pt"
r_path = f"res/Best{method}NanoNeRF/{scene}/Best{method}NanoNeRF.png"
g_path = f"res/Best{method}NanoNeRF/{scene}/Best{method}NanoNeRF.gif"
i_path = f"res/Best{method}NanoNeRF/{scene}/Best{method}NanoNeRF.irm.png"
b_path = f"res/Best{method}NanoNeRF/{scene}/Best{method}NanoNeRF_benchmark.csv"

# ==== MODELS
phi_p = PositionalEncoder(3, 10).to(device)
phi_d = PositionalEncoder(3,  4).to(device)

tnerf = NeRFFactory.NeRF(phi_p.o_dim, phi_d.o_dim).to(device)
snerf = NeRFFactory.NanoNeRF(phi_p.o_dim, phi_d.o_dim).to(device)

ckpt = f"res/ReptileNeRF/{scene}/ReptileNeRF.pt"
if not os.path.isfile(ckpt): raise FileNotFoundError(f"Teacher Model {ckpt} Not Found")
tnerf.load_state_dict(torch.load(ckpt, map_location="cpu")["nerf"])

# ==== DISTILL
if method == "Distill":
    bbox = IRM.refine(bbox, 256, tnerf, phi_p, phi_d, 16, big_batch_size, threshold=threshold)
    irm = IRM(bbox, 16, samples=500_000)
    irm.scan(tnerf, phi_p, phi_d, 256, big_batch_size)

    distiller = NeRFDistiller(tnerf, snerf, phi_p, phi_d).to(device)
    distiller.set_data(irm, batch_size=big_batch_size)
    _, h = distiller.distill(distill_epochs, lr=1e-3, delta=delta, model_path=m_path)

    plt.figure()
    plt.plot(np.arange(len(h)), h)
    plt.yscale("log")
    plt.savefig(f"res/Best{method}NanoNeRF/{scene}/Best{method}NanoNeRF.distill.png")

# ==== TRAIN
trainer = NeRFTrainer(snerf, phi_p, phi_d, Samples(64, 64)).to(device)
trainer.set_data(blender, scene=scene, step=1, scale=.5, batch_size=batch_size)

# ==== REPTILE
if method == "Reptile":
    trainer.reptile(reptile_steps, lr=5e-4, perturb=True, model_path=m_path, render_path=r_path)

# ==== TRAIN
_, h = trainer.fit(train_epochs, lr=5e-4, perturb=True, log=5, model_path=m_path, render_path=r_path)

plt.figure()
plt.plot(np.arange(len(h)), h)
plt.savefig(f"res/Best{method}NanoNeRF/{scene}/Best{method}NanoNeRF.train.png")

# ==== INFER
inferer = NeRFInferer(snerf, phi_p, phi_d).to(device)
inferer.set_data(TurnTable(800, 800, .5 * 800 / np.tan(.5 * 0.69), 40, 4), big_batch_size)
inferer.render(15, CameraPlanes(2., 6.), Samples(64, 64), g_path, bbox=bbox)

# ==== IRM
irm = NeRFIRM(snerf, phi_p, phi_d).to(device)
irm.set_data(bbox, 16)
irm.scan(8, 64, big_batch_size, i_path)

# ==== BENCHMARK
benchmarker = NeRFBenchmarker(snerf, phi_p, phi_d, Samples(64, 64)).to(device)
benchmarker.set_data(blender, scene, step=1, scale=.5, batch_size=big_batch_size)
ben = benchmarker.benchmark(bbox=bbox)

with open(b_path, "w") as bfp:
    bfp.writelines([
        f"fps;mse;psnr;ssim;size\n",
        f"{ben.fps};{ben.mse};{ben.psnr};{ben.ssim};{ben.size}",
    ])