from tqdm import tqdm
from PIL import Image

import numpy as np
import os


os.makedirs("figures", exist_ok=True)

models  = ["NeRF", "TinyNeRF", "MicroNeRF", "NanoNeRF"]
models += [f"Distill{model}" for model in models]

blender = "dataset/blender"
scenes = sorted([s for s in os.listdir(blender) if os.path.isdir(os.path.join(blender, s))])

for scene in tqdm(scenes):
    irm = os.path.join("res", "DistillNeRF", scene, "DistillNeRF.irm.png")
    irm = Image.open(irm).convert("RGB")

    renders = [os.path.join("res", model, scene, f"{model}.png") for model in models]
    renders = [np.array(Image.open(render)) for render in renders]
    renders = [render[render.shape[0] // 3:, :, :] for render in renders]
    renders = np.hstack(renders)
    
    a = renders.shape[1] / irm.size[0]
    w = renders.shape[1]
    h = int(a * irm.size[1])
    
    irm = np.array(irm.resize((w, h)))
    
    img = np.vstack([irm, renders])
    Image.fromarray(img).save(f"figures/{scene}.figure.png")