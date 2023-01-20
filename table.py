import os


models = ["NeRF", "TinyNeRF", "MicroNeRF", "NanoNeRF"]
l = max(map(len, models)) + max(len("Reptile"), len("Distill"))

blender = "dataset/blender"
scenes = sorted([s for s in os.listdir(blender) if os.path.isdir(os.path.join(blender, s))])

for scene in scenes:
    print("====", scene)

    n_table = ""
    r_table = ""
    d_table = ""

    for model in models:
        path = f"res/{model}/{scene}/{model}_benchmark.csv"
        with open(path, "r") as fp: _, content = fp.read().split("\n")
        fps, mse, psnr, ssim, size = map(float, content.split(";"))
        n_table += f"{model:{l}s} & {fps:.2f} & {mse:.2e} & {psnr:.2f} & {ssim:.2f} & {size:.2f} \\\\ \n"

        path = f"res/Reptile{model}/{scene}/Reptile{model}_benchmark.csv"
        with open(path, "r") as fp: _, content = fp.read().split("\n")
        fps, mse, psnr, ssim, size = map(float, content.split(";"))
        r_table += f"{'Reptile' + model:{l}s} & {fps:.2f} & {mse:.2e} & {psnr:.2f} & {ssim:.2f} & {size:.2f} \\\\ \n"

        path = f"res/Distill{model}/{scene}/Distill{model}_benchmark.csv"
        with open(path, "r") as fp: _, content = fp.read().split("\n")
        fps, mse, psnr, ssim, size = map(float, content.split(";"))
        d_table += f"{'Distill' + model:{l}s} & {fps:.2f} & {mse:.2e} & {psnr:.2f} & {ssim:.2f} & {size:.2f} \\\\ \n"
        
    print("\\toprule")
    print("\\textbf{Architecture} & \\textbf{Rendering (FPS)} $\\uparrow$ & \\textbf{MSE} $\\downarrow$ & \\textbf{PSNR} $\\uparrow$ & \\textbf{SSIM} $\\uparrow$ & \\textbf{Size (MB)} $\\downarrow$ \\\\")
    print(f"\\midrule\n{n_table}\\midrule\n{r_table}\\midrule\n{d_table}\\bottomrule\n")

print("====\n")

n_table = ""
r_table = ""
d_table = ""
for model in models:
    N_FPS, N_MSE, N_PSNR, N_SSIM, N_SIZE = 0., 0., 0., 0., 0.
    R_FPS, R_MSE, R_PSNR, R_SSIM, R_SIZE = 0., 0., 0., 0., 0.
    D_FPS, D_MSE, D_PSNR, D_SSIM, D_SIZE = 0., 0., 0., 0., 0.
    
    for scene in scenes:
        path = f"res/{model}/{scene}/{model}_benchmark.csv"
        with open(path, "r") as fp: _, content = fp.read().split("\n")
        fps, mse, psnr, ssim, size = map(float, content.split(";"))
       
        N_FPS  += fps  / len(scenes)
        N_MSE  += mse  / len(scenes)
        N_PSNR += psnr / len(scenes)
        N_SSIM += ssim / len(scenes)
        N_SIZE += size / len(scenes)

        path = f"res/Reptile{model}/{scene}/Reptile{model}_benchmark.csv"
        with open(path, "r") as fp: _, content = fp.read().split("\n")
        fps, mse, psnr, ssim, size = map(float, content.split(";"))
       
        R_FPS  += fps  / len(scenes)
        R_MSE  += mse  / len(scenes)
        R_PSNR += psnr / len(scenes)
        R_SSIM += ssim / len(scenes)
        R_SIZE += size / len(scenes)

        path = f"res/Distill{model}/{scene}/Distill{model}_benchmark.csv"
        with open(path, "r") as fp: _, content = fp.read().split("\n")
        fps, mse, psnr, ssim, size = map(float, content.split(";"))
        
        D_FPS  += fps  / len(scenes)
        D_MSE  += mse  / len(scenes)
        D_PSNR += psnr / len(scenes)
        D_SSIM += ssim / len(scenes)
        D_SIZE += size / len(scenes)

    n_table += f"{            model:{l}s} & {N_FPS:.2f} & {N_MSE:.2e} & {N_PSNR:.2f} & {N_SSIM:.2f} & {N_SIZE:.2f} \\\\ \n"
    r_table += f"{'Reptile' + model:{l}s} & {R_FPS:.2f} & {R_MSE:.2e} & {R_PSNR:.2f} & {R_SSIM:.2f} & {R_SIZE:.2f} \\\\ \n"
    d_table += f"{'Distill' + model:{l}s} & {D_FPS:.2f} & {D_MSE:.2e} & {D_PSNR:.2f} & {D_SSIM:.2f} & {D_SIZE:.2f} \\\\ \n"

print("\\toprule")
print("\\textbf{Architecture} & \\textbf{Rendering (FPS)} $\\uparrow$ & \\textbf{MSE} $\\downarrow$ & \\textbf{PSNR} $\\uparrow$ & \\textbf{SSIM} $\\uparrow$ & \\textbf{Size (MB)} $\\downarrow$ \\\\")
print(f"\\midrule\n{n_table}\\midrule\n{r_table}\\midrule\n{d_table}\\bottomrule\n")
