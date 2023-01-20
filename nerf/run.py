import numpy as np
import os
import torch
import traceback

from enum import auto, Enum
from nerf.benchmark import NeRFBenchmarker
from nerf.core.nn import NeRFFactory, NeRFModel, PositionalEncoder
from nerf.core.render import Samples
from nerf.data.irm import IRM
from nerf.data.turntable import TurnTable
from nerf.data.utils import BoundingBox, CameraPlanes
from nerf.distill import NeRFDistiller
from nerf.infer import NeRFInferer
from nerf.irm import NeRFIRM
from nerf.train import NeRFTrainer
from torch.multiprocessing import Queue


BBOX    = True
REFINED = True


class WorkType(Enum):
    TRAIN: int = auto()
    REPTILE: int = auto()
    DISTILL: int = auto()
    STOP: int = auto()


class Worker:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def train(
        self,
        nerf: NeRFModel,
        phi_p: PositionalEncoder,
        phi_d: PositionalEncoder,
        blender: str,
        scene: str,
        bbox: BoundingBox,
        threshold: float,
        batch_size: int,
    ) -> None:
        os.makedirs(f"res/{nerf.__class__.__name__}/{scene}", exist_ok=True)

        m_path = f"res/{nerf.__class__.__name__}/{scene}/{nerf.__class__.__name__}.pt"
        r_path = f"res/{nerf.__class__.__name__}/{scene}/{nerf.__class__.__name__}.png"
        g_path = f"res/{nerf.__class__.__name__}/{scene}/{nerf.__class__.__name__}.gif"
        i_path = f"res/{nerf.__class__.__name__}/{scene}/{nerf.__class__.__name__}.irm.png"
        b_path = f"res/{nerf.__class__.__name__}/{scene}/{nerf.__class__.__name__}_benchmark.csv"


        trainer = NeRFTrainer(nerf, phi_p, phi_d, Samples(64, 64)).to(self.device)
        trainer.set_data(blender, scene=scene, step=1, scale=.5, batch_size=batch_size)
        trainer.fit(20, lr=5e-4, perturb=True, log=5, model_path=m_path, render_path=r_path)

        inferer = NeRFInferer(nerf, phi_p, phi_d).to(self.device)
        inferer.set_data(TurnTable(800, 800, .5 * 800 / np.tan(.5 * 0.69), 40, 4), batch_size)
        inferer.render(15, CameraPlanes(2., 6.), Samples(64, 64), g_path)

        bbox = IRM.refine(bbox, 256, nerf, phi_p, phi_d, 16, batch_size, threshold=threshold)

        irm = NeRFIRM(nerf, phi_p, phi_d).to(self.device)
        irm.set_data(bbox, 16)
        irm.scan(8, 64, batch_size, i_path)

        benchmarker = NeRFBenchmarker(nerf, phi_p, phi_d, Samples(64, 64)).to(self.device)
        benchmarker.set_data(blender, scene, step=1, scale=.5, batch_size=batch_size)
        ben = benchmarker.benchmark(bbox=bbox)

        with open(b_path, "w") as bfp:
            bfp.writelines([
                f"fps;mse;psnr;ssim;size\n",
                f"{ben.fps};{ben.mse};{ben.psnr};{ben.ssim};{ben.size}",
            ])

    def reptile(
        self,
        nerf: NeRFModel,
        phi_p: PositionalEncoder,
        phi_d: PositionalEncoder,
        blender: str,
        scene: str,
        bbox: BoundingBox,
        threshold: float,
        batch_size: int,
    ) -> None:
        os.makedirs(f"res/{nerf.__class__.__name__}/{scene}", exist_ok=True)

        m_path = f"res/Reptile{nerf.__class__.__name__}/{scene}/Reptile{nerf.__class__.__name__}.pt"
        r_path = f"res/Reptile{nerf.__class__.__name__}/{scene}/Reptile{nerf.__class__.__name__}.png"
        g_path = f"res/Reptile{nerf.__class__.__name__}/{scene}/Reptile{nerf.__class__.__name__}.gif"
        i_path = f"res/Reptile{nerf.__class__.__name__}/{scene}/Reptile{nerf.__class__.__name__}.irm.png"
        b_path = f"res/Reptile{nerf.__class__.__name__}/{scene}/Reptile{nerf.__class__.__name__}_benchmark.csv"

        trainer = NeRFTrainer(nerf, phi_p, phi_d, Samples(64, 64)).to(self.device)
        trainer.set_data(blender, scene=scene, step=1, scale=.5, batch_size=batch_size)
        trainer.reptile(16, lr=5e-4, perturb=True, model_path=m_path, render_path=r_path)
        trainer.fit(20, lr=5e-4, perturb=True, log=5, model_path=m_path, render_path=r_path)

        inferer = NeRFInferer(nerf, phi_p, phi_d).to(self.device)
        inferer.set_data(TurnTable(800, 800, .5 * 800 / np.tan(.5 * 0.69), 40, 4), batch_size)
        inferer.render(15, CameraPlanes(2., 6.), Samples(64, 64), g_path)

        bbox = IRM.refine(bbox, 256, nerf, phi_p, phi_d, 16, batch_size, threshold=threshold)
        
        irm = NeRFIRM(nerf, phi_p, phi_d).to(self.device)
        irm.set_data(bbox, 16)
        irm.scan(8, 64, batch_size, i_path)

        benchmarker = NeRFBenchmarker(nerf, phi_p, phi_d, Samples(64, 64)).to(self.device)
        benchmarker.set_data(blender, scene, step=1, scale=.5, batch_size=batch_size)
        ben = benchmarker.benchmark(bbox=bbox)

        with open(b_path, "w") as bfp:
            bfp.writelines([
                f"fps;mse;psnr;ssim;size\n",
                f"{ben.fps};{ben.mse};{ben.psnr};{ben.ssim};{ben.size}",
            ])

    def distill(
        self,
        tnerf: NeRFModel,
        snerf: NeRFModel,
        phi_p: PositionalEncoder,
        phi_d: PositionalEncoder,
        blender: str,
        scene: str,
        bbox: BoundingBox,
        threshold: float,
        batch_size: int,
    ) -> None:
        os.makedirs(f"res/Distill{snerf.__class__.__name__}/{scene}", exist_ok=True)

        m_path = f"res/Distill{snerf.__class__.__name__}/{scene}/Distill{snerf.__class__.__name__}.pt"
        r_path = f"res/Distill{snerf.__class__.__name__}/{scene}/Distill{snerf.__class__.__name__}.png"
        g_path = f"res/Distill{snerf.__class__.__name__}/{scene}/Distill{snerf.__class__.__name__}.gif"
        i_path = f"res/Distill{snerf.__class__.__name__}/{scene}/Distill{snerf.__class__.__name__}.irm.png"
        b_path = f"res/Distill{snerf.__class__.__name__}/{scene}/Distill{snerf.__class__.__name__}_benchmark.csv"

        tnerf = tnerf.to(self.device)
        phi_p = phi_p.to(self.device)
        phi_d = phi_d.to(self.device)
        
        if REFINED: bbox = IRM.refine(bbox, 256, tnerf, phi_p, phi_d, 16, batch_size, threshold=threshold)
        if BBOX: res = 16
        else:    res =  1

        irm = IRM(bbox, res, samples=500_000)
        irm.scan(tnerf, phi_p, phi_d, 256, batch_size)

        delta = (6. - 2.) / 128

        distiller = NeRFDistiller(tnerf, snerf, phi_p, phi_d).to(self.device)
        distiller.set_data(irm, batch_size=batch_size)
        distiller.distill(200, lr=1e-3, delta=delta, model_path=m_path)

        trainer = NeRFTrainer(snerf, phi_p, phi_d, Samples(64, 64)).to(self.device)
        trainer.set_data(blender, scene=scene, step=1, scale=.5, batch_size=batch_size)
        trainer.fit(20, lr=5e-4, perturb=True, log=5, model_path=m_path, render_path=r_path)

        inferer = NeRFInferer(snerf, phi_p, phi_d).to(self.device)
        inferer.set_data(TurnTable(800, 800, .5 * 800 / np.tan(.5 * 0.69), 40, 4), batch_size)
        inferer.render(15, CameraPlanes(2., 6.), Samples(64, 64), g_path, bbox=bbox)

        irm = NeRFIRM(snerf, phi_p, phi_d).to(self.device)
        irm.set_data(bbox, 16)
        irm.scan(8, 64, batch_size, i_path)

        benchmarker = NeRFBenchmarker(snerf, phi_p, phi_d, Samples(64, 64)).to(self.device)
        benchmarker.set_data(blender, scene, step=1, scale=.5, batch_size=batch_size)
        ben = benchmarker.benchmark(bbox=bbox)

        with open(b_path, "w") as bfp:
            bfp.writelines([
                f"fps;mse;psnr;ssim;size\n",
                f"{ben.fps};{ben.mse};{ben.psnr};{ben.ssim};{ben.size}",
            ])

    def run(self, queue: Queue) -> None:
        try:
            while data := queue.get():
                work_type, model, worker_args = data
                
                if work_type == WorkType.STOP:
                    print(f"[Device|{self.device}] Done")
                    return None
                
                phi_p = PositionalEncoder(3, 10)
                phi_d = PositionalEncoder(3,  4)

                nerf = getattr(NeRFFactory, model)
                nerf = nerf(phi_p.o_dim, phi_d.o_dim)

                if work_type == WorkType.TRAIN:
                    self.train(nerf, phi_p, phi_d, **worker_args)

                elif work_type == WorkType.REPTILE:
                    self.reptile(nerf, phi_p, phi_d, **worker_args)

                elif work_type == WorkType.DISTILL:
                    ckpt = f"res/ReptileNeRF/{worker_args['scene']}/ReptileNeRF.pt"
                    if not os.path.isfile(ckpt):
                        raise FileNotFoundError(f"Teacher Model {ckpt} Not Found")

                    student = nerf
                    teacher = NeRFFactory.NeRF(phi_p.o_dim, phi_d.o_dim)
                    teacher.load_state_dict(torch.load(ckpt, map_location="cpu")["nerf"])

                    self.distill(teacher, student, phi_p, phi_d, **worker_args)

                else: raise ValueError("Unknown WorkType (available types: 'train', 'reptile', 'distill')")

        except Exception:
            print(f"[Device|{self.device}]", traceback.format_exc())
            return None


if __name__ == "__main__":
    from argparse import ArgumentParser
    from torch.multiprocessing import Process, set_start_method


    torch.backends.cudnn.benchmark = True
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.autograd.set_detect_anomaly(mode=False)
    set_start_method("spawn")


    parser = ArgumentParser()
    parser.add_argument("-w", "--work",  type=str,                help="Work to run", required=True )
    parser.add_argument("-b", "--batch", type=int, default=4_096, help="Batch size",  required=False)
    parser.add_argument("-l", "--limit", type=int, default=2,     help="GPU limit",   required=False)
    parser.add_argument("-s", "--size",  type=int, default=20,    help="BBOX size",   required=False)
    args = parser.parse_args()


    devices = min(torch.cuda.device_count(), args.limit)
    devices = [torch.device(f"cuda:{i}") for i in range(devices)]
    workers = [Worker(device) for device in devices]
    queue = Queue()

    # models = ["NeRF", "TinyNeRF", "MicroNeRF", "NanoNeRF"]
    models = ["NanoNeRF"]

    bbox_size = -args.size, args.size
    bbox = BoundingBox(*bbox_size, *bbox_size, *bbox_size)

    blender = "dataset/blender"
    # scenes     = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    # thresholds = [   1e-2,    1e-2,    1e-2,     1e-2,   1e-2,        1e-2,  1e-2,   1e-2]
    scenes     = ["hotdog"]
    thresholds = [    1e-2]

    for scene, threshold in zip(scenes, thresholds):
        wargs = {"blender": blender, "scene": scene, "batch_size": args.batch, "bbox": bbox, "threshold": threshold}
        for model in models: queue.put((WorkType[args.work.upper()], model, wargs))
    
    for _ in workers: queue.put((WorkType.STOP, None, None))

    processes = [Process(target=worker.run, args=(queue, )) for worker in workers]
    for process in processes: process.start()
    for process in processes: process.join()
    for process in processes: process.close()
    
    torch.cuda.empty_cache()
    queue.close()