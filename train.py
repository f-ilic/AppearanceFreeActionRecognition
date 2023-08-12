import sys
import torch
from os.path import join

from torch import nn, optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from config import cfg, parser, build_cfg
from dataset.db_factory import DBfactory, DBfactory2Stream
from utils.TwoStreamTrainer import TwoStreamTrainer
from simulation.simulation import Simulation
from model.E2SX3D import E2SX3D
from utils.info_print import print_data_augmentation_transform, print_learnable_params
from utils.model_utils import set_requires_grad, load_weights
from pytorchvideo.models import x3d
from transforms.transform import valid_models

parser.add_argument("--rgbarch", type=str)
parser.add_argument("--flowarch", type=str)
parser.add_argument("--datasetname_motionpath", type=str)
parser.add_argument("--num_workers", type=int)
cfg = build_cfg()

import matplotlib

matplotlib.use("Agg")


def build_model_name(cfg):
    rgbarch = cfg["rgbarch"]
    flowarch = cfg["flowarch"]

    rgb_sampleing_cfg = valid_models[rgbarch]
    flow_sampling_cfg = valid_models[flowarch]

    name = f"E2SX3D__rgbStream__{rgbarch}__{rgb_sampleing_cfg['num_frames']}x{rgb_sampleing_cfg['sample_rate']}____flowStream__{flowarch}__{flow_sampling_cfg['num_frames']}x{flow_sampling_cfg['sample_rate']}"
    return name


def main():
    torch.backends.cudnn.benchmark = True
    datasetname = cfg["datasetname"]
    datasetname_motionpath = cfg["datasetname_motionpath"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    rgb_cfg = {
        "architecture": cfg["rgbarch"],
        "datasetname": cfg["datasetname"],
        "num_frames": valid_models[cfg["rgbarch"]]["num_frames"],
        "sample_rate": valid_models[cfg["rgbarch"]]["sample_rate"],
    }

    flow_cfg = {
        "architecture": cfg["flowarch"],
        "datasetname": cfg["datasetname_motionpath"],
        "num_frames": valid_models[cfg["flowarch"]]["num_frames"],
        "sample_rate": valid_models[cfg["flowarch"]]["sample_rate"],
    }

    train_dataset = DBfactory2Stream(
        datasetname,
        datasetname_motionpath,
        train=True,
        fold=1,
        rgbcfg=rgb_cfg,
        flowcfg=flow_cfg,
    )
    test_dataset = DBfactory2Stream(
        datasetname,
        datasetname_motionpath,
        train=False,
        fold=1,
        rgbcfg=rgb_cfg,
        flowcfg=flow_cfg,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size, num_workers=num_workers, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size, num_workers=num_workers, shuffle=True, drop_last=True
    )

    # ----------------- Setup Model & Load weights if supplied -----------------
    model = E2SX3D(
        cfg["rgbarch"],
        cfg["flowarch"],
        train_dataset.num_classes,
        cfg["weights_path"],
        cfg["pretrained"],
        cfg["train_rgbbackbone"],
        cfg["train_flowbackbone"],
    )

    model.name = build_model_name(cfg)
    model.configuration = cfg
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    criterion = nn.CrossEntropyLoss().cuda()
    model = model.cuda()

    sim_name = f"{cfg['datasetname']}__{cfg['datasetname_motionpath']}/{model.name}"
    with Simulation(sim_name=sim_name, output_root="runs") as sim:
        cfg["executed"] = f'python {" ".join(sys.argv)}'
        print(f'Running: {cfg["executed"]}\n\n\n')
        print_learnable_params(model)
        print(f"Begin training: {model.name}")

        writer = SummaryWriter(join(sim.outdir, "tensorboard"))
        trainer = TwoStreamTrainer(sim)

        # -------------- MAIN TRAINING LOOP  ----------------------
        for epoch in range(cfg["num_epochs"]):
            # checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
            # sim.save_pytorch(checkpoint, epoch=epoch)

            trainer.do(
                "train",
                model,
                train_dataloader,
                epoch,
                criterion,
                optimizer,
                writer,
                log_video=False,
            )

            with torch.no_grad():
                if epoch % 10 == 0 or epoch == cfg["num_epochs"] - 1:
                    trainer.do(
                        "test",
                        model,
                        test_dataloader,
                        epoch,
                        criterion,
                        None,
                        writer,
                        log_video=False,
                    )
                    checkpoint = {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    sim.save_pytorch(checkpoint, epoch=epoch)

        print(f"\nRun {sim.outdir} finished\n")

        writer.close


if __name__ == "__main__":
    main()
