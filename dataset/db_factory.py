from config import build_cfg
from dataset.TwoStreamucf101 import WrapForTwoStream
from dataset.ucf101 import UCF101
from dataset.db_stats import db_stats
import os
from matplotlib import pyplot as plt
import pickle
from transforms.transform import FLOWucf101transform, trial_transform, ucf101transform, InverseNormalizeVideo, ucf101transform_singleframe
from utils.VideoTensorViewer import VideoTensorViewer
import torch
from torch.utils.data import DataLoader

from utils.info_print import print_data_augmentation_transform


def DBfactory2Stream(dbname1, dbname2, train, fold, rgbcfg, flowcfg):

    rgb_root = f'data/{dbname1}'
    split_rgb_root = f'data/splits/{dbname1}'

    flow_root = f'data/{dbname2}'
    split_flow_root = f'data/splits/{dbname2}'

    
    Trgb = ucf101transform(rgbcfg)
    T_inv = InverseNormalizeVideo(rgbcfg)

    Tflow = FLOWucf101transform(flowcfg)
    Tflow_inv = InverseNormalizeVideo(flowcfg)

    db = WrapForTwoStream(rgb_root, split_rgb_root, flow_root, split_flow_root,
    fold, train, Trgb, T_inv, Tflow, Tflow_inv, False, rgbcfg, flowcfg)

    return db 

def DBfactory(dbname, train, fold, config, is_human_trial=False):
    """
    This Database Factory returns datasets with a fixed number of frames per clip sampled at the same fps
    """


    return_paths = is_human_trial

    if is_human_trial or config==None:
        T = trial_transform()
        T_inv = None
    else:
        if config['num_frames'] == 1:
            T = ucf101transform_singleframe(config)
        elif config['num_frames'] > 1:
            T = ucf101transform(config)
        else:
            raise ValueError("something wrong with config['num_frames']")
        
        T_inv = InverseNormalizeVideo(config)

    if dbname in ['ucf101', 'ucf101flow', 'afd101', 'afd101flow', 'ucf5', 'ucf5flow', 'afd5', 'afd5flow']:
        data_root = f'data/{dbname}'
        split_root = f'data/splits/{dbname}'
        db = UCF101(data_root, split_root, fold, train, T, T_inv, return_paths, config)
    else:
        raise ValueError(f"Invalid Database name {dbname}")

    return db


def show_single_videovolume():
    cfg = build_cfg()
    cfg['num_frames']=30
    cfg['sampling_rate']=1
    cfg = None
    normal = DBfactory('ucf5', train=True, fold=1, config=cfg)

    for s, lbl in normal:
        sample_normal = normal.inverse_normalise(s)
        VideoTensorViewer(sample_normal)
        plt.show(block=True)


if __name__ == "__main__":
    cfg = build_cfg()
    cfg['datasetname'] = 'ucf101'
    cfg['num_frames']=30
    cfg['sampling_rate']=1
    cfg['architecture'] = 'x3d_s'
    normal = DBfactory2Stream('ucf101', 'ucf101flow', train=True, fold=1, rgbcfg=cfg, flowcfg=cfg)
    loader = DataLoader(normal, 1, num_workers=4, shuffle=True)

    for rgb_sample, flow_sample, lbl in loader:
        rgb_sample = rgb_sample.squeeze()
        flow_sample = flow_sample.squeeze()
        rgb_normal = loader.dataset.rgb_dataset.inverse_normalise(rgb_sample)
        flow_normal = loader.dataset.flow_dataset.inverse_normalise(flow_sample)

        VideoTensorViewer(torch.cat([rgb_normal, flow_normal], dim=2))
        plt.show(block=True)
    
