from tkinter import Pack
import torch
import random
from dataset.db_stats import db_stats

from torchvision.transforms import (
    Compose,
    Lambda,
    ColorJitter,
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomHorizontalFlipVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
    Permute,
    RandomShortSideScale,
)

from torchvision.transforms._transforms_video import NormalizeVideo

valid_models = {
    "TwoStreamSimonyan_rgb": {
        "crop_size": 224,
        "num_frames": 4,
        "sample_rate": 2,
    },
    "TwoStreamSimonyan_flow": {
        "crop_size": 224,
        "num_frames": 10,
        "sample_rate": 4,
    },
    "x3d_xs": {
        "crop_size": 182,
        "num_frames": 4,
        "sample_rate": 12,
    },
    "x3d_s": {
        "crop_size": 182,
        "num_frames": 13,
        "sample_rate": 6,
    },
    "x3d_m": {
        "crop_size": 224,
        "num_frames": 16,
        "sample_rate": 5,
    },
    "x3d_l": {
        "crop_size": 312,
        "num_frames": 16,
        "sample_rate": 5,
    },
    "slowfast_r50": {
        "crop_size": 224,
        # "num_frames": 32,
        # "sample_rate": 2,
        "frames_per_second": 30,
        "slowfast_alpha": 4,
    },
    "slow_r50": {
        "crop_size": 224,
        # "num_frames": 8,
        # "sample_rate": 8,
    },
    "fast_r50": {
        "crop_size": 224,
        # "num_frames": 32,
        # "sample_rate": 2,
    },
    "i3d_r50": {
        "crop_size": 224,
        # "num_frames": 8,
        # "sample_rate": 8,
    },
    "c2d_r50": {
        "crop_size": 224,
        # "num_frames": 8,
        # "sample_rate": 8,
    },
    "r2plus1d_r50": {
        "crop_size": 224,
        # "num_frames": 16,
        # "sample_rate": 4,
    },
    "mvit_base_16x4": {
        "crop_size": 224,
        # "num_frames": 16,
        # "sample_rate": 4,
    },
}


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def ucf101transform(cfg):
    num_frames = cfg["num_frames"]
    mean = db_stats[cfg["datasetname"]]["mean"]
    std = db_stats[cfg["datasetname"]]["std"]
    crop_size = valid_models[cfg["architecture"]]["crop_size"]
    pack = lambda x: x
    if cfg["architecture"] == "slowfast_r50":
        alpha = valid_models[cfg["architecture"]]["slowfast_alpha"]
        pack = PackPathway(alpha)

    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                Permute((1, 0, 2, 3)),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                Permute((1, 0, 2, 3)),
                NormalizeVideo(mean, std),
                RandomShortSideScale(min_size=crop_size, max_size=320),
                CenterCropVideo(crop_size),
                RandomHorizontalFlipVideo(0.5),
                pack,
            ]
        ),
    )


def FLOWucf101transform(cfg):
    num_frames = cfg["num_frames"]
    mean = db_stats[cfg["datasetname"]]["mean"]
    std = db_stats[cfg["datasetname"]]["std"]
    crop_size = valid_models[cfg["architecture"]]["crop_size"]
    pack = lambda x: x
    if cfg["architecture"] == "slowfast_r50":
        alpha = valid_models[cfg["architecture"]]["slowfast_alpha"]
        pack = PackPathway(alpha)

    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                RandomShortSideScale(min_size=crop_size, max_size=320),
                CenterCropVideo(crop_size),
                RandomHorizontalFlipVideo(0.5),
                pack,
            ]
        ),
    )


class SelectRandomSingleFrame(torch.nn.Module):
    def __init__():
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C, T, H, W = x.shape
        idx = random.randint(0, T - 1)
        return x[:, idx, ...].unsqueeze(1)


def ucf101transform_singleframe(cfg):
    mean = db_stats[cfg["datasetname"]]["mean"]
    std = db_stats[cfg["datasetname"]]["std"]
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                # UniformTemporalSubsample(1),
                SelectRandomSingleFrame(),
                Lambda(lambda x: x / 255.0),
                Permute((1, 0, 2, 3)),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                Permute((1, 0, 2, 3)),
                NormalizeVideo(mean, std),
                CenterCropVideo(224),
                Lambda(lambda x: x.squeeze()),
            ]
        ),
    )


def trial_transform():
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                Lambda(lambda x: x / 255.0),
                CenterCropVideo(224),
            ]
        ),
    )


def InverseNormalizeVideo(cfg):
    mean = db_stats[cfg["datasetname"]]["mean"]
    std = db_stats[cfg["datasetname"]]["std"]
    return NormalizeVideo(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std],
    )
