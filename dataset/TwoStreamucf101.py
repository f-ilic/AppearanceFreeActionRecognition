import glob
import os
from pytorchvideo.data.encoded_video import EncodedVideo
import random
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset

from dataset.ucf101 import UCF101


class WrapForTwoStream(VisionDataset):
    def __init__(self, imgroot,annotation_path_rgb, flowroot, annotation_path_flow,
                fold=1, train=True, transform=None, inverse_normalise=None, FLOWtransform=None, FLOWinverse_normalise=None ,return_paths=False, rgbcfg=None, flowcfg=None):

        self.rgb_dataset = UCF101(imgroot, annotation_path_rgb, fold, train, transform, inverse_normalise, return_paths, rgbcfg)
        self.flow_dataset = UCF101(flowroot, annotation_path_flow, fold, train, FLOWtransform, FLOWinverse_normalise, return_paths, flowcfg)

        assert(len(self.rgb_dataset.video_clips) == len(self.flow_dataset.video_clips))
        self.class_to_idx   = self.rgb_dataset.class_to_idx
        self.idx_to_class   = self.rgb_dataset.idx_to_class
        self.classes        = self.rgb_dataset.classes     
        self.num_classes    = self.rgb_dataset.num_classes 

    def __len__(self):
        return len(self.rgb_dataset.video_clips)

    def internal__getitem__(self, dataset, idx):
        video_path, label = dataset.samples[dataset.indices[idx]]
        video = EncodedVideo.from_path(video_path)

        num_frames = dataset.cfg['num_frames']
        sample_rate = dataset.cfg['sample_rate']
        clip_duration = (num_frames * sample_rate)/dataset.fps

        start_sec = random.uniform(0, video.duration-clip_duration)
        end_sec = start_sec+clip_duration

        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        if dataset.transform is not None:
            video = dataset.transform(video_data)
        
        if dataset.return_paths == True:
            return video['video'], label, video_path
        else:
            return video['video'], label

    def __getitem__(self, idx):
        rgb, rgb_lbl = self.internal__getitem__(self.rgb_dataset, idx)
        flow, flow_lbl = self.internal__getitem__(self.flow_dataset, idx)

        dataset = self.rgb_dataset
        rgb_video_path, rgb_label = dataset.samples[dataset.indices[idx]]
        rgb_video = EncodedVideo.from_path(rgb_video_path)

        num_frames = dataset.cfg['num_frames']
        sample_rate = dataset.cfg['sample_rate']
        clip_duration = (num_frames * sample_rate)/dataset.fps
        start_sec = random.uniform(0, rgb_video.duration-clip_duration)
        end_sec = start_sec+clip_duration

        video_data = rgb_video.get_clip(start_sec=start_sec, end_sec=end_sec)
        if dataset.transform is not None:
            rgb_video = dataset.transform(video_data)

        dataset = self.flow_dataset
        flow_video_path, rgb_label = dataset.samples[dataset.indices[idx]]
        flow_video = EncodedVideo.from_path(flow_video_path)

        num_frames = dataset.cfg['num_frames']
        sample_rate = dataset.cfg['sample_rate']
        clip_duration = (num_frames * sample_rate)/dataset.fps
        end_sec = start_sec+clip_duration

        video_data = flow_video.get_clip(start_sec=start_sec, end_sec=end_sec)
        if dataset.transform is not None:
            flow_video = dataset.transform(video_data)

        assert(rgb_lbl == flow_lbl)
        return rgb_video['video'], flow_video['video'], rgb_label