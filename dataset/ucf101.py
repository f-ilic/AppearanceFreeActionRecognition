import glob
import os
from pytorchvideo.data.encoded_video import EncodedVideo
import random
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset



class UCF101(VisionDataset):
    def __init__(self, root, annotation_path,
                fold=1, train=True, transform=None, inverse_normalise=None, return_paths=False, config=None):
        super(UCF101, self).__init__(root)
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))
        
        self.fps = 30
        extensions = ('avi',)
        self.fold = fold
        self.train = train
        self.cfg = config

        classes = list(sorted(list_dir(root)))
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}
        self.samples = make_dataset(self.root, self.class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        
        self.indices = self._select_fold(video_list, annotation_path, fold, train)
        self.video_clips = [video_list[i] for i in self.indices]

        self.transform = transform
        self.inverse_normalise = inverse_normalise
        self.num_classes = len(classes)
        self.return_paths = return_paths

    def _select_fold(self, video_list, annotation_path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(annotation_path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [x[0] for x in data]
            selected_files.extend(data)
        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i][len(self.root) + 1:] in selected_files]
        return indices

    def __len__(self):
        return len(self.video_clips)

    def __getitem__(self, idx):
        video_path, label = self.samples[self.indices[idx]]
    
        video = EncodedVideo.from_path(video_path)
        
        
        if self.cfg == None or self.cfg['num_frames']==1: # this gets the whole clip
            start_sec = 0 
            end_sec = video.duration
        else: # this gets the right amount of the clip, based on what num_frames x sample_rate the network takes. 
              # this is best understood in conjuction with transforms.transform.ucf101transform
            num_frames = self.cfg['num_frames']
            sample_rate = self.cfg['sample_rate']
            clip_duration = (num_frames * sample_rate)/self.fps

            start_sec = random.uniform(0, video.duration-clip_duration)
            end_sec = start_sec+clip_duration

        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

        if self.transform is not None:
            video = self.transform(video_data)
            
        if self.return_paths == True:
            return video['video'], label, video_path
        else:
            return video['video'], label
