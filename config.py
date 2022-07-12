import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--datasetname', type=str)

parser.add_argument('--weights_path', type=str, default=None, help='Path to weights to use')
parser.add_argument('--num_epochs', type=int, default=80, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--num_frames', type=int, default=16, help='Data param; Length of frames of each video')
parser.add_argument('--sample_rate', type=int, default=3, help='Sampling Rate within each video')
parser.add_argument('--lr', type=float, default=3e-4, help='Inital learning rate (Default optimizer: AdamW)')
parser.add_argument('-pretrained', action='store_false')
parser.add_argument('-train_backbone', action='store_true')

cfg = dict()


def build_cfg():
    args = parser.parse_args()
    cfg = args.__dict__.copy()
    print(f'-----------------------------------\n')
    print(f'Running Config:')
    for k,v in cfg.items():
        print(f"{k}: {v}")
    print(f'-----------------------------------\n')
    return cfg
