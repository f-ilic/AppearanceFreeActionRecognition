import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--datasetname", type=str)

parser.add_argument("--weights_path", type=str, default=None)
parser.add_argument("--num_epochs", type=int, default=80)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("-pretrained", action="store_false")
parser.add_argument("-train_backbone", action="store_true")

cfg = dict()


def build_cfg():
    args = parser.parse_args()
    cfg = args.__dict__.copy()
    print(f"-----------------------------------\n")
    print(f"Running Config:")
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print(f"-----------------------------------\n")
    return cfg
