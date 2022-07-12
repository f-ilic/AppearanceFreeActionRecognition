import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def set_requires_grad(model, flag: bool):
    for param in model.parameters():
        param.requires_grad = flag

def load_weights(model, optimizer, weights_path, verbose=True):
    if weights_path != None and weights_path != 'None':
        pth = torch.load(weights_path)
        model.load_state_dict(pth['state_dict'])

        if verbose:
            print(f"Load Model from: {weights_path}")
        
        if "name" in pth:
            model.name = pth['name']

        if optimizer != None:
            if verbose:
                print(f"Load Optimizer from: {weights_path}")
            optimizer.load_state_dict(pth['optimizer'])
            # hacky garbage from https://github.com/pytorch/pytorch/issues/2830
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
    else:
        if verbose:
            print(f"Nothing to Load")