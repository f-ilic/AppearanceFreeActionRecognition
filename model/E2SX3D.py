import torch
import torch.nn as nn

from utils.model_utils import load_weights, set_requires_grad

class E2SX3D(nn.Module):
    def init_backbone(self, arch, num_classes, pretrained=False):
        model = torch.hub.load('facebookresearch/pytorchvideo', arch, pretrained=pretrained)
        in_features = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_features, num_classes)
        model.blocks[-1].activation = nn.Identity()
        set_requires_grad(model, True)
        return model

    def __init__(self, rgb_arch, flow_arch, num_classes, weights=None,pretrained=False):
        super(E2SX3D, self).__init__()
        self.rgb_arch = rgb_arch
        self.flow_arch = flow_arch
        self.rgbstream = self.init_backbone(rgb_arch, num_classes, pretrained)
        self.flowstream = self.init_backbone(flow_arch, num_classes, pretrained)
        self.head = nn.Linear(2*num_classes,num_classes)
        load_weights(self, None, weights)

    def forward(self, rgbs, flows):
        fs = self.flowstream(flows)
        aps = self.rgbstream(rgbs)
        x = torch.cat([fs, aps], dim=1)
        x = self.head(x)
        return x

