
import torch
import torch.nn.functional as F
from torch import nn
from util.misc import NestedTensor
from .vgg_ import vgg16_bn

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int):
        super().__init__()
        features = list(backbone.features.children())
        self.body1 = nn.Sequential(*features[:13]); self.body2 = nn.Sequential(*features[13:23])
        self.body3 = nn.Sequential(*features[23:33]); self.body4 = nn.Sequential(*features[33:43])
        self.num_channels = num_channels
        if not train_backbone:
            for name, parameter in backbone.named_parameters(): parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
        out = {}
        xs = self.body1(xs); out[0] = xs
        xs = self.body2(xs); out[1] = xs
        xs = self.body3(xs); out[2] = xs
        xs = self.body4(xs); out[3] = xs
        return out

class Backbone_VGG(BackboneBase):
    def __init__(self, name: str, train_backbone: bool):
        backbone = vgg16_bn(pretrained=True)
        super().__init__(backbone, train_backbone, 512)

def build_backbone(args):
    train_backbone = args.lr_backbone > 0
    return Backbone_VGG(args.backbone, train_backbone)
